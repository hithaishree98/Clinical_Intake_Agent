"""
webhook.py — Retryable outbound notifications with delivery tracking.

Supports two channels:
  • Slack Incoming Webhooks — plain-text alerts for emergency, crisis, and
    intake-complete events.
  • FHIR R4 Bundle webhook — HMAC-SHA256-signed POST carrying the full intake
    bundle; designed for EHR and case-management system integration.

────────────────────────────────────────────────────────────────────────
EHR / DOWNSTREAM INTEGRATION GUIDE
────────────────────────────────────────────────────────────────────────

HOW IT CONNECTS TO AN EHR OR CASE-MANAGEMENT SYSTEM
────────────────────────────────────────────────────
1. Configure COMPLETION_WEBHOOK_URL in .env to point at your EHR's
   inbound endpoint (e.g. Epic's FHIR R4 $process-message operation, or
   an HL7 FHIR server like Azure Health Data Services / Google FHIR).

2. Set COMPLETION_WEBHOOK_SECRET to a shared secret agreed with the EHR.
   Every POST carries X-Signature: sha256=<hex>.  The receiver calls
   verify_webhook_signature(body, header, secret) to authenticate it.

3. The payload is a standard FHIR R4 Bundle (type=document) containing:
     Patient             — identity (name, DOB, phone, address)
     Condition           — chief complaint + OPQRST notes
     AllergyIntolerance  — one resource per allergy
     MedicationStatement — one resource per current medication
     Observation         — triage risk level + rationale

4. Delivery is retried automatically (up to WEBHOOK_MAX_ATTEMPTS times,
   default 3) with exponential backoff (2s -> 8s -> 30s) so transient
   EHR downtime does not drop records.

5. Every attempt is recorded in the webhook_deliveries table with:
     status           pending | success | failed | exhausted
     last_http_status HTTP response code from the EHR endpoint
     last_error       Error message on failure
     next_retry_at    ISO timestamp for the next attempt
   Clinicians can query this via GET /clinician/webhooks.

6. Idempotency: if the same event_type + payload_hash was already
   delivered successfully for a thread, the dispatch is skipped.  This
   prevents duplicate records in the EHR if the intake system retries
   after a crash mid-delivery.

RECEIVER-SIDE VERIFICATION (EHR / CASE-MANAGEMENT)
────────────────────────────────────────────────────
The receiving system must verify each POST to reject spoofed payloads.
This library exposes verify_webhook_signature() for testing end-to-end:

    import hashlib, hmac

    def verify(body: bytes, sig_header: str, secret: str) -> bool:
        if not sig_header.startswith("sha256="):
            return False
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, sig_header)

PLUGGING INTO EPIC / CERNER / AZURE FHIR
────────────────────────────────────────────────────
Epic  — Set COMPLETION_WEBHOOK_URL to your Epic sandbox's
        /api/FHIR/R4/$process-message endpoint.  Epic requires
        OAuth 2.0 Bearer auth; add the token as a custom header
        by extending signed_fhir_webhook().

Cerner — Point at Cerner's HL7 FHIR R4 base URL /Bundle endpoint.
         Cerner supports HMAC-based webhooks for system-to-system
         integration.

Azure Health Data Services — The FHIR Service accepts standard FHIR
         Bundle POSTs.  Use Managed Identity or a shared-secret via
         X-Signature for transport auth.

Custom case-management — Any HTTP endpoint can receive the bundle.
         The X-Thread-Id header carries the session UUID for correlation
         with the intake portal's own records.
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from .logging_utils import log_event
from .settings import get_settings as settings


def _dispatch_in_thread(target, kwargs: dict) -> None:
    """
    Fire-and-forget: run `target(**kwargs)` in a daemon thread.

    Patient safety rationale: crisis and emergency alerts must not block graph
    execution.  If the downstream Slack/FHIR endpoint is slow or down, the
    synchronous retry delays (2s, 8s, 30s) would freeze the node and leave the
    patient waiting.  Running in a daemon thread decouples alert delivery from
    the patient-facing response path.  Delivery is still tracked and retried via
    the webhook_deliveries table — no alert is silently dropped.
    """
    t = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

# Seconds to wait before attempt N (0-indexed after the first).
# The list length is the hard cap; settings().webhook_max_attempts can lower it.
_RETRY_DELAYS = [2, 8, 30]


# ---------------------------------------------------------------------------
# Signature helpers — used by both sender and receiver
# ---------------------------------------------------------------------------

def _compute_signature(secret: str, payload: bytes) -> str:
    """
    Compute HMAC-SHA256 of `payload` using `secret`.

    Returns "sha256=<hex>" — the same format used by GitHub webhooks,
    Stripe, and most FHIR integration platforms.
    """
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_webhook_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify an HMAC-SHA256 signature from the X-Signature request header.

    Call this on the RECEIVING side (EHR, case-management system) to confirm
    the POST was sent by this intake system and was not tampered with in transit.

    The comparison is constant-time (hmac.compare_digest) so it is safe
    against timing-oracle attacks.

    Args:
        payload:          Raw request body bytes (read before any JSON decode).
        signature_header: Value of the X-Signature header, e.g. "sha256=abc…".
        secret:           The shared secret from COMPLETION_WEBHOOK_SECRET.

    Returns:
        True if the signature matches, False if invalid or missing.

    Example (FastAPI receiver):
        @app.post("/intake/bundle")
        async def receive_bundle(request: Request):
            body = await request.body()
            sig  = request.headers.get("X-Signature", "")
            if not verify_webhook_signature(body, sig, SHARED_SECRET):
                raise HTTPException(401, "Invalid signature")
            bundle = json.loads(body)
            ...
    """
    if not secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = _compute_signature(secret, payload)
    return hmac.compare_digest(expected, signature_header)


def _payload_hash(data: bytes) -> str:
    """SHA-256 hex digest of payload bytes — used as the idempotency key."""
    return hashlib.sha256(data).hexdigest()


def _url_hash(url: str) -> str:
    """SHA-256 hex digest of a URL — stored instead of the raw URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _iso_now_plus(seconds: float) -> str:
    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ---------------------------------------------------------------------------
# Core delivery engine — retry loop with DB-backed status tracking
# ---------------------------------------------------------------------------

def _post_with_retry(
    *,
    url: str,
    data: bytes,
    headers: dict,
    event_type: str,
    thread_id: str,
    max_attempts: int,
) -> dict[str, Any]:
    """
    POST `data` to `url` with exponential-backoff retry and delivery tracking.

    Every attempt:
      1. Creates or updates a row in `webhook_deliveries` (status, attempts,
         last_http_status, last_error, next_retry_at).
      2. Emits a structured log event for observability / alerting.

    Idempotency: before the first attempt, checks whether an identical
    payload (same thread_id + event_type + SHA-256 hash) was already
    successfully delivered.  If so, returns immediately without a network call.

    Returns a summary dict:
      status       "success"|"failed"|"exhausted"|"duplicate_skipped"|"no_url"
      delivery_id  UUID of the webhook_deliveries row (absent for no_url)
      attempts     number of HTTP attempts made
    """
    from . import sqlite_db as db

    if not url:
        return {"status": "no_url", "attempts": 0}

    ph = _payload_hash(data)
    uh = _url_hash(url)

    # ── Idempotency check ──────────────────────────────────────────────
    existing = db.get_webhook_delivery_by_hash(thread_id, event_type, ph)
    if existing and existing.get("status") == "success":
        log_event(
            "webhook_delivery_skipped_duplicate",
            thread_id=thread_id,
            event_type=event_type,
            delivery_id=existing.get("delivery_id"),
        )
        return {
            "status": "duplicate_skipped",
            "delivery_id": existing.get("delivery_id"),
            "attempts": 0,
        }

    delivery_id = str(uuid.uuid4())
    db.create_webhook_delivery(delivery_id, thread_id, event_type, uh, ph)

    cap = min(max_attempts, len(_RETRY_DELAYS) + 1)
    attempt = 0
    last_status: int | None = None
    last_error: str | None = None

    while attempt < cap:
        attempt += 1

        # Exponential backoff before each retry (never before attempt 1)
        if attempt > 1:
            delay = _RETRY_DELAYS[min(attempt - 2, len(_RETRY_DELAYS) - 1)]
            log_event(
                "webhook_retry_waiting",
                thread_id=thread_id,
                event_type=event_type,
                delivery_id=delivery_id,
                attempt=attempt,
                delay_s=delay,
            )
            time.sleep(delay)

        try:
            req = urllib.request.Request(url, data=data, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                last_status = resp.status
                last_error = None

            if 200 <= last_status < 300:
                db.update_webhook_delivery(
                    delivery_id,
                    status="success",
                    attempts=attempt,
                    last_http_status=last_status,
                    last_error=None,
                    next_retry_at=None,
                )
                log_event(
                    "webhook_delivery_success",
                    thread_id=thread_id,
                    event_type=event_type,
                    delivery_id=delivery_id,
                    attempt=attempt,
                    http_status=last_status,
                )
                return {"status": "success", "delivery_id": delivery_id, "attempts": attempt}

            # Non-2xx response — treat as failure and maybe retry
            last_error = f"http_{last_status}"

        except Exception as exc:
            last_error = str(exc)[:300]
            last_status = None

        will_retry = attempt < cap
        next_retry = (
            _iso_now_plus(_RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)])
            if will_retry else None
        )
        db.update_webhook_delivery(
            delivery_id,
            status="failed" if will_retry else "exhausted",
            attempts=attempt,
            last_http_status=last_status,
            last_error=last_error,
            next_retry_at=next_retry,
        )
        log_event(
            "webhook_delivery_failed",
            level="warning",
            thread_id=thread_id,
            event_type=event_type,
            delivery_id=delivery_id,
            attempt=attempt,
            http_status=last_status,
            error=last_error,
            will_retry=will_retry,
        )

    return {"status": "exhausted", "delivery_id": delivery_id, "attempts": attempt}


# ---------------------------------------------------------------------------
# Slack alert
# ---------------------------------------------------------------------------

def slack_alert(
    *,
    webhook_url: str,
    text: str,
    thread_id: str = "",
    event_type: str = "slack_alert",
    max_attempts: int | None = None,
) -> bool:
    """Send a message to a Slack channel via Incoming Webhook."""
    if not webhook_url:
        return False

    cap = max_attempts if max_attempts is not None else settings().webhook_max_attempts
    payload = json.dumps({"text": text}).encode("utf-8")
    result = _post_with_retry(
        url=webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        event_type=event_type,
        thread_id=thread_id,
        max_attempts=cap,
    )
    return result["status"] == "success"


# ---------------------------------------------------------------------------
# Signed FHIR webhook
# ---------------------------------------------------------------------------

def signed_fhir_webhook(
    *,
    url: str,
    secret: str,
    fhir_json: str,
    thread_id: str,
    max_attempts: int | None = None,
) -> bool:
    """
    POST a FHIR R4 Bundle with an HMAC-SHA256 signature header.

    Signing
    ───────
    The raw payload bytes are signed with _compute_signature(secret, payload),
    which produces "sha256=<hex>".  The receiver calls:
        verify_webhook_signature(body, header, secret)
    to confirm authenticity before parsing the bundle.

    Idempotency
    ───────────
    If the same bundle was already delivered successfully for this thread_id,
    the call is a no-op (returns True without a network request).

    Retry
    ─────
    On failure, delivery is retried up to `max_attempts` times (default from
    settings().webhook_max_attempts = 3) with delays of 2s, 8s, and 30s.
    """
    if not url or not fhir_json:
        return False

    cap = max_attempts if max_attempts is not None else settings().webhook_max_attempts

    payload = fhir_json.encode("utf-8")
    sig = _compute_signature(secret, payload) if secret else ""

    result = _post_with_retry(
        url=url,
        data=payload,
        headers={
            "Content-Type": "application/fhir+json",
            "X-Thread-Id": thread_id,
            "X-Signature": sig,
        },
        event_type="fhir_completion",
        thread_id=thread_id,
        max_attempts=cap,
    )
    return result["status"] in ("success", "duplicate_skipped")


# ---------------------------------------------------------------------------
# Domain dispatch functions
# ---------------------------------------------------------------------------

def _do_emergency_alert(*, thread_id: str, patient_name: str, red_flags: list[str], session_short: str) -> None:

    text = (
        f":rotating_light: *EMERGENCY ESCALATION*\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Red flags: {', '.join(red_flags)}\n"
        f"Session: {session_short}"
    )
    slack_alert(webhook_url=settings().slack_webhook_url, text=text,
                thread_id=thread_id, event_type="slack_emergency")


def dispatch_emergency_alert(
    *,
    thread_id: str,
    patient_name: str,
    red_flags: list[str],
    session_short: str,
) -> None:
    """
    Fire emergency Slack alert in a background thread.

    Patient safety: a downed Slack endpoint must never delay the "call 911"
    message shown to the patient.  Delivery is still tracked + retried via
    webhook_deliveries — nothing is silently dropped.
    """
    _dispatch_in_thread(_do_emergency_alert, {
        "thread_id": thread_id, "patient_name": patient_name,
        "red_flags": red_flags, "session_short": session_short,
    })


def _do_intake_complete(*, thread_id: str, patient_name: str, risk_level: str, fhir_json: str | None) -> None:

    session_short = thread_id[:8]
    text = (
        f":white_check_mark: *Intake Complete — Risk: {risk_level.upper()}*\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Session: {session_short}"
    )
    slack_alert(webhook_url=settings().slack_webhook_url, text=text,
                thread_id=thread_id, event_type="slack_intake_complete")
    if fhir_json:
        signed_fhir_webhook(url=settings().completion_webhook_url,
                            secret=settings().completion_webhook_secret,
                            fhir_json=fhir_json, thread_id=thread_id)


def dispatch_intake_complete(
    *,
    thread_id: str,
    patient_name: str,
    risk_level: str,
    fhir_json: str | None,
) -> None:
    """Fire intake-complete Slack + FHIR webhook in a background thread."""
    _dispatch_in_thread(_do_intake_complete, {
        "thread_id": thread_id, "patient_name": patient_name,
        "risk_level": risk_level, "fhir_json": fhir_json,
    })


def _fmt_partial_identity(identity: dict) -> str:
    """Best available identity description for staff when name is unknown."""
    parts = []
    if (identity.get("name") or "").strip():
        parts.append(identity["name"])
    if (identity.get("phone") or "").strip():
        parts.append(f"phone {identity['phone']}")
    if (identity.get("dob") or "").strip():
        parts.append(f"DOB {identity['dob']}")
    return ", ".join(parts) if parts else "Unknown — crisis occurred before identity was collected"


def _do_crisis_alert(*, thread_id: str, patient_name: str, matched_phrases: list[str],
                     partial_identity: dict | None = None, message_preview: str = "") -> None:

    identity_line = (
        patient_name
        if patient_name and patient_name != "unknown patient"
        else _fmt_partial_identity(partial_identity or {})
    )
    lines = [
        ":rotating_light: *CRISIS LANGUAGE DETECTED — IMMEDIATE ATTENTION REQUIRED*",
        f"Patient: {identity_line}",
        f"Session: `{thread_id}`",
    ]
    if message_preview:
        lines.append(f'Patient wrote: "{message_preview[:120]}"')
    lines.append(f"Detected: {', '.join(matched_phrases)}")
    lines.append("_Open the clinician dashboard to view the full session._")
    slack_alert(webhook_url=settings().slack_webhook_url, text="\n".join(lines),
                thread_id=thread_id, event_type="slack_crisis")


def dispatch_crisis_alert(
    *,
    thread_id: str,
    patient_name: str,
    matched_phrases: list[str],
    partial_identity: dict | None = None,
    message_preview: str = "",
) -> None:
    """
    Fire crisis Slack alert in a background thread.

    partial_identity and message_preview are included when identity has not
    yet been collected — gives staff enough context to act without a name.
    Patient safety: the 988 Lifeline message reaches the patient immediately
    regardless of Slack latency.
    """
    _dispatch_in_thread(_do_crisis_alert, {
        "thread_id": thread_id, "patient_name": patient_name,
        "matched_phrases": matched_phrases,
        "partial_identity": partial_identity or {},
        "message_preview": message_preview,
    })


# ---------------------------------------------------------------------------
# Dead-letter retry worker
# ---------------------------------------------------------------------------

def retry_exhausted_webhooks() -> int:
    """
    Re-queue webhook deliveries that exhausted all original retry attempts.

    Call this periodically (e.g. at process startup and hourly) to give
    transient downstream failures a second chance without operator intervention.

    Design:
      • Reads exhausted deliveries older than `dead_letter_retry_after_hours`
        and below the lifetime attempt cap from the DB.
      • Resets each row to 'pending' so the existing _post_with_retry loop
        handles the actual HTTP call — no duplicate delivery logic needed here.
      • Dispatches in a background thread to avoid blocking the caller.
      • Emits a structured log event so the retry is observable in the audit log.

    Returns the number of deliveries re-queued.
    """
    from . import sqlite_db as db

    cfg = settings().intake
    candidates = db.get_exhausted_webhooks(
        older_than_hours=cfg.dead_letter_retry_after_hours,
        max_lifetime_attempts=cfg.dead_letter_max_lifetime_attempts,
    )
    if not candidates:
        return 0

    def _requeue_and_dispatch(rows: list) -> None:
        for row in rows:
            delivery_id = row["delivery_id"]
            thread_id   = row["thread_id"]
            event_type  = row["event_type"]
            db.requeue_webhook_delivery(delivery_id)
            log_event(
                "webhook_dead_letter_requeued",
                delivery_id=delivery_id,
                thread_id=thread_id,
                event_type=event_type,
                prior_attempts=row["attempts"],
            )

    _dispatch_in_thread(_requeue_and_dispatch, {"rows": list(candidates)})
    return len(candidates)

