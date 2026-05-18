"""
webhook.py — Retryable outbound notifications with delivery tracking.

Slack alerts for emergency/crisis events; HMAC-SHA256-signed FHIR R4 Bundle POST for EHR integration.
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
    """Fire-and-forget in a daemon thread; alerts must never block graph execution."""
    threading.Thread(target=target, kwargs=kwargs, daemon=True).start()


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
    Verify HMAC-SHA256 from X-Signature header. Call on the receiving side (EHR).
    Uses constant-time compare to guard against timing attacks.
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
    POST with exponential-backoff retry, idempotency check, and DB delivery tracking.
    Returns dict with status, delivery_id, and attempts.
    """
    from . import sqlite_db as db

    if not url:
        return {"status": "no_url", "attempts": 0}
    if not url.startswith(("https://", "http://")):
        return {"status": "invalid_url_scheme", "attempts": 0}

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
    # Persist the raw body so the dead-letter worker can replay it later.
    # URLs/headers are re-derived from event_type at replay time, so secrets
    # and rotated webhook URLs aren't pinned to the row.
    db.create_webhook_delivery(delivery_id, thread_id, event_type, uh, ph, data)

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
    """POST a FHIR R4 Bundle with HMAC-SHA256 signature. Refuses to send unsigned bundles."""
    if not url or not fhir_json:
        return False

    # Unsigned bundles are a security hole — some EHRs accept empty headers silently.
    if not secret:
        log_event(
            "fhir_webhook_blocked_no_secret",
            level="warning",
            thread_id=thread_id,
            reason="completion_webhook_secret unset; refusing to send unsigned FHIR bundle",
        )
        return False

    cap = max_attempts if max_attempts is not None else settings().webhook_max_attempts

    payload = fhir_json.encode("utf-8")
    sig = _compute_signature(secret, payload)

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
    """Fire emergency Slack alert in a background thread."""
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
                     partial_identity: dict | None = None) -> None:

    identity_line = (
        patient_name
        if patient_name and patient_name != "unknown patient"
        else _fmt_partial_identity(partial_identity or {})
    )
    lines = [
        ":rotating_light: *CRISIS LANGUAGE DETECTED — IMMEDIATE ATTENTION REQUIRED*",
        f"Patient: {identity_line}",
        f"Session: `{thread_id}`",
        f"Detected: {', '.join(matched_phrases)}",
        "_Open the clinician dashboard to view the full session._",
    ]
    slack_alert(webhook_url=settings().slack_webhook_url, text="\n".join(lines),
                thread_id=thread_id, event_type="slack_crisis")


def dispatch_crisis_alert(
    *,
    thread_id: str,
    patient_name: str,
    matched_phrases: list[str],
    partial_identity: dict | None = None,
) -> None:
    """
    Fire crisis Slack alert in a background thread.
    Verbatim patient message is omitted — clinician dashboard has it.
    """
    _dispatch_in_thread(_do_crisis_alert, {
        "thread_id": thread_id, "patient_name": patient_name,
        "matched_phrases": matched_phrases,
        "partial_identity": partial_identity or {},
    })


# ---------------------------------------------------------------------------
# Dead-letter retry worker
# ---------------------------------------------------------------------------

def _resolve_dispatch_url(event_type: str) -> str:
    """Re-derive the target URL for an event_type from current settings."""
    cfg = settings()
    if event_type.startswith("slack_"):
        return cfg.slack_webhook_url or ""
    if event_type == "fhir_completion":
        return cfg.completion_webhook_url or ""
    return ""


def _build_replay_headers(event_type: str, thread_id: str, payload: bytes) -> dict:
    """Re-derive the request headers (and signature) for a replay."""
    if event_type.startswith("slack_"):
        return {"Content-Type": "application/json"}
    if event_type == "fhir_completion":
        secret = settings().completion_webhook_secret
        headers = {
            "Content-Type": "application/fhir+json",
            "X-Thread-Id": thread_id,
        }
        # Recompute the signature against the *current* secret so a rotation
        # doesn't lock dead-lettered bundles out of the EHR.
        if secret:
            headers["X-Signature"] = _compute_signature(secret, payload)
        return headers
    return {}


def _replay_delivery(row: dict) -> bool:
    """
    Single-shot HTTP replay of one exhausted delivery row.

    Updates the existing row in place: success → 'success', failure →
    back to 'exhausted' with attempts incremented.  The dead-letter
    cycle runs hourly, so the next sweep picks the row back up if it's
    still under the lifetime cap.
    """
    from . import sqlite_db as db

    delivery_id = row["delivery_id"]
    event_type  = row["event_type"]
    thread_id   = row.get("thread_id") or ""
    payload     = row.get("payload_body")
    prior       = int(row.get("attempts") or 0)

    if not payload:
        # Pre-migration row — body was never persisted, so replay is impossible.
        # Mark as permanently exhausted with a distinct error so the worker
        # stops re-selecting it and ops can grep for it.
        db.update_webhook_delivery(
            delivery_id, status="exhausted", attempts=prior,
            last_http_status=None, last_error="dead_letter_no_payload_body",
            next_retry_at=None,
        )
        log_event("webhook_dead_letter_skipped", level="warning",
                  delivery_id=delivery_id, thread_id=thread_id,
                  event_type=event_type, reason="no_payload_body")
        return False

    url = _resolve_dispatch_url(event_type)
    if not url:
        db.update_webhook_delivery(
            delivery_id, status="exhausted", attempts=prior,
            last_http_status=None,
            last_error=f"dead_letter_url_unset_for_{event_type}",
            next_retry_at=None,
        )
        log_event("webhook_dead_letter_skipped", level="warning",
                  delivery_id=delivery_id, thread_id=thread_id,
                  event_type=event_type, reason="url_unset")
        return False
    if not url.startswith(("https://", "http://")):
        db.update_webhook_delivery(
            delivery_id, status="exhausted", attempts=prior,
            last_http_status=None,
            last_error="dead_letter_invalid_url_scheme",
            next_retry_at=None,
        )
        log_event("webhook_dead_letter_skipped", level="warning",
                  delivery_id=delivery_id, thread_id=thread_id,
                  event_type=event_type, reason="invalid_url_scheme")
        return False

    headers = _build_replay_headers(event_type, thread_id, payload)
    attempt = prior + 1

    try:
        req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            ok = 200 <= resp.status < 300
            db.update_webhook_delivery(
                delivery_id,
                status="success" if ok else "exhausted",
                attempts=attempt,
                last_http_status=resp.status,
                last_error=None if ok else f"http_{resp.status}",
                next_retry_at=None,
            )
            log_event(
                "webhook_dead_letter_replay",
                level="info" if ok else "warning",
                delivery_id=delivery_id, thread_id=thread_id,
                event_type=event_type, attempt=attempt,
                http_status=resp.status, ok=ok,
            )
            return ok
    except Exception as exc:
        db.update_webhook_delivery(
            delivery_id, status="exhausted", attempts=attempt,
            last_http_status=None, last_error=str(exc)[:300],
            next_retry_at=None,
        )
        log_event(
            "webhook_dead_letter_replay",
            level="warning",
            delivery_id=delivery_id, thread_id=thread_id,
            event_type=event_type, attempt=attempt,
            error=str(exc)[:200], ok=False,
        )
        return False


def retry_exhausted_webhooks() -> int:
    """Replay exhausted dead-letter deliveries in a daemon thread. Returns candidate count."""
    from . import sqlite_db as db

    cfg = settings().intake
    candidates = db.get_exhausted_webhooks(
        older_than_hours=cfg.dead_letter_retry_after_hours,
        max_lifetime_attempts=cfg.dead_letter_max_lifetime_attempts,
    )
    if not candidates:
        return 0

    def _replay_all(rows: list) -> None:
        for row in rows:
            _replay_delivery(row)

    _dispatch_in_thread(_replay_all, {"rows": list(candidates)})
    return len(candidates)

