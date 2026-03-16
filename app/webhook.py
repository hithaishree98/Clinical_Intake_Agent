"""
webhook.py — Outbound notifications (Slack + HMAC-signed FHIR webhook).
All sends are fire-and-forget: failures are logged, never raised.
"""
from __future__ import annotations

import json
import hashlib
import hmac
import urllib.request

from .logging_utils import log_event


def _post(url: str, data: bytes, headers: dict, tag: str, thread_id: str = "") -> bool:
    """Fire-and-forget POST. Returns True on success, False on failure."""
    try:
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            log_event(tag, thread_id=thread_id, http_status=resp.status)
            return True
    except Exception as e:
        log_event(
            f"{tag}_error",
            level="warning",
            thread_id=thread_id,
            error=str(e)[:200],
        )
        return False


def slack_alert(*, webhook_url: str, text: str, thread_id: str = "") -> bool:
    """Send a message to a Slack channel via Incoming Webhook."""
    if not webhook_url:
        return False

    payload = json.dumps({"text": text}).encode("utf-8")
    return _post(
        url=webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        tag="slack_sent",
        thread_id=thread_id,
    )


def signed_fhir_webhook(*, url: str, secret: str, fhir_json: str, thread_id: str) -> bool:
    """POST a FHIR R4 Bundle with an HMAC-SHA256 signature header."""
    if not url or not fhir_json:
        return False

    payload = fhir_json.encode("utf-8")
    sig = ""
    if secret:
        sig = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    return _post(
        url=url,
        data=payload,
        headers={
            "Content-Type": "application/fhir+json",
            "X-Thread-Id": thread_id,
            "X-Signature": sig,
        },
        tag="fhir_webhook_fired",
        thread_id=thread_id,
    )


def dispatch_emergency_alert(
    *,
    thread_id: str,
    patient_name: str,
    red_flags: list[str],
    session_short: str,
) -> None:
    """Called when an emergency escalation is triggered."""
    from .settings import settings

    text = (
        f":rotating_light: *EMERGENCY ESCALATION*\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Red flags: {', '.join(red_flags)}\n"
        f"Session: {session_short}"
    )

    slack_alert(
        webhook_url=settings.slack_webhook_url,
        text=text,
        thread_id=thread_id,
    )


def dispatch_intake_complete(
    *,
    thread_id: str,
    patient_name: str,
    risk_level: str,
    fhir_json: str | None,
) -> None:
    """Called when a patient finishes the full intake flow."""
    from .settings import settings

    session_short = thread_id[:8]

    text = (
        f":white_check_mark: *Intake Complete — Risk: {risk_level.upper()}*\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Session: {session_short}"
    )

    slack_alert(
        webhook_url=settings.slack_webhook_url,
        text=text,
        thread_id=thread_id,
    )
    if fhir_json:
        signed_fhir_webhook(
            url=settings.completion_webhook_url,
            secret=settings.completion_webhook_secret,
            fhir_json=fhir_json,
            thread_id=thread_id,
        )


def dispatch_crisis_alert(
    *,
    thread_id: str,
    patient_name: str,
    matched_phrases: list[str],
) -> None:
    """Called when self-harm / crisis language is detected."""
    from .settings import settings

    text = (
        f":warning: *CRISIS LANGUAGE DETECTED*\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Matched: {', '.join(matched_phrases)}\n"
        f"Session: {thread_id[:8]}"
    )

    slack_alert(
        webhook_url=settings.slack_webhook_url,
        text=text,
        thread_id=thread_id,
    )
