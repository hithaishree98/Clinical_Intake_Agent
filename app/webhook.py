"""
webhook.py — Outbound notification integrations.

All integrations are free-tier compatible and require no paid accounts.
All sends are fire-and-forget: failures are logged, never raised.
Configure via .env (see .env.example for all options).

─────────────────────────────────────────────────────────
ntfy.sh  (RECOMMENDED for local dev + prod)
  Free, open-source push notification service.
  Setup (30 seconds):
    1. Go to https://ntfy.sh
    2. Pick any unique topic name (e.g. "hospital-intake-abc123")
    3. Set NTFY_TOPIC=hospital-intake-abc123 in .env
    4. Subscribe via browser: https://ntfy.sh/hospital-intake-abc123
       OR install the ntfy mobile app and subscribe to your topic
    Emergency alerts arrive as high-priority push notifications on your phone.
  Self-host option: docker run -p 80:80 binwiederhier/ntfy
─────────────────────────────────────────────────────────
Discord  (RECOMMENDED for team alerts)
  Free, zero infrastructure needed.
  Setup (2 minutes):
    1. In your Discord server: Edit Channel → Integrations → Webhooks → New Webhook
    2. Copy the webhook URL
    3. Set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/... in .env
  Emergency alerts appear as red embeds; intake completions as blue.
─────────────────────────────────────────────────────────
Generic signed webhook  (EHR integration)
  HMAC-SHA256 signed POST of the FHIR R4 Bundle.
  Use https://webhook.site for testing (free, no signup, shows request body).
  Set COMPLETION_WEBHOOK_URL=https://webhook.site/your-uuid in .env
─────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import hashlib
import hmac
import urllib.request

from .logging_utils import log_event


# ---------------------------------------------------------------------------
# Shared low-level POST helper
# ---------------------------------------------------------------------------

def _post(url: str, data: bytes, headers: dict, tag: str, thread_id: str = "") -> bool:
    """
    Returns True on success, False on any failure.
    Never raises — webhook failures must never affect patient flow.
    """
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


# ---------------------------------------------------------------------------
# ntfy.sh — free push notifications
# Docs: https://docs.ntfy.sh
# ---------------------------------------------------------------------------

NTFY_PRIORITIES = {"min", "low", "default", "high", "urgent"}
# "urgent" bypasses Do Not Disturb on the ntfy mobile app — use for emergencies only


def ntfy_alert(
    *,
    topic: str,
    title: str,
    message: str,
    priority: str = "default",
    tags: str = "hospital",
    thread_id: str = "",
    click_url: str = "",
) -> bool:
    """
    Send a push notification via ntfy.sh.

    priority values:
      default  → standard push notification
      high     → vibrates device
      urgent   → bypasses DND; use only for real emergencies

    tags: comma-separated emoji shortcodes shown in notification
      Examples: "warning,hospital" → ⚠️🏥
                "rotating_light"   → 🚨
                "white_check_mark" → ✅

    click_url: if set, tapping the notification opens this URL (e.g. clinician dashboard)
    """
    if not topic:
        return False

    if priority not in NTFY_PRIORITIES:
        priority = "default"

    url = f"https://ntfy.sh/{topic}"
    headers = {
        "Title": title[:250],          # ntfy title limit
        "Priority": priority,
        "Tags": tags,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if click_url:
        headers["Click"] = click_url

    return _post(
        url=url,
        data=message.encode("utf-8"),
        headers=headers,
        tag="ntfy_sent",
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# Discord — free webhook alerts
# Docs: https://discord.com/developers/docs/resources/webhook
# ---------------------------------------------------------------------------

# Colour palette for Discord embeds
DISCORD_COLOURS = {
    "emergency": 0xFF0000,   # red
    "warning":   0xFF9900,   # orange
    "info":      0x00B0F4,   # blue
    "success":   0x00C851,   # green
}


def discord_alert(
    *,
    webhook_url: str,
    title: str,
    message: str,
    colour: str | int = "info",
    fields: list[dict] | None = None,
    thread_id: str = "",
) -> bool:
    """
    Send an embed message to a Discord channel via webhook.

    colour: one of "emergency", "warning", "info", "success" or an int hex colour.
    fields: optional list of {"name": str, "value": str, "inline": bool}
    """
    if not webhook_url:
        return False

    color_int = DISCORD_COLOURS.get(colour, colour) if isinstance(colour, str) else colour

    embed: dict = {
        "title": title[:256],
        "description": message[:4096],
        "color": color_int,
    }
    if fields:
        embed["fields"] = [
            {
                "name": f.get("name", "")[:256],
                "value": f.get("value", "")[:1024],
                "inline": bool(f.get("inline", False)),
            }
            for f in fields[:25]  # Discord limit
        ]

    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    return _post(
        url=webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        tag="discord_sent",
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# Generic HMAC-signed webhook (for EHR / FHIR integration)
# Test with https://webhook.site — get a free unique URL, paste into .env
# ---------------------------------------------------------------------------

def signed_fhir_webhook(
    *,
    url: str,
    secret: str,
    fhir_json: str,
    thread_id: str,
) -> bool:
    """
    POST a FHIR R4 Bundle with an HMAC-SHA256 signature header.
    The receiving system can verify: hmac.new(secret, body, sha256) == X-Signature.

    For local testing: set COMPLETION_WEBHOOK_URL=https://webhook.site/your-uuid
    Requests appear in real time at https://webhook.site/#/your-uuid
    """
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


# ---------------------------------------------------------------------------
# High-level dispatcher — single call for all configured targets
# ---------------------------------------------------------------------------

def dispatch_emergency_alert(
    *,
    thread_id: str,
    patient_name: str,
    red_flags: list[str],
    session_short: str,
) -> None:
    """Called when an emergency escalation is triggered."""
    from .settings import settings

    title = "🚨 EMERGENCY ESCALATION"
    message = (
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Red flags: {', '.join(red_flags)}\n"
        f"Session: {session_short}"
    )

    ntfy_alert(
        topic=getattr(settings, "ntfy_topic", ""),
        title=title,
        message=message,
        priority="urgent",
        tags="rotating_light,hospital",
        thread_id=thread_id,
    )
    discord_alert(
        webhook_url=getattr(settings, "discord_webhook_url", ""),
        title=title,
        message=message,
        colour="emergency",
        fields=[
            {"name": "Patient", "value": patient_name or "Unknown", "inline": True},
            {"name": "Session", "value": session_short, "inline": True},
            {"name": "Red Flags", "value": ", ".join(red_flags) or "—", "inline": False},
        ],
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
    colour = "emergency" if risk_level == "high" else ("warning" if risk_level == "medium" else "success")
    ntfy_priority = "high" if risk_level in ("high", "medium") else "default"

    title = f"✅ Intake Complete — Risk: {risk_level.upper()}"
    message = (
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Risk: {risk_level}\n"
        f"Session: {session_short}"
    )

    ntfy_alert(
        topic=getattr(settings, "ntfy_topic", ""),
        title=title,
        message=message,
        priority=ntfy_priority,
        tags="white_check_mark,hospital",
        thread_id=thread_id,
    )
    discord_alert(
        webhook_url=getattr(settings, "discord_webhook_url", ""),
        title=title,
        message=message,
        colour=colour,
        fields=[
            {"name": "Patient", "value": patient_name or "Unknown", "inline": True},
            {"name": "Risk Level", "value": risk_level.upper(), "inline": True},
        ],
        thread_id=thread_id,
    )
    if fhir_json:
        signed_fhir_webhook(
            url=getattr(settings, "completion_webhook_url", ""),
            secret=getattr(settings, "completion_webhook_secret", ""),
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

    title = "⚠️ CRISIS LANGUAGE DETECTED"
    message = (
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Matched: {', '.join(matched_phrases)}\n"
        f"Session: {thread_id[:8]}"
    )

    ntfy_alert(
        topic=getattr(settings, "ntfy_topic", ""),
        title=title,
        message=message,
        priority="urgent",
        tags="warning,hospital",
        thread_id=thread_id,
    )
    discord_alert(
        webhook_url=getattr(settings, "discord_webhook_url", ""),
        title=title,
        message=message,
        colour="emergency",
        thread_id=thread_id,
    )