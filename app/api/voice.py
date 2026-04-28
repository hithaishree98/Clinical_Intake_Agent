"""
voice.py — Patient-facing voice transcription endpoints.

Voice is a thin shell on top of /chat:

  Browser (MediaRecorder)
    │  POST /transcribe (audio blob)
    ▼
  /transcribe  →  Groq Whisper  →  text
    │
    └── client posts the text to /chat (unchanged)
"""
import time

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from .. import sqlite_db as db
from ..logging_utils import log_event, set_trace_id
from ..settings import get_settings
from .deps import limiter, require_session_token

router = APIRouter()


# Browser MediaRecorder produces audio/webm by default.  We accept the
# common alternatives so Safari (mp4) and Firefox (ogg) work without
# special-casing the frontend.
_ALLOWED_AUDIO_MIMES = frozenset({
    "audio/webm",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
})


@router.get("/voice/config")
def voice_config():
    """
    Public configuration probe used by the frontend to decide whether to
    render the mic button.  Exposes only feature flags + non-secret
    limits; no auth required because the answer would otherwise force us
    to leak the same info via 4xx/5xx response patterns.
    """
    settings = get_settings()
    cfg = settings.intake
    return {
        "enabled":     bool(settings.voice_enabled and settings.groq_api_key),
        "max_seconds": cfg.transcribe_max_seconds,
        "max_bytes":   cfg.transcribe_max_bytes,
    }


@router.post("/transcribe")
@limiter.limit("30/minute")
async def transcribe(
    request: Request,
    thread_id: str = Form(...),
    audio: UploadFile = File(...),
    authorization: str = Header(default=""),
):
    """
    Transcribe a short patient audio clip via Groq Whisper.

    Returns ``{"text": "..."}`` on success.  Empty string when the clip
    is silent or matches a known Whisper hallucination — the frontend
    treats that as a soft "didn't catch that" prompt and never forwards
    nonsense into /chat.

    Hard guardrails (audio leg):
      • Same session-token auth as /chat.  A patient can only transcribe
        for their own thread.
      • Per-IP rate limit (30/min).
      • MIME whitelist — text/plain or images return 415, no Groq call.
      • Size cap from IntakeConfig.transcribe_max_bytes — defends
        against someone shipping a 100 MB blob through a stolen token.
      • Empty audio rejected with 400 (no point billing Groq for nothing).
      • Hallucination filter inside transcribe_audio() returns "" for
        canned YouTube-caption phrases.
      • Audio is never logged, never written to disk.  Only metadata
        (audio_bytes, duration_ms, empty?) goes to the log stream.
    """
    require_session_token(thread_id, authorization)

    settings = get_settings()
    if not settings.voice_enabled or not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Voice transcription is not configured on this server.",
        )

    # Mirror the /chat session-state checks so voice cannot resurrect
    # a closed session and burn Groq quota against it.
    sess = db.fetch_one(
        "SELECT thread_id, status FROM sessions WHERE thread_id=?", (thread_id,)
    )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess["status"] in ("done", "expired"):
        raise HTTPException(
            status_code=410,
            detail=f"Session already {sess['status']}. Please start a new intake.",
        )

    # MIME whitelist — reject before reading the body so a multi-MB
    # text/plain upload returns 415 immediately.
    content_type = (audio.content_type or "").lower().split(";", 1)[0].strip()
    if content_type not in _ALLOWED_AUDIO_MIMES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format. Use webm, wav, ogg, or mp4.",
        )

    max_bytes = settings.intake.transcribe_max_bytes

    # Read the upload with a hard ceiling.  We read max_bytes + 1 so we
    # can detect overflow without buffering an arbitrary-sized payload.
    blob = await audio.read(max_bytes + 1)
    if not blob:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(blob) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large (max {max_bytes // 1000} KB).",
        )

    set_trace_id(thread_id)
    audio_bytes = len(blob)
    t0 = time.time()
    try:
        # Imported lazily so the Groq SDK is not required at import time.
        # If the install is missing or partial, the 503 path fires and
        # the rest of the app continues unaffected.
        from ..transcription import transcribe_audio
        text = transcribe_audio(blob, content_type)
    except HTTPException:
        raise
    except Exception as e:
        # Don't leak provider error messages.  Log enough to diagnose,
        # surface a generic 503 so the UI falls back to typing.
        duration_ms = int((time.time() - t0) * 1000)
        db.record_voice_usage(
            thread_id=thread_id,
            audio_bytes=audio_bytes,
            duration_ms=duration_ms,
            empty=False,
            error=True,
        )
        log_event(
            "transcribe_failed",
            level="warning",
            thread_id=thread_id,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
        raise HTTPException(
            status_code=503,
            detail="Voice service temporarily unavailable. Please type instead.",
        )
    finally:
        # Drop the local reference promptly.  Python will GC the bytes;
        # this just makes intent explicit and keeps the audio out of any
        # exception traceback that might be logged upstream.
        del blob

    duration_ms = int((time.time() - t0) * 1000)
    is_empty = not bool(text)
    # Persist a usage row so the dashboard can show voice activity.  The
    # transcript text is NEVER stored — record_voice_usage only writes
    # metadata (audio_bytes, duration_ms, empty/error flags).
    db.record_voice_usage(
        thread_id=thread_id,
        audio_bytes=audio_bytes,
        duration_ms=duration_ms,
        empty=is_empty,
        error=False,
    )
    log_event(
        "transcribe_done",
        thread_id=thread_id,
        audio_bytes=audio_bytes,
        empty=is_empty,
        duration_ms=duration_ms,
    )

    return {"text": text}
