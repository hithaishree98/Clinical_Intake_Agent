"""
transcription.py — Audio → text via Groq Whisper.

Voice is a thin shell around /chat.  The audio leg lives here:
  POST /transcribe (audio) → Groq → text → client → POST /chat (text)

Why a separate module:
  - Isolates the only place that talks to Groq.  Swapping providers (to
    self-hosted faster-whisper for HIPAA, or AWS Transcribe Medical) is a
    one-file change.
  - Keeps the audio buffer in a tight scope.  No globals hold raw audio,
    nothing is logged, nothing is persisted.  The bytes leave Python's
    heap as soon as transcribe_audio() returns.
  - Houses the hallucination filter — Whisper occasionally emits canned
    YouTube-caption phrases on silence ("Thanks for watching!"), which
    would otherwise be sent verbatim into /chat.

Failure mode: any exception bubbles up to the caller, which converts it
to a 503 so the frontend falls back to typing.  The chat path is never
affected by Groq outages.
"""
from __future__ import annotations

import re
from typing import Optional

from .settings import get_settings


# ---------------------------------------------------------------------------
# Lazy Groq client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """
    Return the singleton Groq client, importing the SDK lazily.

    Lazy import means the project still installs and runs without `groq`
    in the environment, as long as voice is left disabled.  Tests that
    don't exercise /transcribe never trigger the import.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    from groq import Groq  # imported lazily — see docstring
    _client = Groq(api_key=settings.groq_api_key)
    return _client


def reset_client_for_tests() -> None:
    """Drop the cached client so a test can patch the SDK and re-init."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

# Whisper's most common silence/noise hallucinations.  These phrases appear
# in YouTube captions in the training data, so the model emits them when
# fed a near-silent clip.  Matching as substrings (case-insensitive) is
# coarse but correct: a real patient saying "thanks for watching" in their
# medical history is virtually impossible.
_HALLUCINATION_PATTERNS = [
    r"thanks?\s+for\s+watching",
    r"please\s+(like\s+(and\s+)?)?subscribe",
    r"subscribe\s+to\s+my\s+channel",
    r"don'?t\s+forget\s+to\s+subscribe",
    r"see\s+you\s+(in\s+the\s+)?next\s+video",
    r"\[music\]",
    r"\[applause\]",
    r"♪",
]
_HALLUCINATION_RE = re.compile("|".join(_HALLUCINATION_PATTERNS), re.IGNORECASE)


def is_likely_hallucination(text: str) -> bool:
    """
    True if the transcript looks like a Whisper canned phrase rather than
    real speech.  Conservative — false positives here are tolerable
    (patient just retries) but a false negative means a nonsense phrase
    flows into the LangGraph extraction step.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_HALLUCINATION_RE.search(stripped))


# ---------------------------------------------------------------------------
# Transcription entry point
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    content_type: str,
    *,
    filename: str = "audio.webm",
) -> str:
    """
    Send audio to Groq Whisper and return the transcript text.

    Returns "" when the model produces no usable speech (silent clip,
    pure noise, or a known hallucination).  An empty string lets the
    caller respond with a soft "didn't catch that, try again" rather
    than pushing nonsense into /chat.

    Raises any underlying Groq SDK error so the endpoint can fail closed
    with a 503; we don't retry here because the per-clip cost of a retry
    is non-trivial and the caller can simply ask the patient to repeat.
    """
    client = _get_client()
    settings = get_settings()

    # The SDK accepts a (filename, bytes) tuple for the file argument.
    # The filename's extension hints at format detection on Groq's side.
    response = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=settings.intake.groq_stt_model,
        response_format="text",
        language="en",
        # Bias the model toward common medical context so drug / anatomy
        # terms transcribe more reliably without inflating prompt cost.
        prompt="Patient describing symptoms, medications, allergies, or medical history.",
    )

    # Groq returns a plain string when response_format="text".  Defensive
    # str() coercion in case the SDK ever wraps it in a model object.
    text = str(response).strip() if response is not None else ""

    if not text:
        return ""
    if is_likely_hallucination(text):
        return ""
    return text


__all__ = [
    "transcribe_audio",
    "is_likely_hallucination",
    "reset_client_for_tests",
]
