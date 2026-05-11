"""
Tests for the voice transcription leg.

Voice is layered on top of /chat — the audio→text endpoint is
/transcribe.  These tests cover the audio leg only; the chat-side
guardrails (idempotency, prompt injection, cost cap) are exercised by
test_integration.py and apply automatically because voice routes its
output back through /chat.

We never make real Groq calls in CI: the transcribe_audio function is
monkeypatched to return a fixed string, and the SDK is never imported.
"""
from __future__ import annotations

import io
import os
import uuid

import pytest


# ---------------------------------------------------------------------------
# Fixtures — small, self-contained TestClient + a fresh session helper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def voice_client(tmp_path_factory):
    """TestClient with voice enabled (fake key) and rate limiting off."""
    tmp = tmp_path_factory.mktemp("voice_db")
    os.environ["APP_DB_PATH"]        = str(tmp / "app.db")
    os.environ["CHECKPOINT_DB_PATH"] = str(tmp / "checkpoints.db")
    os.environ["GROQ_API_KEY"]       = "test-groq-key"
    os.environ["VOICE_ENABLED"]      = "true"

    from app.settings import get_settings
    from app import sqlite_db as _db

    s = get_settings()
    s.app_db_path        = os.environ["APP_DB_PATH"]
    s.checkpoint_db_path = os.environ["CHECKPOINT_DB_PATH"]
    s.groq_api_key       = "test-groq-key"
    s.voice_enabled      = True
    _db.close_all_connections()

    from starlette.testclient import TestClient
    from app.main import app
    from app.api.deps import limiter
    limiter.enabled = False

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    limiter.enabled = True


def _start(client) -> dict:
    r = client.post("/start", data={"mode": "clinic"})
    assert r.status_code == 200, r.text
    return r.json()


# A small chunk of bytes is fine — we mock transcribe_audio so the
# Groq SDK never sees them.  The bytes only need to satisfy the size
# and MIME-type guards at the endpoint.
def _audio_payload(size: int = 20_000) -> bytes:
    return b"\x00" * size


# ---------------------------------------------------------------------------
# /voice/config
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    def test_returns_enabled_when_key_set(self, voice_client):
        r = voice_client.get("/voice/config")
        assert r.status_code == 200
        body = r.json()
        assert body["enabled"] is True
        assert body["max_seconds"] >= 1
        assert body["max_bytes"]  >= 1

    def test_returns_disabled_when_key_blank(self, voice_client, monkeypatch):
        from app.settings import get_settings
        s = get_settings()
        monkeypatch.setattr(s, "groq_api_key", "")
        r = voice_client.get("/voice/config")
        assert r.json()["enabled"] is False


# ---------------------------------------------------------------------------
# /transcribe — auth, validation, guardrails
# ---------------------------------------------------------------------------

class TestTranscribeGuardrails:
    def test_requires_session_token(self, voice_client):
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": "ghost"},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
        )
        # No Authorization header → 401 from require_session_token
        assert r.status_code == 401

    def test_rejects_unknown_thread(self, voice_client):
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": "no-such-thread"},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
            headers={"Authorization": "Bearer fake"},
        )
        # Bad token first → 401
        assert r.status_code == 401

    def test_rejects_oversized_audio(self, voice_client):
        sess = _start(voice_client)
        from app.settings import get_settings
        big = _audio_payload(get_settings().intake.transcribe_max_bytes + 100)
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", big, "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 413

    def test_rejects_bad_mime(self, voice_client):
        sess = _start(voice_client)
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.txt", b"hello", "text/plain")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 415

    def test_rejects_empty_audio(self, voice_client):
        sess = _start(voice_client)
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", b"", "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 400

    def test_returns_503_when_voice_disabled(self, voice_client, monkeypatch):
        sess = _start(voice_client)
        from app.settings import get_settings
        monkeypatch.setattr(get_settings(), "voice_enabled", False)
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# /transcribe — happy path with mocked Groq
# ---------------------------------------------------------------------------

class TestTranscribeHappyPath:
    def test_returns_text_from_groq(self, voice_client, monkeypatch):
        sess = _start(voice_client)
        from app import transcription
        monkeypatch.setattr(
            transcription, "transcribe_audio",
            lambda blob, ct, **kw: "I have a headache",
        )
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"text": "I have a headache"}

    def test_returns_empty_string_for_silent_clip(self, voice_client, monkeypatch):
        sess = _start(voice_client)
        from app import transcription
        monkeypatch.setattr(
            transcription, "transcribe_audio",
            lambda blob, ct, **kw: "",
        )
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 200
        assert r.json()["text"] == ""

    def test_groq_failure_returns_503(self, voice_client, monkeypatch):
        sess = _start(voice_client)
        from app import transcription

        def boom(*a, **kw):
            raise RuntimeError("groq exploded")
        monkeypatch.setattr(transcription, "transcribe_audio", boom)
        r = voice_client.post(
            "/transcribe",
            data={"thread_id": sess["thread_id"]},
            files={"audio": ("a.webm", _audio_payload(), "audio/webm")},
            headers={"Authorization": f"Bearer {sess['session_token']}"},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Hallucination filter — pure unit tests, no HTTP
# ---------------------------------------------------------------------------

class TestHallucinationFilter:
    def test_catches_thanks_for_watching(self):
        from app.transcription import is_likely_hallucination
        assert is_likely_hallucination("Thanks for watching!")
        assert is_likely_hallucination("thank you for watching, see you next video")
        assert is_likely_hallucination("Please subscribe to my channel.")

    def test_catches_caption_markers(self):
        from app.transcription import is_likely_hallucination
        assert is_likely_hallucination("[Music] some words")
        assert is_likely_hallucination("[applause]")
        assert is_likely_hallucination("♪ la la la")

    def test_passes_real_speech(self):
        from app.transcription import is_likely_hallucination
        assert not is_likely_hallucination("I have a headache")
        assert not is_likely_hallucination("My name is John Smith")
        assert not is_likely_hallucination("I take metoprolol every morning")

    def test_handles_empty_or_whitespace(self):
        from app.transcription import is_likely_hallucination
        assert not is_likely_hallucination("")
        assert not is_likely_hallucination("   ")
        assert not is_likely_hallucination(None or "")
