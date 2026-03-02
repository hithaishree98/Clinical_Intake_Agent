import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel


# Idempotency

class TestIdempotency:
    def test_same_message_returns_cached_response(self, tmp_db):
        from app import sqlite_db as db

        thread_id = "test-thread-1"
        db.create_session(thread_id)

        response = {"reply": "hello", "status": "active", "phase": "identity"}
        db.save_idempotent_response(
            thread_id,
            client_msg_id="msg-1",
            request_hash="abc123",
            response_obj=response,
        )

        cached = db.get_idempotent_response(thread_id, "msg-1")
        assert cached is not None
        assert cached["request_hash"] == "abc123"

    def test_different_key_returns_nothing(self, tmp_db):
        from app import sqlite_db as db

        thread_id = "test-thread-2"
        db.create_session(thread_id)

        cached = db.get_idempotent_response(thread_id, "msg-never-sent")
        assert cached is None

    def test_unknown_thread_returns_nothing(self, tmp_db):
        from app import sqlite_db as db

        cached = db.get_idempotent_response("ghost-thread", "msg-1")
        assert cached is None


# LLM fallback

class SimpleSchema(BaseModel):
    value: str = ""
    is_complete: bool = False


class TestLLMFallback:
    def test_uses_fallback_when_api_fails(self):
        from app.llm import run_json_step, LLMResult

        fallback = {"value": "fallback", "is_complete": False}

        with patch("app.llm.get_gemini") as mock_gemini:
            mock_gemini.return_value.generate_text.return_value = LLMResult(
                ok=False, text="", error="api_error"
            )
            obj, meta = run_json_step(
                system="test",
                prompt="test",
                schema=SimpleSchema,
                fallback=fallback,
            )

        assert obj.value == "fallback"
        assert meta["fallback_used"] is True

    def test_uses_fallback_when_json_invalid(self):
        from app.llm import run_json_step, LLMResult

        fallback = {"value": "fallback", "is_complete": False}

        with patch("app.llm.get_gemini") as mock_gemini:
            mock_gemini.return_value.generate_text.return_value = LLMResult(
                ok=True, text="not valid json at all"
            )
            obj, meta = run_json_step(
                system="test",
                prompt="test",
                schema=SimpleSchema,
                fallback=fallback,
            )

        assert obj.value == "fallback"
        assert meta["fallback_used"] is True

    def test_parses_valid_json_correctly(self):
        from app.llm import run_json_step, LLMResult

        with patch("app.llm.get_gemini") as mock_gemini:
            mock_gemini.return_value.generate_text.return_value = LLMResult(
                ok=True, text='{"value": "parsed", "is_complete": true}'
            )
            obj, meta = run_json_step(
                system="test",
                prompt="test",
                schema=SimpleSchema,
                fallback={"value": "fallback", "is_complete": False},
            )

        assert obj.value == "parsed"
        assert obj.is_complete is True
        assert meta["fallback_used"] is False