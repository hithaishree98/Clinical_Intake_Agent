import pytest
from pydantic import BaseModel

from app.llm import LLMProvider, LLMResult, set_provider


class _StubProvider(LLMProvider):
    """
    Test double that returns a pre-configured LLMResult on every call.

    Used in TestLLMFallback to drive run_json_step deterministically without
    network access.  Plug it in via set_provider(); the registry's circuit
    breaker is reset automatically on swap.
    """
    def __init__(self, result: LLMResult) -> None:
        self._result = result

    def generate_text(self, *, system, prompt, temperature=0.2, max_tokens=900,
                      response_mime_type="application/json", cache_key=None):
        return self._result

    def validate(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def input_cost_per_million(self) -> float:
        return 0.0

    @property
    def output_cost_per_million(self) -> float:
        return 0.0


@pytest.fixture
def restore_provider():
    """
    Save / restore the registry's provider so swapping in a stub for one
    test doesn't leak into the next.
    """
    import app.llm.registry as reg
    original = reg._provider
    yield
    reg._provider = original


# Idempotency

class TestIdempotency:
    def test_same_message_returns_cached_response(self, tmp_db):
        from app import sqlite_db as db

        thread_id = "test-thread-1"
        db.create_session(thread_id)

        response = {"reply": "hello", "status": "active", "phase": "identity"}
        db.save_idempotent_response(
            thread_id,
            key="msg-1",
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
    def test_uses_fallback_when_api_fails(self, restore_provider):
        from app.llm import run_json_step

        set_provider(_StubProvider(LLMResult(ok=False, text="", error="api_error")))
        obj, meta = run_json_step(
            system="test",
            prompt="test",
            schema=SimpleSchema,
            fallback={"value": "fallback", "is_complete": False},
        )

        assert obj.value == "fallback"
        assert meta["fallback_used"] is True

    def test_uses_fallback_when_json_invalid(self, restore_provider):
        from app.llm import run_json_step

        set_provider(_StubProvider(LLMResult(ok=True, text="not valid json at all")))
        obj, meta = run_json_step(
            system="test",
            prompt="test",
            schema=SimpleSchema,
            fallback={"value": "fallback", "is_complete": False},
        )

        assert obj.value == "fallback"
        assert meta["fallback_used"] is True

    def test_parses_valid_json_correctly(self, restore_provider):
        from app.llm import run_json_step

        set_provider(_StubProvider(LLMResult(
            ok=True, text='{"value": "parsed", "is_complete": true}'
        )))
        obj, meta = run_json_step(
            system="test",
            prompt="test",
            schema=SimpleSchema,
            fallback={"value": "fallback", "is_complete": False},
        )

        assert obj.value == "parsed"
        assert obj.is_complete is True
        assert meta["fallback_used"] is False