import os
import pytest

# Force test values rather than setdefault().  When tests run inside Docker
# compose (env_file: .env) or in any shell that has sourced .env, the real
# values end up in os.environ already, so setdefault becomes a no-op and the
# auth tests fail because they log in with "test-password" while settings
# carry the developer's real password.  Forcing makes the suite hermetic:
# the same `pytest` command produces the same result whether run on the
# host, inside Docker, or from an IDE that injects .env.
os.environ["GEMINI_API_KEY"]      = "test-key"
os.environ["JWT_SECRET"]          = "test-secret"
os.environ["CLINICIAN_PASSWORD"]  = "test-password"
os.environ["DEBUG_MODE"]          = "true"
# Voice is opt-in; tests that exercise /transcribe set their own key via
# the voice_client fixture (see tests/test_voice.py).  Default off so the
# regression suite never accidentally calls the real Groq API.
os.environ["GROQ_API_KEY"]        = ""
os.environ["VOICE_ENABLED"]       = "false"


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """
    Reset the LLM circuit breaker before every test, module-wide.

    Any test that triggers a real (failing) LLM call — because run_json_step
    is not mocked — increments the breaker's failure counter.  Without this
    reset, accumulated failures from earlier tests trip the breaker and cause
    subsequent tests to fail with 503 at the /chat guard, even when the test
    itself correctly mocks the LLM.

    Use ``_get_breaker()`` (not the raw module-level ``_breaker``) because the
    breaker is lazily initialised on first use.  Reading ``_breaker`` before any
    LLM path has run yields ``None`` and the ``with breaker._lock`` line below
    raises AttributeError, which would crash the very first test of the suite.
    """
    import app.llm as llm_mod
    breaker = llm_mod._get_breaker()
    with breaker._lock:
        breaker._state     = llm_mod.CircuitBreaker.CLOSED
        breaker._failures  = 0
        breaker._opened_at = 0.0
    yield


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Spin up a fresh SQLite database for each test so tests never share state."""
    db_file = str(tmp_path / "test.db")

    from app import sqlite_db as db
    from app.settings import get_settings

    settings = get_settings()

    # Patch the settings object and close any per-thread connections so the
    # next conn() call opens against the new file.  Reloading modules would
    # invalidate other modules' imported references, so we mutate the live
    # settings object instead.
    original_path = settings.app_db_path

    monkeypatch.setattr(settings, "app_db_path", db_file)
    db.close_all_connections()

    db.init_schema()

    yield db_file

    # Close every per-thread connection opened during the test before
    # restoring the original path, so the next test that uses tmp_db does
    # not inherit a connection pointing at this test's (deleted) DB file.
    db.close_all_connections()
    monkeypatch.setattr(settings, "app_db_path", original_path)
