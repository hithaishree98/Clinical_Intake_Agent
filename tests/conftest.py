import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("CLINICIAN_PASSWORD", "test-password")
os.environ.setdefault("DEBUG_MODE", "true")


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """
    Reset the LLM circuit breaker before every test, module-wide.

    Any test that triggers a real (failing) LLM call — because run_json_step
    is not mocked — increments the breaker's failure counter.  Without this
    reset, accumulated failures from earlier tests trip the breaker and cause
    subsequent tests to fail with 503 at the /chat guard, even when the test
    itself correctly mocks the LLM.
    """
    import app.llm as llm_mod
    breaker = llm_mod._breaker
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

    # Patch the settings object and the module-level connection cache directly.
    # Reloading modules would invalidate other modules' imported references, so
    # we mutate the live settings object and force a fresh DB connection instead.
    original_path = settings.app_db_path
    original_conn = db._db_conn

    monkeypatch.setattr(settings, "app_db_path", db_file)
    db._db_conn = None  # force conn() to open the new file

    db.init_schema()

    yield db_file

    # Restore everything so other tests are not affected.
    if db._db_conn is not None:
        try:
            db._db_conn.close()
        except Exception:
            pass
    db._db_conn = original_conn
    monkeypatch.setattr(settings, "app_db_path", original_path)
