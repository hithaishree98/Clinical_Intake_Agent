import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("CLINICIAN_PASSWORD", "test-password")
os.environ.setdefault("DEBUG_MODE", "true")


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Spin up a fresh SQLite database for each test so tests never share state."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("APP_DB_PATH", db_file)

    from app import sqlite_db as db
    import importlib
    importlib.reload(db)
    db.init_schema()

    yield db_file
