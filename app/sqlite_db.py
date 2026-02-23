from __future__ import annotations
import sqlite3
import time
import json
import uuid
import threading
import os
from pathlib import Path

from .settings import settings

_db_lock = threading.Lock()
_db_conn: sqlite3.Connection | None = None

DB_PATH = os.getenv("DB_PATH", "data/app.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def conn() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        Path(settings.app_db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(settings.app_db_path, timeout=10.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA busy_timeout=10000;")
        _db_conn = c
    return _db_conn

def _retry_db_operation(func, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if ("locked" in msg or "busy" in msg) and attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
                continue
            raise

def init_schema() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    if not schema_path.exists():
        raise RuntimeError(f"Schema file not found: {schema_path}")
    sql = schema_path.read_text(encoding="utf-8")

    def _init():
        with _db_lock:
            c = conn()
            c.executescript(sql)
            c.commit()

    _retry_db_operation(_init)

def exec_one(q: str, p: tuple = ()) -> None:
    def _exec():
        with _db_lock:
            c = conn()
            c.execute(q, p)
            c.commit()
    _retry_db_operation(_exec)

def fetch_one(q: str, p: tuple = ()):
    def _fetch():
        with _db_lock:
            c = conn()
            row = c.execute(q, p).fetchone()
            return dict(row) if row else None
    return _retry_db_operation(_fetch)

def fetch_all(q: str, p: tuple = ()):
    def _fetch():
        with _db_lock:
            c = conn()
            rows = c.execute(q, p).fetchall() or []
            return [dict(r) for r in rows]
    return _retry_db_operation(_fetch)


def create_session(thread_id: str):
    exec_one("INSERT INTO sessions (thread_id, status) VALUES (?, 'active')", (thread_id,))

def set_session_status(thread_id: str, status: str):
    exec_one("UPDATE sessions SET status=?, updated_at=datetime('now') WHERE thread_id=?", (status, thread_id))

def save_message(thread_id: str, role: str, text: str):
    exec_one("INSERT INTO messages (thread_id, role, text) VALUES (?,?,?)", (thread_id, role, text))


def get_idempotent_response(thread_id: str, key: str):
    return fetch_one(
        "SELECT response_json, request_hash FROM idempotency WHERE thread_id=? AND key=?",
        (thread_id, key),
    )

def save_idempotent_response(thread_id: str, key: str, request_hash: str, response_obj: dict):
    exec_one(
        "INSERT OR REPLACE INTO idempotency(thread_id, key, request_hash, response_json) VALUES (?,?,?,?)",
        (thread_id, key, request_hash, json.dumps(response_obj)),
    )


def get_stored_identity_by_name(name: str):
    q = "SELECT name, data_json FROM mock_ehr WHERE LOWER(TRIM(name))=LOWER(TRIM(?))"
    row = fetch_one(q, (name,))
    if not row:
        return None
    try:
        data = json.loads(row.get("data_json") or "{}")
    except Exception:
        data = {}
    ident = data.get("identity") or {}
    return {
        "name": row.get("name") or "",
        "phone": ident.get("phone", "") or "",
        "address": ident.get("address", "") or "",
    }


def create_escalation(thread_id: str, kind: str, payload: dict):
    exec_one(
        "INSERT INTO escalations (esc_id, thread_id, kind, payload_json) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), thread_id, kind, json.dumps(payload)),
    )

def list_pending_escalations():
    rows = fetch_all(
        "SELECT esc_id, thread_id, kind, payload_json, created_at FROM escalations WHERE resolved=0 ORDER BY created_at DESC"
    )
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {"error": "invalid_payload_json"}
        out.append({**r, "payload": payload})
    return out

def resolve_escalation(thread_id: str, esc_id: str, nurse_note: str = ""):
    exec_one(
        "UPDATE escalations SET resolved=1, nurse_note=? WHERE thread_id=? AND esc_id=?",
        (nurse_note, thread_id, esc_id),
    )


def save_report(thread_id: str, risk_level: str, visit_type: str, report_text: str):
    exec_one(
        "INSERT INTO reports (report_id, thread_id, risk_level, visit_type, report_text) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), thread_id, risk_level, visit_type, report_text),
    )

def get_latest_report(thread_id: str):
    return fetch_one(
        "SELECT rowid, * FROM reports WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
        (thread_id,),
    )

def max_upload_bytes() -> int:
    return int(settings.max_upload_size_mb) * 1024 * 1024

def create_job(thread_id: str, kind: str) -> str:
    job_id = str(uuid.uuid4())
    exec_one(
        "INSERT INTO jobs(job_id, thread_id, kind, status) VALUES (?,?,?,?)",
        (job_id, thread_id, kind, "queued"),
    )
    return job_id

def update_job(job_id: str, status: str, error: str | None = None):
    exec_one(
        "UPDATE jobs SET status=?, error=?, updated_at=datetime('now') WHERE job_id=?",
        (status, error, job_id),
    )

def get_job(job_id: str):
    return fetch_one("SELECT * FROM jobs WHERE job_id=?", (job_id,))


def save_session_state(thread_id: str, state_obj: dict):
    exec_one(
        "INSERT INTO session_state(thread_id, state_json, updated_at) VALUES (?,?,datetime('now')) "
        "ON CONFLICT(thread_id) DO UPDATE SET state_json=excluded.state_json, updated_at=datetime('now')",
        (thread_id, json.dumps(state_obj)),
    )

def get_session_state(thread_id: str):
    row = fetch_one("SELECT state_json, updated_at FROM session_state WHERE thread_id=?", (thread_id,))
    if not row:
        return None
    try:
        return {"state": json.loads(row["state_json"]), "updated_at": row["updated_at"]}
    except Exception:
        return None