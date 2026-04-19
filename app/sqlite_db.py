from __future__ import annotations
import hashlib
import hmac
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from .logging_utils import log_event
from .settings import get_settings as settings

_db_lock = threading.Lock()
_db_conn: sqlite3.Connection | None = None

def conn() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        Path(settings().app_db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(settings().app_db_path, timeout=10.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA busy_timeout=10000;")
        _db_conn = c
    return _db_conn


@contextmanager
def transaction():
    global _db_conn
    with _db_lock:
        c = conn()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise

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
            # Migration: add fhir_bundle column to reports if it doesn't exist yet.
            existing_reports = {
                row[1]
                for row in c.execute("PRAGMA table_info(reports)").fetchall()
            }
            if "fhir_bundle" not in existing_reports:
                c.execute("ALTER TABLE reports ADD COLUMN fhir_bundle TEXT")
            if "pending_review" not in existing_reports:
                c.execute(
                    "ALTER TABLE reports ADD COLUMN pending_review INTEGER NOT NULL DEFAULT 0"
                )
            # Migration: add session_token column to sessions for patient auth.
            existing_sessions = {
                row[1]
                for row in c.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "session_token" not in existing_sessions:
                c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
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


def create_session(thread_id: str, session_token: str | None = None):
    # Store the hash, never the raw token. Verification uses the same hash path.
    token_hash = _hash_token(session_token) if session_token else None
    exec_one(
        "INSERT INTO sessions (thread_id, status, session_token) VALUES (?, 'active', ?)",
        (thread_id, token_hash),
    )


def _hash_token(token: str) -> str:
    """SHA-256 the token before storage/comparison. The raw token is never persisted."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def derive_patient_id(name: str, dob: str) -> str | None:
    """
    Deterministic patient_id from normalized (name, dob).
    Returns None until both fields are known — callers should defer
    persistence until identity is captured.

    Rationale: using a hash rather than a raw join key means the ID
    itself carries no PHI and can appear in logs safely.
    """
    n = (name or "").strip().lower()
    d = (dob or "").strip()
    if not n or not d:
        return None
    raw = f"{n}|{d}".encode("utf-8")
    return "pat_" + hashlib.sha256(raw).hexdigest()[:16]


def set_session_patient_id(thread_id: str, patient_id: str) -> None:
    exec_one(
        "UPDATE sessions SET patient_id=?, updated_at=datetime('now') WHERE thread_id=?",
        (patient_id, thread_id),
    )

def verify_session_token(thread_id: str, token: str) -> bool:
    """
    Constant-time comparison of the token against its stored hash.
    Tokens are stored as SHA-256(token) so a DB read never exposes the
    raw bearer value.
    """
    row = fetch_one(
        "SELECT session_token FROM sessions WHERE thread_id=?",
        (thread_id,),
    )
    if not row:
        return False
    stored = row.get("session_token") or ""
    return bool(stored) and hmac.compare_digest(stored, _hash_token(token))

def set_session_status(thread_id: str, status: Literal["active", "done", "expired", "escalated"]):
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

def persist_chat_turn(
    *,
    thread_id: str,
    user_message: str,
    assistant_reply: str,
    state_snapshot: dict,
    status: str,
    client_msg_id: str,
    request_hash: str,
    response_obj: dict,
    job_id: str | None = None,
) -> None:
    with transaction() as c:
        c.execute(
            "INSERT INTO session_state(thread_id, state_json, updated_at) VALUES (?,?,datetime('now')) "
            "ON CONFLICT(thread_id) DO UPDATE SET state_json=excluded.state_json, updated_at=datetime('now')",
            (thread_id, json.dumps(state_snapshot)),
        )
        c.execute(
            "INSERT INTO messages (thread_id, role, text) VALUES (?,?,?)",
            (thread_id, "user", user_message),
        )
        c.execute(
            "INSERT INTO messages (thread_id, role, text) VALUES (?,?,?)",
            (thread_id, "assistant", assistant_reply),
        )
        c.execute(
            "UPDATE sessions SET status=?, updated_at=datetime('now') WHERE thread_id=?",
            (status, thread_id),
        )
        c.execute(
            "INSERT OR REPLACE INTO idempotency(thread_id, key, request_hash, response_json) VALUES (?,?,?,?)",
            (thread_id, client_msg_id, request_hash, json.dumps(response_obj)),
        )
        if job_id:
            c.execute(
                "INSERT INTO jobs(job_id, thread_id, kind, status) VALUES (?,?,?,?)",
                (job_id, thread_id, "report", "queued"),
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
        "dob": ident.get("dob", "") or "",
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


def save_report(
    thread_id: str,
    risk_level: str,
    visit_type: str,
    report_text: str,
    fhir_bundle: str | None = None,
    pending_review: bool = False,
):
    """
    Persist a completed clinician report.

    pending_review=True means human review is advised before acting on this
    report (e.g. soft safety signals were present).  Does NOT mean the report
    is blocked — it was generated and saved, but clinicians should double-check.
    """
    exec_one(
        "INSERT INTO reports"
        " (report_id, thread_id, risk_level, visit_type, report_text, fhir_bundle, pending_review)"
        " VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), thread_id, risk_level, visit_type, report_text,
         fhir_bundle, int(pending_review)),
    )

def get_patient_summary(patient_id: str) -> dict | None:
    """
    Return the Layer-2 summary for this patient, or None if first visit.
    Shape:
      { identity, allergies, medications, conditions,
        recent_complaints, flags, visit_count, first_seen_at, updated_at }
    """
    row = fetch_one(
        "SELECT * FROM patient_summary WHERE patient_id=?", (patient_id,)
    )
    if not row:
        return None
    out = {
        "patient_id":        row["patient_id"],
        "visit_count":       row["visit_count"],
        "first_seen_at":     row["first_seen_at"],
        "updated_at":        row["updated_at"],
    }
    for key, col in [
        ("identity",          "identity_json"),
        ("allergies",         "allergies_json"),
        ("medications",       "medications_json"),
        ("conditions",        "conditions_json"),
        ("recent_complaints", "recent_complaints_json"),
        ("flags",             "flags_json"),
    ]:
        try:
            out[key] = json.loads(row[col])
        except Exception:
            out[key] = {} if key == "identity" else []
    return out


def upsert_patient_summary(patient_id: str, merged: dict) -> None:
    """
    Write the merged summary back. Caller is responsible for merge logic —
    this function is dumb persistence only.
    """
    exec_one(
        """
        INSERT INTO patient_summary(
          patient_id, identity_json, allergies_json, medications_json,
          conditions_json, recent_complaints_json, flags_json,
          visit_count, first_seen_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(patient_id) DO UPDATE SET
          identity_json          = excluded.identity_json,
          allergies_json         = excluded.allergies_json,
          medications_json       = excluded.medications_json,
          conditions_json        = excluded.conditions_json,
          recent_complaints_json = excluded.recent_complaints_json,
          flags_json             = excluded.flags_json,
          visit_count            = excluded.visit_count,
          updated_at             = datetime('now')
        """,
        (
            patient_id,
            json.dumps(merged.get("identity") or {}),
            json.dumps(merged.get("allergies") or []),
            json.dumps(merged.get("medications") or []),
            json.dumps(merged.get("conditions") or []),
            json.dumps(merged.get("recent_complaints") or []),
            json.dumps(merged.get("flags") or []),
            int(merged.get("visit_count") or 1),
        ),
    )


def get_fhir_bundle(thread_id: str) -> str | None:
    """Return the raw FHIR R4 Bundle JSON for the latest report, or None."""
    row = fetch_one(
        "SELECT fhir_bundle FROM reports WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
        (thread_id,),
    )
    if not row:
        return None
    return row.get("fhir_bundle")

def get_latest_report(thread_id: str):
    return fetch_one(
        "SELECT rowid, * FROM reports WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
        (thread_id,),
    )

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

def get_jobs_for_thread(thread_id: str) -> list:
    return fetch_all("SELECT * FROM jobs WHERE thread_id=? ORDER BY created_at DESC", (thread_id,))

def mark_stale_jobs_failed(stale_minutes: int = 10) -> int:
    """
    Mark jobs that have been 'running' for longer than stale_minutes as failed.
    Called on job-status reads so stale background tasks are always surfaced.
    Returns the number of rows updated.
    """
    def _mark():
        with _db_lock:
            c = conn()
            cur = c.execute(
                "UPDATE jobs SET status='failed', "
                "error='stale: job exceeded time limit without completing', "
                "updated_at=datetime('now') "
                "WHERE status='running' AND updated_at <= datetime('now', ?)",
                (f"-{stale_minutes} minutes",),
            )
            c.commit()
            return cur.rowcount
    return _retry_db_operation(_mark)


def record_llm_failure(
    thread_id: str,
    node: str,
    failure_type: str,
    raw_snippet: str | None = None,
    error_detail: str | None = None,
) -> None:
    """
    Persist one LLM failure event to llm_failure_log.

    failure_type must be one of:
      "fallback_used"  — parse failed; hardcoded fallback was returned
      "repair_used"    — first parse failed; repair attempt succeeded
      "parse_error"    — parse failed; repair also failed
      "api_error"      — LLM API call itself returned an error
    """
    exec_one(
        "INSERT INTO llm_failure_log (thread_id, node, failure_type, raw_snippet, error_detail) "
        "VALUES (?,?,?,?,?)",
        (thread_id, node, failure_type, (raw_snippet or "")[:300], error_detail),
    )


def get_llm_failure_stats(days: int = 7) -> dict:
    """Return LLM failure counts by type over the last N days."""
    date_bound = f"-{days} days"
    rows = fetch_all(
        "SELECT failure_type, COUNT(*) AS n FROM llm_failure_log "
        "WHERE created_at >= date('now', ?) GROUP BY failure_type",
        (date_bound,),
    )
    by_type = {r["failure_type"]: r["n"] for r in rows}
    total = fetch_one(
        "SELECT COUNT(*) AS n FROM llm_failure_log WHERE created_at >= date('now', ?)",
        (date_bound,),
    ) or {}
    top_node = fetch_one(
        "SELECT node, COUNT(*) AS n FROM llm_failure_log "
        "WHERE created_at >= date('now', ?) GROUP BY node ORDER BY n DESC LIMIT 1",
        (date_bound,),
    )
    return {
        "total_llm_failures": total.get("n", 0),
        "fallbacks_used":     by_type.get("fallback_used", 0),
        "repairs_used":       by_type.get("repair_used", 0),
        "parse_errors":       by_type.get("parse_error", 0),
        "api_errors":         by_type.get("api_error", 0),
        "most_failing_node":  (top_node or {}).get("node"),
    }


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


def get_emergency_phrases() -> list[str]:
    """
    Load emergency phrases from the DB. Falls back to an empty list if the
    table hasn't been seeded yet — the caller should handle the fallback.
    """
    rows = fetch_all("SELECT phrase FROM emergency_phrases ORDER BY phrase ASC")
    return [r["phrase"] for r in rows]


def add_emergency_phrase(phrase: str) -> None:
    exec_one(
        "INSERT OR IGNORE INTO emergency_phrases (phrase) VALUES (?)",
        (phrase.strip().lower(),),
    )


def delete_emergency_phrase(phrase: str) -> bool:
    """Returns True if a row was deleted, False if phrase didn't exist."""
    def _del():
        with _db_lock:
            c = conn()
            cur = c.execute(
                "DELETE FROM emergency_phrases WHERE phrase = ?",
                (phrase.strip().lower(),),
            )
            c.commit()
            return cur.rowcount > 0
    return _retry_db_operation(_del)


def seed_emergency_phrases(phrases: list[str]) -> None:
    """Insert defaults, skipping any that already exist."""
    for p in phrases:
        add_emergency_phrase(p)


# ---------------------------------------------------------------------------
# LLM token usage tracking
# ---------------------------------------------------------------------------

# Gemini 2.0 Flash pricing (USD per 1 000 tokens, as of 2025-Q2).
# Update GEMINI_INPUT_COST_PER_1K / OUTPUT when pricing changes.
_GEMINI_INPUT_COST_PER_1K  = 0.000075   # $0.075 / 1M input tokens
_GEMINI_OUTPUT_COST_PER_1K = 0.0003     # $0.30  / 1M output tokens


def record_llm_usage(
    thread_id: str,
    node: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """
    Persist token counts and cost for one LLM call.

    cost_usd is computed here so billing queries never need to re-derive it.
    Failures are silently swallowed — usage tracking must never break the call path.
    """
    try:
        cost = (
            (input_tokens  / 1000) * _GEMINI_INPUT_COST_PER_1K
            + (output_tokens / 1000) * _GEMINI_OUTPUT_COST_PER_1K
        )
        exec_one(
            "INSERT INTO llm_usage (thread_id, node, input_tokens, output_tokens, cost_usd)"
            " VALUES (?,?,?,?,?)",
            (thread_id, node, input_tokens, output_tokens, round(cost, 8)),
        )
    except Exception:
        pass  # never break the happy path over accounting


def get_llm_usage_for_thread(thread_id: str) -> dict:
    """Return aggregated token counts and cost for a single session."""
    row = fetch_one(
        "SELECT SUM(input_tokens) AS inp, SUM(output_tokens) AS out,"
        "       SUM(cost_usd) AS cost, COUNT(*) AS calls"
        " FROM llm_usage WHERE thread_id=?",
        (thread_id,),
    ) or {}
    return {
        "total_input_tokens":  row.get("inp") or 0,
        "total_output_tokens": row.get("out") or 0,
        "total_cost_usd":      round(row.get("cost") or 0.0, 6),
        "llm_calls":           row.get("calls") or 0,
    }


# ---------------------------------------------------------------------------
# Webhook delivery tracking
# ---------------------------------------------------------------------------

def create_webhook_delivery(
    delivery_id: str,
    thread_id: str,
    event_type: str,
    url_hash: str,
    payload_hash: str,
) -> None:
    """Insert a new delivery record in 'pending' status."""
    exec_one(
        "INSERT INTO webhook_deliveries"
        " (delivery_id, thread_id, event_type, url_hash, payload_hash, status, attempts)"
        " VALUES (?,?,?,?,?,'pending',0)",
        (delivery_id, thread_id, event_type, url_hash, payload_hash),
    )


def update_webhook_delivery(
    delivery_id: str,
    *,
    status: str,
    attempts: int,
    last_http_status: int | None = None,
    last_error: str | None = None,
    next_retry_at: str | None = None,
) -> None:
    exec_one(
        "UPDATE webhook_deliveries"
        " SET status=?, attempts=?, last_http_status=?, last_error=?,"
        "     next_retry_at=?, updated_at=datetime('now')"
        " WHERE delivery_id=?",
        (status, attempts, last_http_status, last_error, next_retry_at, delivery_id),
    )


def get_webhook_delivery_by_hash(
    thread_id: str,
    event_type: str,
    payload_hash: str,
) -> dict | None:
    """Return the delivery record if an identical payload was already dispatched."""
    return fetch_one(
        "SELECT * FROM webhook_deliveries"
        " WHERE thread_id=? AND event_type=? AND payload_hash=?"
        " ORDER BY created_at DESC LIMIT 1",
        (thread_id, event_type, payload_hash),
    )


def get_webhook_deliveries(thread_id: str | None = None, limit: int = 50) -> list:
    """Return recent delivery records, optionally filtered to one thread."""
    if thread_id:
        return fetch_all(
            "SELECT * FROM webhook_deliveries WHERE thread_id=?"
            " ORDER BY created_at DESC LIMIT ?",
            (thread_id, limit),
        )
    return fetch_all(
        "SELECT * FROM webhook_deliveries ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )


def get_exhausted_webhooks(older_than_hours: int = 24, max_lifetime_attempts: int = 6) -> list:
    """
    Return webhook delivery records that are exhausted AND have not exceeded
    the lifetime attempt cap.

    These are candidates for a dead-letter re-queue: they failed all original
    retry attempts but may succeed now (e.g. the downstream EHR is back up).
    The max_lifetime_attempts guard prevents infinite retry storms against a
    permanently broken endpoint.
    """
    return fetch_all(
        "SELECT * FROM webhook_deliveries"
        " WHERE status='exhausted'"
        "   AND attempts < ?"
        "   AND updated_at <= datetime('now', ? || ' hours')"
        " ORDER BY updated_at ASC"
        " LIMIT 100",
        (max_lifetime_attempts, f"-{older_than_hours}"),
    )


def requeue_webhook_delivery(delivery_id: str) -> None:
    """
    Reset an exhausted delivery to 'pending' so the dispatcher will retry it.

    Keeps the existing attempt count intact so the lifetime cap in
    get_exhausted_webhooks() continues to work correctly.
    """
    exec_one(
        "UPDATE webhook_deliveries"
        " SET status='pending', next_retry_at=datetime('now'),"
        "     last_error='requeued_by_dead_letter_worker',"
        "     updated_at=datetime('now')"
        " WHERE delivery_id=?",
        (delivery_id,),
    )


def expire_stale_sessions(ttl_hours: int = 4) -> int:
    """
    Mark sessions that have been 'active' (not done/escalated/expired) for longer
    than ttl_hours as 'expired'.

    A patient who abandons mid-intake stays active forever without this.
    Expired sessions appear in the clinician dashboard so abandoned intakes
    are visible rather than silently lingering.

    Returns the number of sessions marked expired.
    """
    def _expire():
        with _db_lock:
            c = conn()
            cur = c.execute(
                "UPDATE sessions SET status='expired', updated_at=datetime('now')"
                " WHERE status='active'"
                "   AND updated_at <= datetime('now', ?)",
                (f"-{ttl_hours} hours",),
            )
            c.commit()
            return cur.rowcount
    return _retry_db_operation(_expire)


def prune_old_checkpoints(days: int = 30) -> int:
    """
    Delete LangGraph checkpoint rows for sessions completed more than `days` ago.
    """
    from .settings import get_settings
    checkpoint_path = get_settings().checkpoint_db_path
    try:
        rows = fetch_all(
            "SELECT thread_id FROM sessions "
            "WHERE status IN ('done', 'expired') "
            "AND updated_at <= datetime('now', ?)",
            (f"-{days} days",),
        )
        thread_ids = [r["thread_id"] for r in rows]
        if not thread_ids:
            return 0

        cp = sqlite3.connect(checkpoint_path, timeout=5.0)
        placeholders = ",".join("?" for _ in thread_ids)

        total_deleted = 0
        for table in ("checkpoints", "checkpoint_writes"):
            try:
                cur = cp.execute(
                    f"DELETE FROM {table} WHERE thread_id IN ({placeholders})",
                    thread_ids,
                )
                total_deleted += cur.rowcount
            except sqlite3.OperationalError:
                pass

        cp.commit()
        cp.close()
        return total_deleted
    except Exception:
        return 0


def reset_demo_data() -> None:
    """Wipe all session data. Does not touch emergency_phrases or checkpoint DB."""
    def _reset():
        with _db_lock:
            c = conn()
            for table in [
                "sessions", "messages", "reports", "escalations",
                "jobs", "session_state", "idempotency", "llm_failure_log",
                "webhook_deliveries",
            ]:
                c.execute(f"DELETE FROM {table}")
            c.commit()
    _retry_db_operation(_reset)


_DEMO_PATIENTS = [
    {
        "patient_id": "demo-ava",
        "name": "Ava Johnson",
        "history": "Prior visit: Hypertension. Penicillin allergy.",
        "data_json": json.dumps({
            "identity": {"dob": "03/15/1985", "phone": "4125550199", "address": "100 Forbes Ave, Pittsburgh, PA"},
            "allergies": ["penicillin"],
            "medications": ["lisinopril 10mg daily"],
            "pmh": ["hypertension"],
            "recent_results": ["CBC normal (2025-11-10)"],
        }),
    },
    {
        "patient_id": "demo-marcus",
        "name": "Marcus Thorne",
        "history": "Prior cardiac stent placement in 2023.",
        "data_json": json.dumps({
            "identity": {"dob": "07/22/1970", "phone": "5550388844", "address": "12 Market St, Pittsburgh, PA"},
            "allergies": [],
            "medications": ["atorvastatin 40mg nightly"],
            "pmh": ["coronary artery disease", "cardiac stent (2023)"],
            "recent_results": [],
        }),
    },
    {
        "patient_id": "demo-nina",
        "name": "Nina Shah",
        "history": "Prior visit: Anxiety. No known drug allergies.",
        "data_json": json.dumps({
            "identity": {"dob": "11/03/1992", "phone": "5557772222", "address": "44 Walnut St, Chicago, IL"},
            "allergies": [],
            "medications": [],
            "pmh": ["anxiety"],
            "recent_results": [],
        }),
    },
]


def seed_demo_patients() -> None:
    """Re-seed mock EHR demo patients, replacing any existing demo rows."""
    def _seed():
        with _db_lock:
            c = conn()
            c.execute("DELETE FROM mock_ehr WHERE patient_id LIKE 'demo-%'")
            for p in _DEMO_PATIENTS:
                c.execute(
                    "INSERT INTO mock_ehr (patient_id, name, history, data_json) VALUES (?,?,?,?)",
                    (p["patient_id"], p["name"], p["history"], p["data_json"]),
                )
            c.commit()
    _retry_db_operation(_seed)



# ---------------------------------------------------------------------------
# Prompt A/B experiments
# ---------------------------------------------------------------------------

def create_experiment(name: str, prompt_key: str, variant_a: str, variant_b: str) -> str:
    """Create a new A/B experiment. Returns the experiment_id."""
    exp_id = str(uuid.uuid4())
    exec_one(
        "INSERT INTO prompt_experiments (experiment_id, name, prompt_key, variant_a, variant_b)"
        " VALUES (?,?,?,?,?)",
        (exp_id, name, prompt_key, variant_a, variant_b),
    )
    return exp_id


def get_active_experiment(prompt_key: str) -> dict | None:
    """Return the active experiment for a given prompt key, or None."""
    return fetch_one(
        "SELECT * FROM prompt_experiments WHERE status='active' AND prompt_key=? LIMIT 1",
        (prompt_key,),
    )


def assign_experiment_variant(thread_id: str, experiment_id: str) -> str:
    """
    Deterministically assign variant 'a' or 'b' based on thread_id hash.
    Same thread always gets the same variant. Increments the session counter.
    Returns 'a' or 'b'.
    """
    import hashlib
    variant = "a" if int(hashlib.md5(thread_id.encode()).hexdigest(), 16) % 2 == 0 else "b"
    col = f"sessions_{variant}"
    exec_one(
        f"UPDATE prompt_experiments SET {col}={col}+1, updated_at=datetime('now')"
        " WHERE experiment_id=?",
        (experiment_id,),
    )
    return variant


def resolve_prompt_variant(thread_id: str, prompt_key: str, default_version: str) -> tuple[str, str | None]:
    """
    Returns (version_string, experiment_id_or_None).

    If an active experiment exists for prompt_key, the session is assigned to
    variant_a or variant_b deterministically (md5(thread_id) % 2) and the
    corresponding version string is returned.  The session counter is bumped.

    If no active experiment, returns (default_version, None).
    """
    exp = get_active_experiment(prompt_key)
    if not exp:
        return default_version, None
    variant = assign_experiment_variant(thread_id, exp["experiment_id"])
    version = exp[f"variant_{variant}"]
    return version, exp["experiment_id"]


def list_experiments() -> list:
    return fetch_all("SELECT * FROM prompt_experiments ORDER BY created_at DESC")


def update_experiment_status(experiment_id: str, status: str) -> None:
    exec_one(
        "UPDATE prompt_experiments SET status=?, updated_at=datetime('now') WHERE experiment_id=?",
        (status, experiment_id),
    )


def get_analytics() -> dict:
    """Return operational metrics over the last 7 days."""
    rows_today = fetch_one(
        "SELECT COUNT(*) AS n FROM sessions WHERE created_at >= date('now')"
    ) or {}
    rows_week = fetch_one(
        "SELECT COUNT(*) AS n FROM sessions WHERE created_at >= date('now', '-7 days')"
    ) or {}
    completed = fetch_one(
        "SELECT COUNT(*) AS n FROM sessions WHERE status = 'done' "
        "AND created_at >= date('now', '-7 days')"
    ) or {}
    escalations_week = fetch_one(
        "SELECT COUNT(*) AS n FROM escalations "
        "WHERE created_at >= date('now', '-7 days')"
    ) or {}
    emergency_esc = fetch_one(
        "SELECT COUNT(*) AS n FROM escalations "
        "WHERE kind = 'emergency' AND created_at >= date('now', '-7 days')"
    ) or {}
    reports_week = fetch_one(
        "SELECT COUNT(*) AS n FROM reports "
        "WHERE created_at >= date('now', '-7 days')"
    ) or {}
    failed_jobs = fetch_one(
        "SELECT COUNT(*) AS n FROM jobs WHERE status = 'failed' "
        "AND created_at >= date('now', '-7 days')"
    ) or {}
    pending_esc = fetch_one(
        "SELECT COUNT(*) AS n FROM escalations WHERE resolved = 0"
    ) or {}

    total_week = rows_week.get("n") or 0
    done_week = completed.get("n") or 0
    completion_rate = round(done_week / total_week * 100, 1) if total_week else 0.0

    llm_stats = get_llm_failure_stats(days=7)

    cost_row = fetch_one(
        "SELECT COALESCE(SUM(CAST(input_tokens AS REAL)/1000000.0*0.075 "
        "     + CAST(output_tokens AS REAL)/1000000.0*0.30), 0.0) AS cost "
        "FROM llm_usage WHERE created_at >= date('now', '-7 days')"
    ) or {}
    llm_cost_7d = round(float(cost_row.get("cost") or 0.0), 6)

    sessions_with_cost = fetch_one(
        "SELECT COUNT(DISTINCT thread_id) AS n FROM llm_usage "
        "WHERE created_at >= date('now', '-7 days')"
    ) or {}
    sessions_n = sessions_with_cost.get("n") or 0
    avg_cost = round(llm_cost_7d / sessions_n, 6) if sessions_n else 0.0

    cost_today_row = fetch_one(
        "SELECT COALESCE(SUM(CAST(input_tokens AS REAL)/1000000.0*0.075 "
        "     + CAST(output_tokens AS REAL)/1000000.0*0.30), 0.0) AS cost "
        "FROM llm_usage WHERE created_at >= date('now')"
    ) or {}
    llm_cost_today = round(float(cost_today_row.get("cost") or 0.0), 6)

    # Repair rate over the last hour — a sudden spike signals a prompt regression
    # that silently doubles LLM cost for affected calls.
    calls_1h = fetch_one(
        "SELECT COUNT(*) AS n FROM llm_usage "
        "WHERE created_at >= datetime('now', '-1 hour')"
    ) or {}
    repairs_1h = fetch_one(
        "SELECT COUNT(*) AS n FROM llm_failure_log "
        "WHERE failure_type = 'repair_used' "
        "AND created_at >= datetime('now', '-1 hour')"
    ) or {}
    calls_1h_n   = calls_1h.get("n") or 0
    repairs_1h_n = repairs_1h.get("n") or 0
    repair_rate_1h = round(repairs_1h_n / calls_1h_n, 3) if calls_1h_n else 0.0

    _repair_threshold = settings().intake.repair_rate_alert_threshold
    repair_rate_alert = repair_rate_1h > _repair_threshold

    if repair_rate_alert:
        log_event(
            "repair_rate_alert",
            level="warning",
            repair_rate_1h=repair_rate_1h,
            threshold=_repair_threshold,
            calls=calls_1h_n,
            repairs=repairs_1h_n,
        )

    return {
        "sessions_today": rows_today.get("n", 0),
        "sessions_last_7_days": total_week,
        "completed_last_7_days": done_week,
        "completion_rate_pct": completion_rate,
        "escalations_last_7_days": escalations_week.get("n", 0),
        "emergency_escalations_last_7_days": emergency_esc.get("n", 0),
        "pending_escalations": pending_esc.get("n", 0),
        "reports_generated_last_7_days": reports_week.get("n", 0),
        "failed_report_jobs_last_7_days": failed_jobs.get("n", 0),
        "llm_failures_last_7_days": llm_stats,
        "llm_cost_last_7_days_usd": llm_cost_7d,
        "llm_cost_today_usd": llm_cost_today,
        "avg_cost_per_session_usd": avg_cost,
        "repair_rate_last_1h": repair_rate_1h,
        "repair_rate_alert": repair_rate_alert,
    }
