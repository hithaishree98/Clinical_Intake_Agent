"""Baseline schema — all tables that exist at initial production deploy.

Revision ID: 001
Revises:
Create Date: 2026-04-19

This migration captures the full schema as it existed before Alembic was
introduced.  New deployments run this first; existing deployments that
already have the schema should stamp without running:

    alembic stamp 001

All subsequent schema changes (ADD COLUMN, CREATE INDEX, etc.) go in new
revision files — never edit this baseline.
"""
from __future__ import annotations

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("PRAGMA foreign_keys = ON")

    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id     TEXT PRIMARY KEY,
            status        TEXT NOT NULL DEFAULT 'active',
            session_token TEXT,
            patient_id    TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id  TEXT NOT NULL,
            role       TEXT NOT NULL,
            text       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_thread_created
        ON messages(thread_id, created_at)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id      TEXT PRIMARY KEY,
            thread_id      TEXT NOT NULL,
            risk_level     TEXT NOT NULL,
            visit_type     TEXT NOT NULL,
            report_text    TEXT NOT NULL,
            fhir_bundle    TEXT,
            pending_review INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_reports_thread_created
        ON reports(thread_id, created_at)
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_reports_patient ON reports(thread_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            esc_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL,
            kind         TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            resolved     INTEGER NOT NULL DEFAULT 0,
            nurse_note   TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_escalations_thread_resolved_created
        ON escalations(thread_id, resolved, created_at)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS mock_ehr (
            patient_id TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            history    TEXT,
            data_json  TEXT
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS idempotency (
            thread_id     TEXT NOT NULL,
            key           TEXT NOT NULL,
            request_hash  TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY(thread_id, key)
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id     TEXT PRIMARY KEY,
            thread_id  TEXT NOT NULL,
            kind       TEXT NOT NULL,
            status     TEXT NOT NULL,
            error      TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_thread ON jobs(thread_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS session_state (
            thread_id  TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS emergency_phrases (
            phrase   TEXT PRIMARY KEY,
            added_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_failure_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id    TEXT NOT NULL,
            node         TEXT NOT NULL,
            failure_type TEXT NOT NULL,
            raw_snippet  TEXT,
            error_detail TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_failure_thread_created
        ON llm_failure_log(thread_id, created_at)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_failure_type_created
        ON llm_failure_log(failure_type, created_at)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS webhook_deliveries (
            delivery_id      TEXT PRIMARY KEY,
            thread_id        TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            url_hash         TEXT NOT NULL,
            payload_hash     TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'pending',
            attempts         INTEGER NOT NULL DEFAULT 0,
            last_http_status INTEGER,
            last_error       TEXT,
            next_retry_at    TEXT,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_thread
        ON webhook_deliveries(thread_id, created_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status
        ON webhook_deliveries(status, next_retry_at)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id     TEXT NOT NULL,
            node          TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd      REAL NOT NULL DEFAULT 0.0,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_usage_thread
        ON llm_usage(thread_id, created_at)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS prompt_experiments (
            experiment_id TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            prompt_key    TEXT NOT NULL,
            variant_a     TEXT NOT NULL,
            variant_b     TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active',
            sessions_a    INTEGER NOT NULL DEFAULT 0,
            sessions_b    INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiments_status
        ON prompt_experiments(status, prompt_key)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_summary (
            patient_id             TEXT PRIMARY KEY,
            identity_json          TEXT NOT NULL DEFAULT '{}',
            allergies_json         TEXT NOT NULL DEFAULT '[]',
            medications_json       TEXT NOT NULL DEFAULT '[]',
            conditions_json        TEXT NOT NULL DEFAULT '[]',
            recent_complaints_json TEXT NOT NULL DEFAULT '[]',
            flags_json             TEXT NOT NULL DEFAULT '[]',
            visit_count            INTEGER NOT NULL DEFAULT 0,
            first_seen_at          TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at             TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def downgrade() -> None:
    for table in [
        "patient_summary", "prompt_experiments", "llm_usage",
        "webhook_deliveries", "llm_failure_log", "emergency_phrases",
        "session_state", "jobs", "idempotency", "mock_ehr",
        "escalations", "reports", "messages", "sessions",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table}")
