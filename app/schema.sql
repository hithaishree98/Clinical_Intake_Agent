PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  thread_id    TEXT PRIMARY KEY,
  status       TEXT NOT NULL DEFAULT 'active',
  session_token TEXT,
  patient_id   TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id TEXT NOT NULL,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
  report_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  visit_type TEXT NOT NULL,
  report_text TEXT NOT NULL,
  fhir_bundle TEXT,            -- FHIR R4 Bundle JSON, nullable for older rows
  pending_review INTEGER NOT NULL DEFAULT 0,  -- 1 when human review is advised before acting on this report
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS escalations (
  esc_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  kind TEXT NOT NULL,         -- "identity_review" | "emergency" | "crisis" | "report_blocked"
  payload_json TEXT NOT NULL,
  resolved INTEGER NOT NULL DEFAULT 0,
  nurse_note TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Demo storage 
CREATE TABLE IF NOT EXISTS mock_ehr (
  patient_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  history TEXT,
  data_json TEXT
);

CREATE TABLE IF NOT EXISTS idempotency (
  thread_id TEXT NOT NULL,
  key TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(thread_id, key)
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,        -- queued|running|done|failed
  error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_state (
  thread_id TEXT PRIMARY KEY,
  state_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_thread ON jobs(thread_id);

CREATE INDEX IF NOT EXISTS idx_messages_thread_created
ON messages(thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_reports_thread_created
ON reports(thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_escalations_thread_resolved_created
ON escalations(thread_id, resolved, created_at);

-- Configurable emergency phrases. Loaded at query time so changes
-- take effect immediately without a redeploy.
CREATE TABLE IF NOT EXISTS emergency_phrases (
  phrase TEXT PRIMARY KEY,
  added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- LLM failure audit log — one row per failed/degraded run_json_step call.
-- Tracks fallbacks, repair attempts, and parse errors so the analytics
-- endpoint can report how often the LLM degrades gracefully vs. cleanly.
-- raw_snippet is capped to 300 chars and must not contain PHI.
CREATE TABLE IF NOT EXISTS llm_failure_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id    TEXT    NOT NULL,
  node         TEXT    NOT NULL,
  failure_type TEXT    NOT NULL,  -- "fallback_used" | "repair_used" | "parse_error" | "api_error"
  raw_snippet  TEXT,              -- first 300 chars of raw LLM output (no PHI)
  error_detail TEXT,              -- error message string if available
  created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_failure_thread_created
ON llm_failure_log(thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_llm_failure_type_created
ON llm_failure_log(failure_type, created_at);

-- Outbound webhook delivery log — one row per logical delivery event.
-- Tracks retry attempts, final status, and payload hash for idempotency.
-- url_hash is SHA-256 of the target URL so the raw URL is never stored.
-- payload_hash is SHA-256 of the request body and is the idempotency key:
--   a second dispatch with the same thread_id + event_type + payload_hash
--   is skipped if the prior delivery already succeeded.
CREATE TABLE IF NOT EXISTS webhook_deliveries (
  delivery_id      TEXT    PRIMARY KEY,
  thread_id        TEXT    NOT NULL,
  event_type       TEXT    NOT NULL,   -- slack_emergency|slack_crisis|slack_intake_complete|fhir_completion
  url_hash         TEXT    NOT NULL,   -- SHA-256(url), for correlation without storing raw URL
  payload_hash     TEXT    NOT NULL,   -- SHA-256(body), used as idempotency key
  status           TEXT    NOT NULL DEFAULT 'pending', -- pending|success|failed|exhausted
  attempts         INTEGER NOT NULL DEFAULT 0,
  last_http_status INTEGER,            -- HTTP status from most recent attempt
  last_error       TEXT,               -- error message from most recent failure
  next_retry_at    TEXT,               -- ISO datetime; NULL when terminal
  created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_thread
ON webhook_deliveries(thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status
ON webhook_deliveries(status, next_retry_at);

-- LLM token usage — one row per run_json_step or generate_text call.
-- input_tokens + output_tokens come from Gemini usage_metadata.
-- cost_usd is computed at write time using current Gemini Flash pricing.
-- Billing teams can aggregate by thread_id for per-session cost reporting.
CREATE TABLE IF NOT EXISTS llm_usage (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id     TEXT    NOT NULL,
  node          TEXT    NOT NULL,
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd      REAL    NOT NULL DEFAULT 0.0,
  created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_thread
ON llm_usage(thread_id, created_at);

-- Prompt A/B experiments — one row per active experiment.
-- variant_a_prompt / variant_b_prompt hold the prompt key names from prompts.py.
-- Sessions are routed deterministically (thread_id hash % 2) so the same
-- session always gets the same variant.  Outcomes are recorded in llm_usage.
CREATE TABLE IF NOT EXISTS prompt_experiments (
  experiment_id  TEXT PRIMARY KEY,
  name           TEXT NOT NULL,                    -- human label, e.g. "subjective_v1.3_vs_v1.4"
  prompt_key     TEXT NOT NULL,                    -- which node this targets, e.g. "subjective"
  variant_a      TEXT NOT NULL,                    -- version string, e.g. "v1.3"
  variant_b      TEXT NOT NULL,                    -- e.g. "v1.4"
  status         TEXT NOT NULL DEFAULT 'active',   -- active | paused | concluded
  sessions_a     INTEGER NOT NULL DEFAULT 0,
  sessions_b     INTEGER NOT NULL DEFAULT 0,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_experiments_status
ON prompt_experiments(status, prompt_key);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_reports_patient  ON reports(thread_id);

-- Layer 2 memory: compact structured patient summary.
-- One row per patient, overwritten on each intake via merge logic.
-- Raw transcripts and full reports stay in messages/reports tables (Layer 3).
CREATE TABLE IF NOT EXISTS patient_summary (
  patient_id            TEXT PRIMARY KEY,
  identity_json         TEXT NOT NULL DEFAULT '{}',   -- latest known identity
  allergies_json        TEXT NOT NULL DEFAULT '[]',   -- deduplicated list
  medications_json      TEXT NOT NULL DEFAULT '[]',   -- currently active meds
  conditions_json       TEXT NOT NULL DEFAULT '[]',   -- chronic/ongoing PMH
  recent_complaints_json TEXT NOT NULL DEFAULT '[]',  -- last 5 complaints w/ dates
  flags_json            TEXT NOT NULL DEFAULT '[]',   -- safety/clinical flags
  visit_count           INTEGER NOT NULL DEFAULT 0,
  first_seen_at         TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
