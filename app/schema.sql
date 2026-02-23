PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  thread_id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
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
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS escalations (
  esc_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  kind TEXT NOT NULL,         --"identity_review" | "emergency"
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
