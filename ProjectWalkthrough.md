# Clinical Intake Assistant — Project Walkthrough

This document explains every file, every design decision, and the complete runtime behavior of the system. Read this and you will understand what runs when a patient sends a message, why each piece exists, and what tradeoffs were made.

The README tells you how to run it. This document tells you how it works.

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [File map — every file explained](#2-file-map--every-file-explained)
3. [Configuration and settings](#3-configuration-and-settings)
4. [The state machine](#4-the-state-machine)
5. [Complete message lifecycle](#5-complete-message-lifecycle)
6. [Each node in detail](#6-each-node-in-detail)
7. [LLM reliability stack](#7-llm-reliability-stack)
8. [Agentic features](#8-agentic-features)
9. [Safety and guardrails](#9-safety-and-guardrails)
10. [Schemas and validated snapshots](#10-schemas-and-validated-snapshots)
11. [Database design](#11-database-design)
12. [Authentication design](#12-authentication-design)
13. [FHIR R4 integration](#13-fhir-r4-integration)
14. [Webhook delivery system](#14-webhook-delivery-system)
15. [Observability](#15-observability)
16. [The frontend](#16-the-frontend)
17. [Router package architecture](#17-router-package-architecture)
18. [Docker setup](#18-docker-setup)
19. [Design decisions index](#19-design-decisions-index)

---

## 1. What this project is

A clinical intake assistant that replaces paper triage forms. A patient opens a browser, types their symptoms and history in plain text, and the system produces:

- A structured OPQRST symptom record
- A normalized medication and allergy list
- A clinician note in narrative form
- A FHIR R4 Bundle ready for EHR import
- Real-time Slack alerts for emergencies and crisis language

It is built from three components:

**FastAPI** handles HTTP — session management, auth, rate limiting, background jobs, and API routing.

**LangGraph** manages the conversation as a state machine — a directed graph of nodes, each representing one phase of the intake. LangGraph checkpoints state to SQLite after each node so a server restart mid-intake resumes exactly where it left off.

**Gemini AI** does natural language work — extracting structured OPQRST fields from free-form symptom descriptions, parsing medication strings into structured records, and generating the final clinician note. A circuit breaker, retry logic, and hardcoded fallback dict mean a Gemini outage degrades gracefully rather than breaking the intake flow.

---

## 2. File map — every file explained

```
app/
├── main.py            Application factory — wires all routers, middleware, lifespan
├── settings.py        All configuration in one place; Pydantic validates at startup
├── graph.py           LangGraph graph — nodes, edges, guard routing, checkpoint connection
├── state.py           IntakeState TypedDict — the shape of all state flowing through nodes
├── nodes.py           Ten LangGraph node functions — the actual intake logic
├── agentic.py         Agent-level decision helpers — quality scoring, gap-fill, question tuning
├── safety.py          Safety preflight — weighted risk scoring before report generation
├── schemas.py         Pydantic models for LLM output — SubjectiveOut, MedsOut, IdentityOut, IntentOut, ReportInputState
├── prompts.py         LLM system prompts with RTCF structure, few-shot examples, version registry
├── llm.py             Gemini client — circuit breaker, retry with full jitter, 3-level degradation
├── extract.py         Deterministic extractors — crisis/emergency phrase detection, drug synonyms
├── fhir_builder.py    FHIR R4 Bundle construction and structural validation
├── webhook.py         Outbound delivery — Slack (with partial identity + message preview), FHIR webhook, retry engine, dead-letter recovery
├── memory.py          Layer-2 patient memory — cross-visit allergy union, condition union, medication replace, complaint history
├── sqlite_db.py       All database access — schema init, Alembic-tracked schema, every query function
├── schema.sql         DDL for all 13 application tables + indexes
├── logging_utils.py   Structured JSON logging with PHI masking and ContextVar trace propagation
│
├── api/               Router package — one file per concern
│   ├── __init__.py
│   ├── deps.py        Shared dependencies: rate limiter, session token auth, clinician JWT auth
│   ├── patient.py     /start, /chat, /resume, /report, /jobs — patient-facing flow
│   ├── clinician.py   /clinician/* — escalation review, FHIR fetch, A/B experiments
│   ├── admin.py       /admin/* — emergency phrase management, demo scenarios
│   └── health.py      /health, /ready, /analytics — observability endpoints
│
├── integrations/
│   ├── __init__.py
│   └── fhir_client.py FHIR server push — converts document bundle to transaction bundle, sends to HAPI/Azure/Epic
│
└── evals/             Offline evaluation harness — 157 component cases + 10 multi-turn scenarios

static/
├── index.html         Patient-facing UI — single HTML file, no framework
├── dashboard.html     Operations dashboard — KPIs, LLM health, escalation charts, auto-refresh
├── app.js             Vanilla JS — session management, chat loop, report polling, clinician panel
└── styles.css         Stylesheet

tests/
├── conftest.py        pytest fixtures — in-memory DB, mock Gemini, test client
├── test_api.py        Idempotency, LLM fallback, circuit breaker, rate limiting
├── test_auth.py       Session token and JWT auth tests
├── test_fhir.py       FHIR R4 bundle structure and resource validation
├── test_guardrails.py Crisis detection, emergency phrases, consent helpers, diagnosis filter
├── test_integration.py End-to-end conversation flow tests
└── test_memory.py     Layer-2 merge logic — allergy union, medication replace, complaint cap, crisis flag

alembic/
├── env.py             Alembic environment — reads DB path from app settings
├── README             Usage instructions for running and stamping migrations
└── versions/
    └── 001_baseline_schema.py  Baseline migration capturing all 14 tables

alembic.ini            Alembic configuration — script location, logging

Dockerfile             Single-stage Python 3.11-slim image
docker-compose.yml     App + HAPI FHIR R4 reference server
docker-compose.test.yml Isolated test environment
.env.example           Template for required environment variables
seed_patients.py       Populates mock_ehr table with demo patients
```

---

## 3. Configuration and settings

**File:** [app/settings.py](app/settings.py)

Everything configurable lives in one `Settings` class backed by Pydantic Settings. Values come from `.env`, fall back to defaults, and are validated at startup — the app fails immediately if a placeholder is still set.

```python
class Settings(BaseSettings):
    app_db_path: str          # path to main SQLite database
    checkpoint_db_path: str   # path to LangGraph checkpoint database
    gemini_api_key: str       # Gemini API key — validated at startup
    gemini_flash_model: str   # model name, e.g. "gemini-2.5-flash-lite"
    max_retries: int = 3      # LLM retry attempts
    jwt_secret: str           # used to sign clinician JWTs
    clinician_password: str   # used to obtain a clinician JWT
    slack_webhook_url: str    # Slack incoming webhook for alerts
    fhir_server_url: str      # HAPI FHIR or Azure/GCP/Epic endpoint
    intake: IntakeConfig      # nested config — thresholds, TTLs, window sizes
```

**`IntakeConfig`** is a nested Pydantic model that holds every magic number that used to be scattered across node files as literals:

```python
class IntakeConfig(BaseModel):
    ed_quality_threshold: float = 0.75      # OPQRST completeness bar for ED mode
    clinic_quality_threshold: float = 0.60  # lower bar for routine clinic visits
    max_quality_retries: int = 2            # gap-fill attempts before giving up
    messages_window_size: int = 30          # sliding window sent to LLM
    session_ttl_hours: int = 4             # sessions expire after 4 hours idle
    checkpoint_retention_days: int = 30    # prune checkpoints.db periodically
    max_session_turns: int = 30            # hard cap on user messages per session
    max_identity_attempts: int = 6         # attempts before front-desk redirect
    max_report_attempts: int = 3           # report retry cap
    use_llm_report_narrative: bool = False  # deterministic template vs LLM prose
    gemini_input_cost_per_million: float = 0.075   # Gemini Flash pricing ($/1M tokens)
    gemini_output_cost_per_million: float = 0.30
    repair_rate_alert_threshold: float = 0.05      # warn if >5% of calls need repair
    max_cost_usd_per_session: float = 0.01         # hard cost cap per session
    dead_letter_retry_after_hours: int = 24
    dead_letter_max_lifetime_attempts: int = 6
```

**Design decision: why one settings object, not scattered constants.**
When ED and clinic mode used different quality thresholds but those numbers were hardcoded in nodes.py, changing one required grepping. When the threshold was wrong (0.65 clinic was too low for our eval results), there was no clear place to change it. `IntakeConfig` centralizes all tuning knobs. A future A/B test that wants to lower the clinic threshold for a specific trial changes one env var, not source code.

**Why LLM pricing constants live in `IntakeConfig` and not in `llm.py` or `patient.py`:**
Both `llm.py` (cost calculation per API call) and `patient.py` (per-session cost cap check) need the pricing values. If those values were hardcoded in each file, a Gemini pricing tier change would require updating two places — and if only one was updated, billing calculations would be inconsistent. Moving them to `IntakeConfig` creates exactly one place to update. This is not a theoretical concern: Gemini Flash pricing changed between preview and GA.

**Why `max_identity_attempts` in `IntakeConfig` and not hardcoded in `nodes.py`:**
The identity node had `if attempts >= 6` hardcoded. That number (6) is an operational policy — how many times should we try before giving up? A clinic treating elderly patients who struggle with DOB formats might need more attempts. A hardcoded `6` is invisible to operations; a config field is visible, documented, and changeable without a code deployment.

**Design decision: fail at startup, not at first API call.**
Three `@field_validator` decorators check that `gemini_api_key`, `jwt_secret`, and `clinician_password` are not still set to placeholder values. If they are, the process exits immediately with a clear error message. Without this, the app starts, the first patient sends a message, Gemini rejects the key, and the patient sees a 500 error. Failing early is more honest.

**Singleton pattern:**
```python
_settings_instance: Settings | None = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
```

`Settings()` reads the `.env` file at construction time. If called on every request, it re-reads the file on every request. The singleton ensures one construction per process lifetime. Callers always write `get_settings()` (with parentheses) — never alias to a module-level variable without calling, which would assign the function object rather than the settings instance.

---

## 4. The state machine

**File:** [app/graph.py](app/graph.py)

The conversation is a LangGraph `StateGraph` with ten nodes. The graph is compiled once at startup with a SQLite checkpointer and reused across all requests.

```
Every patient message
         ↓
guard_node  ← transparent safety pre-processor on EVERY turn
    │ crisis detected → END directly (CRISIS_RESOURCE already set)
    │ no crisis       → route_after_guard() → business node below
    ↓
consent_node
    ↓
identity_node ←──────────────────────────────────────────┐
    ↓                                                     │
identity_review_node ──(mismatch)──→ identity_node        │
    ↓ (confirmed)                                         │
subjective_node ←──────────────────────────────────────┐  │
    ↓ (complete)                                        │  │
validate_node (silent) ─────(fail)─→ subjective_node   │  │
    ↓ (pass)                                            │  │
clinical_history_node ←────────────────────────────┐   │  │
    ↓ (all steps done)                              │   │  │
validate_node (silent) ─────(fail)─→ clinical_history  │  │
    ↓ (pass)                                            │  │
confirm_node ─────(corrections)────→ subjective/clinical/identity
    ↓ (confirmed)
report_node ──→ END
    (also)
handoff_node ─→ END  (emergency path)
```

**guard_node — the transparent safety pre-processor:**

`guard_node` sits at the START of the graph and runs before every other node on every patient message. It handles crisis detection centrally so no business node can accidentally skip the safety check. If crisis language is detected, `guard_node` sets the patient-facing message, writes the DB escalation, fires the Slack webhook, and then `route_after_guard()` routes directly to END — not through `handoff_node`, which would add a second message the patient would never see.

```python
g.add_conditional_edges(START, lambda _: "guard_node")
g.add_conditional_edges("guard_node", route_after_guard)
```

`guard_node` is intentionally NOT in `interrupt_after`. Adding it there would cause the graph to pause after the safety check, consuming the user message, and then starve the business node — the business node would run on the next patient message instead of the current one, causing the conversation to skip one turn.

**How routing works:** `route(state)` in graph.py is the only function that decides what business node runs next. It reads `state["current_phase"]` and does a dictionary lookup — no model call, no probabilistic decision. The LLM never decides flow control.

```python
def route(state: IntakeState):
    return {
        "consent":          "consent_node",
        "identity":         "identity_node",
        "identity_review":  "identity_review_node",
        "subjective":       "subjective_node",
        "validate":         "validate_node",
        "clinical_history": "clinical_history_node",
        "report":           "report_node",
        "handoff":          "handoff_node",
        "confirm":          "confirm_node",
        "done":             END,
    }.get(phase, "identity_node")
```

**Interrupt behavior:** six of the ten nodes have `interrupt_after` set — the graph pauses after them and waits for the next patient message. `guard_node` and `validate_node` are non-interactive. `report_node` and `handoff_node` run straight to `END`.

**Why `current_phase` as a string in state, not a class attribute:**
LangGraph state is a `TypedDict` that is serialized to SQLite. Enum types require custom serialization. A plain string is serialized trivially and is readable in the checkpoint blob without decoding. The valid values are documented in `route()`.

**The checkpoint database:**
`checkpoints.db` is a second SQLite file separate from `app.db`. LangGraph manages this file entirely — schema, writes, and reads. Separating it means:
- `app.db` can be backed up or queried without touching the checkpoint format
- Checkpoint schema changes from LangGraph version upgrades don't affect migrations on `app.db`
- `prune_old_checkpoints()` can delete old checkpoint rows without risk to application tables

---

## 5. Complete message lifecycle

Tracing a single `POST /chat` from browser to response.

### 5.1 Browser

`app.js` increments `clientMsgId` (a counter), builds `FormData` with `thread_id`, `message`, `client_msg_id`, and `Authorization: Bearer <sessionToken>`, then POSTs to `/chat`.

The session token was issued at `/start` and stored in `sessionToken` in memory. It is sent on every request. There is no cookie.

### 5.2 Middleware (CorrelationMiddleware)

Before any route handler runs, `CorrelationMiddleware` in `main.py` generates or reads a `X-Request-Id`, stores it in a `ContextVar`, logs the request start, and attaches the ID to the response header. Every log line emitted during this request will carry the same `request_id` automatically — no manual threading needed.

### 5.3 Rate limiter

`@limiter.limit("60/minute")` on `/chat`. 429 returned if exceeded. Uses the client IP as the key. This is enforced per-IP, not per session — a single patient cannot flood the LLM regardless of how many tabs they open.

### 5.4 Session token verification

`require_session_token(thread_id, authorization)` in `deps.py`:
1. Strips `"Bearer "` prefix from the Authorization header.
2. If the token is empty → 401 "Missing session token."
3. Queries the `sessions` table for the stored SHA-256 hash of the token issued at `/start`.
4. Compares using `hmac.compare_digest` (constant-time, prevents timing oracle attacks).
5. If no match → 401 "Invalid or expired session token."

Without this check, any HTTP client that knows a patient's UUID-shaped `thread_id` can read their session data. The token is never stored in plaintext — only its SHA-256 hash is in the database.

### 5.5 Input validation

- Message length: 1–1200 characters. Empty messages return 400. Messages > 1200 chars return 400.
- `client_msg_id` max 128 chars.
- `check_prompt_injection(message)` scans for jailbreak patterns: "ignore previous instructions", "you are now a", "forget your training", "act as if you have no restrictions". Match returns a fixed neutral reply; the graph never runs.

**Why 1200 chars:** Most clinical descriptions are under 200 words. 1200 chars accommodates a verbose patient with room to spare while preventing prompt-stuffing attacks that try to push the LLM context past the system prompt.

### 5.6 Idempotency check

`db.get_idempotent_response(thread_id, client_msg_id)` queries the `idempotency` table.

- **No match:** new request, continue.
- **Match + same SHA-256 of message body:** duplicate submission — return the cached response. No graph invocation.
- **Match + different SHA-256:** client reused an ID with a different message — 409 Conflict. This prevents a class of bug where a retry loop sends a corrected message with the same ID and gets the old response cached.

**Why client-side IDs rather than server-side dedup:**
Server-side dedup (e.g., last-write-wins on thread_id) doesn't distinguish between "patient double-clicked" (duplicate, safe to return cached) and "patient corrected themselves and is retrying" (different message, should process). The client-generated `clientMsgId` tied to a hash of the message body handles both cases correctly.

### 5.7 Session TTL enforcement

`db.expire_stale_sessions(ttl_hours=4)` runs on every `/chat` call. It executes one SQL UPDATE that marks sessions where `updated_at` is more than 4 hours ago as `expired`. If the current session just became expired, the next `SELECT` returns `status='expired'` and the handler returns 410 Gone.

**Why per-request TTL enforcement, not a background job:**
A background job requires scheduling (cron, asyncio task, threading.Timer), which adds failure modes — if the job crashes silently, sessions never expire and linger as `active`. Per-request enforcement is guaranteed to run for every incoming request. The query is a single UPDATE affecting at most a few rows, so the overhead is negligible.

### 5.8 Graph invocation

```python
output = graph.invoke(
    {"messages": [{"role": "user", "text": message}]},
    {"configurable": {"thread_id": thread_id}}
)
```

LangGraph loads the checkpoint for `thread_id` from `checkpoints.db`, appends the new message to `state["messages"]` via the `operator.add` reducer (which accumulates rather than replaces), calls `route(state)` to decide which node to run, and runs it.

The `operator.add` reducer is the reason messages accumulate. If the node instead returned `{"messages": [...]}` without the reducer, each invocation would overwrite the history. With `operator.add`, each invocation appends, giving a full conversation log in state.

### 5.9 Inside a node (subjective_node as the example)

1. `last_user(state)` scans `_window_messages(state)` in reverse for the last user turn. The window function returns only the last `messages_window_size` (default 30) messages from state. On a 50-turn conversation, only the 30 most recent messages are ever passed to the LLM — token usage is O(1), not O(n).

2. Crisis check runs first (see section 9.1).

3. Emergency red-flag detection runs second (see section 9.2).

4. `run_json_step(SubjectiveOut, ...)` calls Gemini. This is the combined extraction + classification call — one Gemini request returns OPQRST fields, `is_complete`, `reply`, `extraction_confidence`, `intake_classification`, and `classification_confidence`. Previously two sequential calls. See section 8.1.

5. Quality gate: `score_extraction_quality(cc, opqrst)` scores completeness 0–1. Below threshold → `build_gap_fill_question()` returns a targeted question (no LLM, deterministic). See section 8.3.

6. `_track_llm_failure(thread_id, "subjective_node", meta)` records token usage unconditionally and logs failures if any. See section 7.4.

7. `_safe_reply(text)` runs the LLM's `reply` field through the diagnosis-language guardrail. See section 9.4.

8. The node returns a dict of state fields to update. LangGraph merges this with existing state and checkpoints.

### 5.10 After graph invocation

`_compact_snapshot(output)` persists all state fields except `messages` (which lives in the `messages` table) to the `session_state` table. This is a lean read cache — the `/resume` endpoint and clinician case view read from here rather than deserializing the full LangGraph checkpoint blob.

```python
_SNAPSHOT_EXCLUDE: frozenset[str] = frozenset({"messages"})

def _compact_snapshot(output: dict) -> dict:
    return {k: v for k, v in output.items() if k not in _SNAPSHOT_EXCLUDE}
```

**Design decision: exclusion list rather than whitelist.**
The previous implementation was a 22-field whitelist — an explicit list of every `IntakeState` field to persist. When `session_cost_usd` was added to track per-session LLM spend, it was added to `IntakeState` but not to the whitelist. The cost accumulator silently reset to zero on every `/resume` call, defeating the per-session cost cap. With an exclusion list, any new field added to `IntakeState` is automatically persisted — a developer cannot forget to update the snapshot function. The only field excluded is `messages`, which has its own storage. This also eliminates a maintenance burden: adding a new field no longer requires updating a separate list.

**Why two state stores (checkpoints.db + session_state table):**
The LangGraph checkpoint contains the full state including the complete message history. Deserializing it for every API read is wasteful. `session_state` stores a compact dict of just the fields that API responses care about. Writes to `session_state` happen at every turn; reads from `checkpoints.db` happen only when the graph resumes execution.

### 5.11 Response

```json
{"reply": "...", "phase": "subjective", "status": "active"}
```

If `phase == "report"`, a background job is queued instead:
```json
{"reply": "...", "phase": "report_generating", "job_id": "abc-123", "status": "active"}
```

The frontend then polls `GET /jobs/{job_id}` every 2 seconds. When `status == "done"`, it fetches `GET /report/{thread_id}`.

**Why background jobs for report generation:**
Report generation runs Gemini once (clinician note), builds a FHIR bundle, validates it, saves it, and fires webhooks. This can take 3–8 seconds. If it ran synchronously in `/chat`, the patient's browser would hang waiting for a response. The background task model lets the API return immediately and the frontend poll for completion.

---

## 6. Each node in detail

### 6.1 consent_node

**Purpose:** Presents the AI disclosure and consent message before collecting any health information. HIPAA and general healthcare ethics require informed consent before data collection.

**Logic:** `_classify_intent(user, state)` using the two-tier intent classifier:

- `confirm` → `consent_given=True`, phase → `identity`
- `decline` → `consent_given=False`, phase → `done`, session status set to `done`
- `unclear` → re-prompt for yes or no

**Two-tier intent classification:** `_classify_intent()` is called on every binary decision in the system — consent, identity review, confirm — replacing the old hardcoded word lists (`is_yes`, `is_no`). Tier 1 is exact match against `_HARD_YES` / `_HARD_NO` sets. For short ambiguous messages like "I think so" or "I suppose not", Tier 2 sends them to Gemini with a tiny schema (`IntentOut`) and `max_tokens=40`, `temperature=0.0`. This handles natural patient language ("I guess yeah") without maintaining ever-growing keyword lists.

**Why LLM for intent but not for consent itself:** The consent decision is still deterministic — "confirm" means yes, "decline" means no. The LLM is only used to interpret ambiguous phrasing into one of those two categories. The routing logic never sees the raw patient text — it sees the classified intent enum.

### 6.2 identity_node

**Purpose:** Collects name, DOB, phone, and address, then checks the mock EHR for a matching record.

**Logic:** LLM-based extraction via `run_json_step(IdentityOut, ...)` using the `identity_extract_system()` prompt. Normalization happens at the Pydantic schema boundary — not inside the extraction logic:

- `name` validator → Title Case ("john smith" → "John Smith"), rejects "unknown"/"n/a"
- `dob` validator → ISO 8601 ("March 3rd 1992" → "1992-03-03"), handles all common date formats including ordinal suffixes (1st, 2nd, 3rd)
- `phone` validator → 10-digit normalized, strips +1 country code
- `address` validator → stripped, rejects "not provided"/"none"

Fields accumulate across messages — identity is often given across 2–3 turns. The node merges extracted fields into the existing `state["identity"]` dict, only replacing empty slots. When all four fields are populated, `db.get_stored_identity_by_name(name)` checks the mock EHR.

- Match found → phase transitions to `identity_review` with a warm returning-patient message: "Welcome back, Jane! Your phone on file is 4125550199..."
- No match → phase goes to `identity_review` asking the patient to confirm what they entered
- 3 failed attempts → max attempts message, escalation, manual front-desk instruction

**Why LLM instead of regex for identity:**
Patients give identity in every format imaginable: "I was born on the 3rd of March, 1992", "My name's dr. sarah mcallister-jones", "555 867 5309 that's my cell". Regex handles the structured formats but fails on free-form phrasing. The `IdentityOut` schema's Pydantic validators normalize everything at the boundary — the LLM extracts, Pydantic normalises, the node receives clean typed fields regardless of how the patient phrased their input.

### 6.3 identity_review_node

**Purpose:** Confirms or updates patient identity. For returning patients, presents a warm acknowledgment with stored details and asks if anything has changed.

**Logic:** `_classify_intent(user, state)` using the two-tier classifier:

**Returning patient (stored record found):**
- `confirm` → uses stored identity, `identity_status="verified"`, continues to subjective
- `decline` → creates `identity_review` escalation (nurse to reconcile), continues with patient-provided identity
- `unclear` → asks "Would you like to keep the information on file, or use what you provided?"

**New patient (confirming fresh entry):**
- `confirm` → `identity_status="verified"`, continues to subjective
- `decline` or `correction` → routes back to identity_node to re-enter
- `unclear` → asks explicitly for "yes to confirm, no to re-enter"

The escalation payload from `build_reason_trail()` documents exactly which fields differ between the stored and patient-provided identity, so the reviewing nurse sees which specific data points need reconciliation.

### 6.4 subjective_node

**Purpose:** Extracts OPQRST symptom fields from free-form patient description, classifies the visit type, and scores data quality.

This is the most complex node. See section 5.9 for the execution order. The key design choices:

**Crisis is no longer in subjective_node:** Crisis detection was moved entirely to `guard_node`, which runs before every node including subjective. This means subjective_node focuses purely on symptom extraction and does not need to import crisis detection logic. The guard's centralized approach guarantees the safety check always runs regardless of which business node follows.

**Emergency before LLM:** Emergency red-flag detection (`detect_emergency_red_flags`) still runs in `subjective_node` before any LLM call because the emergency response (route to handoff) is phase-specific to the symptom collection step. If a patient mentions chest pain during symptom collection, the system escalates immediately rather than extracting OPQRST fields from an emergency description.

**Combined extraction + classification in one LLM call:** `SubjectiveOut` includes both OPQRST fields and `intake_classification`. The subjective prompt instructs Gemini to return both in a single JSON object. This halves latency on the most common turn (first chief complaint message) and eliminates the failure mode where extraction succeeds but the second classification call fails.

**Quality gate:** After extraction, `score_extraction_quality()` scores completeness on a weighted scale (chief complaint 0.25, onset 0.20, severity 0.20, etc. — see section 8.3). Below threshold → `build_gap_fill_question()` generates a deterministic targeted question for the highest-priority missing field. The gap-fill path uses no LLM — keeping it fast is critical because it runs on a retry path that could already be recovering from an LLM failure.

### 6.5 validate_node

**Purpose:** Non-interactive gate that enforces completeness before a phase transition is allowed.

**Why a separate node instead of validation inside subjective_node:**
Putting validation inside `subjective_node` would mean the node decides its own phase transition. This conflates "process the message" with "decide if we're done." The validate node is a clean separation: `subjective_node` extracts, `validate_node` enforces. It also means validate logic can be unit-tested in isolation.

**Logic:**
- Check `chief_complaint` is non-empty and non-placeholder
- Check `extraction_quality_score >= threshold`
- Check `extraction_confidence != "low"`
- Check `allergies` were asked (when transitioning to confirm)

On pass: sets `current_phase` to `validation_target_phase` (set by the previous node — either `clinical_history` or `confirm`).
On fail: injects `build_validation_gap_message()` as an assistant message, routes back to the source phase.

### 6.6 clinical_history_node

**Purpose:** Collects allergies, medications, past medical history, and recent test results in four sequential sub-steps.

**Sub-step routing:** `clinical_step` in state tracks which sub-step is active: `"allergies"` → `"meds"` → `"pmh"` → `"results"`. This is a sub-state machine inside the node. The node checks `clinical_step` at the top and routes to the appropriate handler.

**Lookahead (`_prescan_volunteered_clinical`):** Before asking about a step, the node scans the patient's earlier messages for volunteered clinical information. If a patient said "I take lisinopril and have no allergies" during the symptom phase, the allergy and medication steps can be pre-populated and skipped. This uses `None` vs `[]` as a sentinel — a field that is `None` was never asked, a field that is `[]` was asked and the patient said none. The lookahead only pre-populates `None` fields, never overwrites existing data.

**Warm questions:** Clinical history questions are phrased warmly, not as data entry prompts. "Do you have any known allergies to medications, foods, or anything else? If none, just say 'none'." This makes the interaction feel like a conversation rather than a form.

**Allergies and PMH/results:** Deterministic extraction via `extract_allergies_simple()` and `extract_list_simple()`. Regex for comma/semicolon separated lists and bullet points. Drug synonym normalization (`normalize_drug_name()`) maps brand names to generics — "Tylenol" becomes "acetaminophen (Tylenol)".

**Medications:** `run_json_step(MedsOut, ...)` uses Gemini because medication parsing is hard with regex — "Lisinopril 10mg once daily" needs to be split into name, dose, frequency, and last-taken fields that don't follow fixed patterns. The `MedicationItem.strip_name` Pydantic validator normalizes the name through the same synonym table.

**Dosage follow-up:** After extracting medications, if any have a name but no dose and no frequency, the node asks for dosage. If the patient didn't provide it on the second attempt either, the node accepts what was given and moves on rather than looping. This is tracked by detecting whether any existing medications already have empty dose/freq fields — the "already asked once" check. A patient who genuinely doesn't know their dosage shouldn't be trapped in a loop.

**Adapted questions:** `adapt_clinical_question(step, classification)` rewrites the question for pediatric or mental health contexts. A pediatric patient is asked about vitamins and OTC children's medications. A mental health patient's PMH question includes prior psychiatric diagnoses and treatments. The adaptation is table-driven — no LLM call.

**Go-back / correction:** `_try_correction(user, state)` runs at the top of clinical_history_node. If the patient says "wait, I need to change my allergies" or "go back", `_try_correction` detects the correction intent (via regex matching on the message text) and routes back to the appropriate section without needing an LLM call.

### 6.7 confirm_node

**Purpose:** Presents the full intake summary as a natural-language paragraph and waits for patient confirmation or correction.

**Summary format:** `_confirm_summary(state)` generates a paragraph rather than a table. Instead of "Name: Jane Doe | DOB: 1990-03-15 | Chief Complaint: headache", the summary reads like: "To confirm — your name is Jane Doe, born March 15, 1990. You came in today for a headache that started two days ago, rated 7 out of 10..." This reads like a nurse reading back notes, not a form printout.

**Logic:** `_try_correction(user, state)` runs first, using regex to detect if the patient wants to change a specific section. If a correction is detected, it routes back to the appropriate phase immediately without needing an LLM call. If no correction is detected, `_classify_intent` is used for the confirm/decline decision:
- `confirm` → phase = `report`, background job queued
- `decline` or `correction` (without a specific section) → asks "What would you like to change?"
- Section-specific correction detected → routes back to `identity`, `subjective`, or `clinical_history`

**`_try_correction` — shared across all interactive nodes:** This helper runs at the top of `subjective_node`, `clinical_history_node`, and `confirm_node`. It prevents patients from getting stuck when they realize mid-intake that something earlier was wrong. The correction regex matches phrases like "go back", "change my", "that was wrong", "actually my [field]".

### 6.8 report_node

**Purpose:** Generates the clinician note and FHIR bundle. The full report is shown directly in the patient chat, not behind a "click here" link. Runs to END without interrupt.

**Execution sequence:**
1. `SafetyChecker.compute(state)` — weighted preflight (see section 9.3). Hard blocks stop generation entirely and create an escalation for clinician review.
2. `_build_validated_report_state(state)` — constructs `ReportInputState` from raw state via Pydantic validation. All field bounds and list caps applied here.
3. `get_gemini().generate_text(report_system(...))` — generates the clinician note from the validated snapshot. Temperature 0.2 for consistency.
4. `_validate_report_content(report_text)` — checks minimum length, required sections, no diagnosis language.
5. `fhir_builder.validate_fhir_input(validated)` — pre-build input warnings.
6. `fhir_builder.build_bundle(validated.model_dump())` — builds FHIR R4 Bundle.
7. `fhir_builder.validate_fhir_bundle(bundle)` — post-build structural validation.
8. `db.save_report(...)` — saves both note and bundle JSON to DB.
9. `db.set_session_status(thread_id, "done")`.
10. `webhook.dispatch_intake_complete(...)` — fires in daemon thread.
11. `log_event("phi_audit_session_complete")` — audit trail.

**Why both the note and the FHIR bundle are built from `ReportInputState` (never raw state):**
Raw `IntakeState` contains unvalidated strings that can be arbitrarily long (whatever the LLM produced). `ReportInputState` enforces `max_length` on every string field and caps list lengths. This means a Gemini output that produced a 5000-char chief complaint can't pollute the clinician note or FHIR bundle — it gets truncated to 300 chars at the model boundary.

### 6.9 handoff_node

**Purpose:** Terminal node for emergency cases. Returns a safety message and exits.

Emergency escalations (from `detect_emergency_red_flags`) set `phase = "handoff"`. The handoff node presents a message telling the patient to call 911 or go to the nearest emergency department, creates an escalation record, and the graph exits to END. There is no report generation for handoff cases — the clinician sees the escalation in their pending queue.

---

## 7. LLM reliability stack

**File:** [app/llm.py](app/llm.py)

Three independent layers of reliability, each handling a different failure mode.

### 7.1 Circuit breaker

```
CLOSED → (5 consecutive failures) → OPEN → (60s timeout) → HALF_OPEN → (one probe) → CLOSED
                                                                              ↓ (probe fails)
                                                                            OPEN (reset timer)
```

The breaker is a module-level singleton (`_breaker = CircuitBreaker()`). When 5 consecutive API calls fail, the breaker opens. All subsequent calls return `LLMResult(ok=False, error="circuit_breaker_open")` immediately — no HTTP attempt, no wait. After 60 seconds, the breaker enters `HALF_OPEN` and allows one probe request. If the probe succeeds, the breaker closes; if it fails, it reopens and resets the 60-second timer.

**Design decision: why a circuit breaker and not just retry:**
Retries help with transient errors (network blip, brief rate limit). A circuit breaker helps with sustained outages. If Gemini is down for 10 minutes and every request retries 3 times with backoff, the intake system queues up hundreds of requests that all wait 2s + 8s + 30s before failing. During a sustained outage, the circuit breaker makes each request fail immediately (in milliseconds), keeping the patient UI responsive (they see "service degraded" immediately) rather than hanging.

**What happens when the breaker is open:**
`run_json_step` proceeds to the fallback dict (Level 3 — see section 7.3). The intake continues with empty OPQRST fields. The patient is asked questions manually by the gap-fill engine. The intake is degraded but functional.

### 7.2 Retry with full jitter

```python
ceiling = min(cap, base * (2 ** attempt))
sleep_s = random.uniform(0, ceiling)   # full jitter
```

Full jitter (uniform random between 0 and the computed ceiling) is used instead of multiplicative jitter (multiply the deterministic delay by a random factor). The distinction matters when multiple processes hit the API simultaneously after an outage: multiplicative jitter concentrates retries near the ceiling, while full jitter spreads them uniformly from 0 to the ceiling. AWS's "Exponential Backoff and Jitter" research paper demonstrates full jitter wins for thundering herd prevention.

Permanent errors (auth failure, invalid argument, not found) are not retried — `is_transient_error()` classifies them and returns immediately.

**Timeout enforcement:**
The google-genai SDK does not reliably respect HTTP timeouts. `generate_text` wraps each API call in a `concurrent.futures.ThreadPoolExecutor` with a single worker and calls `future.result(timeout=llm_timeout_seconds)`. This is a hard wall-clock timeout that works regardless of the SDK's own timeout behavior.

### 7.3 Three-level JSON degradation (run_json_step)

Every structured LLM call goes through `run_json_step` which has three levels:

**Level 1 — Primary:**
Call Gemini → `extract_json(response)` strips markdown fences and finds the first valid JSON object → `schema.model_validate_json(cleaned)` applies Pydantic validation. On success, return the validated model.

**Level 2 — Repair:**
If the LLM responded (no network error) but the JSON was invalid or failed schema validation, send a repair prompt. The repair prompt names the exact validation error, lists required keys, forbids extra text, and includes the first 800 chars of the bad output. This targeted prompt works better than a generic "try again" because it tells the model exactly what it did wrong. If the repair also fails, continue to Level 3.

**Level 3 — Hardcoded fallback:**
If both calls fail (or if the API was unreachable), `schema.model_validate(fallback_dict)` constructs the model from a pre-authored fallback. The fallback dict is in the calling node's source code, not generated at runtime. It produces empty strings for all text fields and `is_complete=False` for the completion flag. The intake continues — the patient gets a gap-fill question because the quality score is 0 — but the system never crashes.

**Design decision: fallback dict in the caller, not in run_json_step:**
Different nodes need different fallbacks. `SubjectiveOut` fallback has `is_complete=False` so the subjective phase continues collecting. A hypothetical triage fallback might default to `risk_level="high"` so uncertain cases get escalated rather than dismissed. Embedding fallback logic in the caller makes this node-specific policy visible at the call site.

### 7.4 Token accounting and failure tracking

Every `run_json_step` call aggregates token counts across the primary call and any repair call:

```python
total_input  = res.input_tokens + (res2.input_tokens if res2 else 0)
total_output = res.output_tokens + (res2.output_tokens if res2 else 0)
```

The `meta` dict returned alongside the parsed model contains `input_tokens`, `output_tokens`, `fallback_used`, `repair_used`, `parse_error`, `latency_ms`, and `cost_usd`. Callers pass this to `_track_llm_failure()` which writes to `llm_usage` and (on failures) `llm_failure_log`.

Token usage is written on every call regardless of success or failure. This gives accurate per-session and per-node cost data even when the LLM degraded.

---

## 8. Agentic features

**File:** [app/agentic.py](app/agentic.py)

Four decision-making capabilities layered on top of the deterministic state machine.

### 8.1 Combined extraction + classification (one LLM call)

`SubjectiveOut` includes `intake_classification` and `classification_confidence` alongside the OPQRST fields. The subjective prompt instructs Gemini to classify the visit as one of: `emergency_visit`, `routine_checkup`, `specialist_referral`, `mental_health`, `pediatric`.

The node reads `out.intake_classification` from the single extraction result — no second round-trip. On the most common turn (first chief complaint), this halves LLM latency.

**Why classification matters:** `intake_classification` is passed to `adapt_clinical_question()` which rewrites history questions for context. A pediatric case gets medication questions adapted for children and vitamins. A mental health case gets PMH questions that include psychiatric history. The classification also goes into the FHIR bundle, clinician note, and SafetyChecker scoring.

### 8.2 Dynamic question adaptation

`adapt_clinical_question(step, classification)` is a lookup table — no LLM. When `step="meds"` and `classification="pediatric"`, it returns a reworded question mentioning children's vitamins and OTC medications. For all other combinations it returns `""` which tells the node to use the default question.

The adaptation is intentionally table-driven rather than LLM-driven because:
1. The question variants are small in number (a few specific overrides)
2. They need to be audited and approved before deployment — an LLM rewriting questions ad-hoc creates unauditable clinical wording
3. It's on the non-critical path (question selection, not data extraction) and must remain fast even during LLM degradation

### 8.3 Extraction quality scoring

`score_extraction_quality(cc, opqrst)` returns a float 0–1:

| Field | Weight | Clinical rationale |
|---|---|---|
| chief_complaint | 0.25 | Without this, nothing downstream makes sense |
| onset | 0.20 | Acute vs chronic distinction drives triage |
| severity | 0.20 | Drives triage risk level directly |
| quality | 0.10 | Differentiates cardiac from musculoskeletal |
| timing | 0.10 | Constant vs intermittent matters clinically |
| provocation | 0.075 | Positional vs exertional modifiers |
| radiation | 0.075 | Arm/jaw radiation is a cardiac red flag |

ED threshold: 0.75 (chief complaint + onset + severity must all be present, plus at least one of quality/timing).
Clinic threshold: 0.60 (chief complaint + onset + severity sufficient).

Below threshold + retries remaining → `build_gap_fill_question()` picks the highest-priority missing field in deterministic priority order (severity → onset → quality → timing → radiation → provocation) and returns a targeted question. No LLM in the gap-fill path — it must work even if Gemini is down.

### 8.4 Layer-2 patient memory

**File:** [app/memory.py](app/memory.py)

After each completed intake, `merge_summary(prior, visit)` folds the session's validated state into a persistent cross-visit summary stored in the `patient_summary` table, keyed by patient name + DOB.

Each field has an explicit merge rule chosen based on its clinical semantics:

| Field | Merge rule | Why |
|---|---|---|
| `identity` | Replace (most recent wins) | Contact info changes — phone, address |
| `allergies` | Union, dedup, cap 20 | A known allergen must never be forgotten |
| `medications` | Replace with current visit | Patients stop/start meds — cumulative union would pollute with stopped drugs |
| `conditions` (PMH) | Union | Chronic conditions accumulate over time |
| `recent_complaints` | Append, keep last 5 | Window into visit history without unbounded growth |
| `flags` | Union, no cap | Crisis history is important; it must never be dropped |

On the next visit, `format_for_prompt(summary)` renders this into a compact `RETURNING_PATIENT` block injected into LLM prompts. The identity node sees it and generates a warm acknowledgment ("Welcome back, Jane!") instead of treating a returning patient as a stranger. The clinical history node sees known allergies and can skip or pre-populate.

**Why separate from `sqlite_db.py`:** Merge logic is domain logic — it answers "how should two versions of clinical data be combined?" That is a different concern from "how do I persist a dict to SQLite?" Mixing them would make the merge rules harder to test in isolation and harder to reason about. `memory.py` has no database imports; `sqlite_db.py` has no merge logic.

### 8.5 Validate node gate

A non-interactive node that runs between phase transitions without pausing for patient input. It enforces completeness rules that would be unsafe to skip:

- A clinical history step cannot be skipped if `allergies` were never asked (missing allergies in a clinical note is a patient safety risk)
- Confirm cannot be reached if `chief_complaint` is empty
- Report cannot be reached if `clinical_complete` is False

When the gate blocks, `build_validation_gap_message(errors, cc, mode)` produces a patient-facing explanation of what is still needed. This message is injected into state as an assistant message — the patient sees it on their next turn without knowing a validation step ran.

---

## 9. Safety and guardrails

### 9.1 Crisis detection — guard_node

**File:** [app/nodes.py](app/nodes.py) → `guard_node`, [app/extract.py](app/extract.py)

Crisis detection was moved from per-node calls into a single centralized `guard_node` that runs before every interactive node on every turn. This eliminates a structural flaw: when crisis detection was called inside each business node, adding a new node required remembering to add the crisis check. With `guard_node` at START, it is architecturally impossible to skip.

**Tier 1 — Keyword / regex (zero latency):**
`detect_crisis(user)` scans against `_CRISIS_PHRASES` (exact phrases) and `_CRISIS_REGEX_PATTERNS` (morphological variants):
- `\bkill\w*\s+myself\b` → "killing myself", "kills myself", "killed myself"
- `\bend\w*\s+my\s+life\b` → "ending my life", "ended my life"
- `\bhurt\w*\s+myself\b`

**Tier 2 — LLM classifier for soft distress (when Tier 1 doesn't fire):**
`has_soft_distress(user)` detects messages that might be distress signals but don't match exact phrases ("I can't take it anymore", "there's no point"). If that fires, `llm_crisis_score(user)` calls Gemini with `CrisisScore` schema to determine whether the language represents genuine suicidal ideation or figurative speech ("this headache is killing me"). Only `is_crisis_risk=True` with `confidence in ("high", "medium")` triggers a full crisis response.

**On crisis match (either tier):**
1. Sets patient-facing message to `CRISIS_RESOURCE` — the 988 Suicide & Crisis Lifeline message
2. `db.create_escalation(kind="crisis")` with `build_reason_trail()` payload
3. `webhook.dispatch_crisis_alert()` fires in a daemon thread — includes `partial_identity` (whatever identity was collected so far, or "Unknown — crisis before identity was collected") and `message_preview` (first 120 chars of what the patient typed)
4. Sets `current_phase="handoff"`, `crisis_detected=True`
5. `route_after_guard()` routes directly to END — not through `handoff_node` which would add a second message

**Identity is never required for a crisis response.** The Slack alert shows whatever identity the system has: if the patient hadn't reached identity yet, the alert says "Unknown — crisis occurred before identity was collected" with the session UUID for lookup.

**Why `route_after_guard` → END and not handoff_node:**
`handoff_node` appends its own patient-facing message ("call 911 or go to the nearest ER"). `guard_node` already set the patient-facing message (`CRISIS_RESOURCE`). Routing through `handoff_node` would add a second message the patient would never see, and the 911 message is inappropriate for a mental health crisis. Routing directly to END serves the correct message once.

### 9.2 Emergency red-flag detection

`detect_emergency_red_flags(chief_complaint, opqrst, user_message)` is negation and history aware:

```python
NEGATIONS = {"no", "not", "without", "denies", "denied", "free", "history of", "never"}
HISTORICAL = {"history of", "past history", "years ago", "prior", "previously"}
```

For each phrase match, the preceding 5 words are checked against negations and historical context:
- "no chest pain" → negation detected → not flagged
- "history of chest pain" → historical context → not flagged
- "severe chest pain" → flagged → escalation + Slack alert + phase → `handoff`

Phrase list is loaded from the `emergency_phrases` DB table on every call — no redeploy needed to add or remove phrases. Clinicians can add phrases via `POST /admin/emergency-phrases`.

### 9.3 Safety preflight (SafetyChecker)

**File:** [app/safety.py](app/safety.py)

Runs at the start of `report_node` before any artifact is generated.

**Hard blocks** (individually sufficient to prevent report generation):
- `chief_complaint_missing` (+35): Cannot write a clinician note without knowing why the patient is there
- `patient_name_missing` (+30): A report that cannot be attributed to a patient is a liability
- `clinical_history_incomplete` (+25): Omitting allergies from a clinical note is a known source of medication errors

**Review signals** (raise the score toward the 50-point threshold):
- `emergency_flag_active` (+50): Any session with an emergency red flag crosses the threshold alone
- `crisis_detected` (+40): Always overrides the score check — crisis always requires human review
- `identity_unverified` (+20): Patient identity not confirmed against EHR
- `identity_mismatch_flagged` (+15): Discrepancy between EHR and patient-provided data
- `extraction_quality_low` (+20): OPQRST below threshold — note may be incomplete
- `extraction_retried` (+10): Quality gate had to retry — indicates initial data was thin
- `ed_mode_baseline` (+10): All ED sessions carry baseline review requirement

Score ≥ 50 → `human_review_required = True`. Hard blocks → `ok = False` → report generation skipped, escalation created for clinician review. The clinician resolves the escalation via `POST /clinician/resolve` and retries the report via `POST /report/{thread_id}/retry`.

### 9.4 Diagnosis language guardrail

`validate_llm_response(text)` scans every LLM reply before it reaches the patient:

```python
r"\byou\s+(have|likely\s+have|probably\s+have)\b"
r"\bdiagnos(is|ed|ing|e)\b"
r"\bconsistent\s+with\b"
r"\bpresent(s|ing)?\s+with\b"
r"\blikely\s+caused?\s+by\b"
```

Match → reply is replaced with: "I've noted your symptoms. The clinician will review everything when they see you."

Also applied to the generated report text in `_validate_report_content()`.

**Why this matters in healthcare:** A clinical intake assistant that says "you likely have appendicitis" creates legal liability and can cause patient harm (patient refuses to go to the ER because the chatbot said it was probably nothing). The guardrail is aggressive — false positives (blocking "presenting with a headache") are acceptable; false negatives (allowing "you have hypertension") are not.

---

## 10. Schemas and validated snapshots

**File:** [app/schemas.py](app/schemas.py)

### 10.1 Why Pydantic for LLM output

The LLM produces text. Text is unreliable. Without schema validation, a Gemini response like `{"severity": null, "opqrst": "chest pain"}` would:
1. Set `severity = None` in state (downstream triage crashes on `None.lower()`)
2. Set `opqrst = "chest pain"` as a string instead of a dict (FHIR builder crashes on string iteration)

Pydantic intercepts these at the boundary. Invalid field types trigger the repair cycle before anything downstream sees the data. Invalid Literal values (e.g., `intake_classification: "hospital"`) fail Pydantic validation and go to repair before being accepted.

### 10.2 SubjectiveOut

```python
class SubjectiveOut(BaseModel):
    chief_complaint:          str (max 300)
    opqrst:                   OPQRSTFields
    is_complete:              bool
    reply:                    str (max 400)
    extraction_confidence:    Literal["high", "medium", "low"]
    intake_classification:    Literal[...] | None    # combined — no second LLM call
    classification_confidence: Literal[...] | None
```

`OPQRSTFields` is a nested model so the LLM cannot invent extra keys. Pydantic rejects any JSON that has keys not in the model definition. This prevents the LLM from adding `"onset_details"` or `"quality_description"` as extra top-level fields that would then silently go nowhere.

All fields have `max_length` bounds. A Gemini output with a 2000-char chief complaint hits the `max_length=300` validator and triggers repair rather than silently storing a 2000-char string in state.

### 10.3 ReportInputState — the canonical validated snapshot

Both the clinician note and the FHIR bundle are generated from `ReportInputState`, never from raw `IntakeState`:

```python
validated = _build_validated_report_state(state)
note  = get_gemini().generate_text(prompt=json.dumps(validated.model_dump()))
bundle = fhir_builder.build_bundle(validated.model_dump())
```

`ReportInputState` applies one final round of validation:
- Allergies: drops blank entries, caps list at 20
- Medications: drops entries with no name, caps list at 30
- PMH/recent results: drops blank entries, caps at 20 each
- All string fields: bounded by `max_length`

This is the explicit provenance boundary: both artifacts are demonstrably generated from validated structured state, not from raw chat content. It's also the audit boundary — if a clinician ever questions what data went into a report, `ReportInputState.model_dump()` is what was passed to Gemini.

---

## 11. Database design

**Files:** [app/sqlite_db.py](app/sqlite_db.py), [app/schema.sql](app/schema.sql)

Two SQLite databases in `/data/`.

### 11.1 Why SQLite

This is a portfolio project running on a single server. SQLite with WAL mode handles multiple concurrent readers and a single writer. It requires no separate server process, no connection pool management, and no separate backup infrastructure. The data volumes are small (a clinical intake is hundreds of rows, not millions).

The design is explicitly not premature optimization — the schema, naming conventions, and query patterns are designed so that a migration to PostgreSQL would require changing only `sqlite_db.py`, not the node or API code.

### 11.2 app.db — 13 tables

| Table | Purpose |
|---|---|
| `sessions` | One row per intake; `thread_id`, `status`, `session_token` (SHA-256 hash), timestamps |
| `messages` | Full conversation history by `thread_id`, role, text |
| `reports` | Clinician note, FHIR bundle JSON, `pending_review` flag |
| `escalations` | Crisis/emergency/identity events with structured `payload_json` |
| `jobs` | Background report generation queue; status: queued/running/done/failed |
| `session_state` | Compact state snapshot for fast API reads without checkpoint deserialization |
| `idempotency` | SHA-256 keyed response cache; prevents duplicate processing |
| `mock_ehr` | Demo patient records for identity verification in development |
| `emergency_phrases` | Hot-reloadable phrase list; no redeploy for updates |
| `llm_failure_log` | Every fallback/repair event with node name and raw_snippet |
| `llm_usage` | Per-call token counts and `cost_usd` for billing; aggregatable by thread_id |
| `webhook_deliveries` | Every outbound delivery attempt with retry state and HTTP status |
| `prompt_experiments` | A/B experiment registry; sessions routed deterministically by thread_id hash |

### 11.3 checkpoints.db

LangGraph's SQLite checkpointer manages this database entirely. One table, opaque binary checkpoint blobs keyed by `thread_id` and checkpoint ID. The application code never queries this table directly — LangGraph handles all reads and writes. `prune_old_checkpoints(days=30)` is the only app-level operation on this file.

### 11.4 Connection model

Single global connection with a threading lock (`_db_lock = threading.Lock()`). All writes go through `exec_one()` which acquires the lock and calls `c.commit()`. All reads go through `fetch_one()` / `fetch_all()` which also acquire the lock.

```python
def exec_one(q: str, p: tuple = ()) -> None:
    def _exec():
        with _db_lock:
            c = conn()
            c.execute(q, p)
            c.commit()
    _retry_db_operation(_exec)
```

WAL mode (`PRAGMA journal_mode=WAL`) allows concurrent reads while a write is in progress. In practice, the clinician dashboard can read session lists while a patient chat is writing — without WAL, the read would be blocked by the write lock.

`_retry_db_operation()` wraps every operation in up to 3 retries on `OperationalError: database is locked`. Under sustained concurrent load, SQLite can still block briefly. The retry prevents a single busy-lock from surfacing as a 500 error.

### 11.5 Schema migrations

`init_schema()` runs `schema.sql` via `executescript()` (all `CREATE TABLE IF NOT EXISTS` — idempotent). Then inline `ALTER TABLE` migrations check `PRAGMA table_info()` for columns added after initial deployment:

```python
if "session_token" not in existing_sessions:
    c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
```

No migration tool, no version tracking file. The migration logic is in `init_schema()` and is safe to run on both fresh and existing databases. For a production system with multiple concurrent workers this would need a migration lock; for a single-process deployment it is sufficient.

### 11.6 Token usage and cost tracking

```sql
CREATE TABLE llm_usage (
  thread_id TEXT, node TEXT,
  input_tokens INT, output_tokens INT, cost_usd REAL,
  created_at TEXT
);
```

`cost_usd` is computed at write time from Gemini Flash pricing (`$0.075/1M input`, `$0.30/1M output`). The constants are named in `sqlite_db.py`. The `llm_usage` table enables:
- Per-session cost reports: `SELECT SUM(cost_usd) FROM llm_usage WHERE thread_id=?`
- Node-level cost breakdown: `SELECT node, SUM(cost_usd) FROM llm_usage GROUP BY node`
- Trend monitoring: `SELECT DATE(created_at), SUM(cost_usd) FROM llm_usage GROUP BY DATE(created_at)`

---

## 12. Authentication design

Two separate auth systems for two separate user populations.

### 12.1 Patient session tokens (Bearer tokens)

**Flow:**
1. `POST /start` generates `secrets.token_hex(32)` — 64 hex characters, 256 bits of entropy.
2. `db.create_session(thread_id, session_token=token)` stores `SHA256(token)` in `sessions.session_token`. The raw token is never stored.
3. The response returns `{"session_token": token, "thread_id": "..."}`. The client stores this in memory.
4. Every subsequent patient request sends `Authorization: Bearer <token>`.
5. `verify_session_token(thread_id, token)` computes `SHA256(token)` and compares with the stored hash using `hmac.compare_digest` (constant-time).

**Why SHA-256 stored, not plaintext:**
If the database is compromised, an attacker who reads the `sessions` table gets hashed tokens, not raw tokens. SHA-256 of a 64-byte random token is not reversible in practice. This is analogous to how passwords are hashed — you never store what you'd accept.

**Why `hmac.compare_digest` instead of `==`:**
String equality in Python short-circuits on the first differing character. An attacker who can measure response time can binary-search the correct token character by character (timing oracle attack). `hmac.compare_digest` always compares all characters in constant time.

**Why per-session tokens rather than a shared secret:**
A shared API secret means any client who knows the secret can read any patient's data. A per-session token means a client can only access the one session for which the token was issued. Even if two patients share a device, their tokens are independent.

### 12.2 Clinician JWT tokens

**Flow:**
1. `POST /clinician/token` with `password=<CLINICIAN_PASSWORD>` from settings.
2. On match: `jwt.encode({"sub": "clinician", "exp": time.time() + 86400}, jwt_secret, "HS256")`.
3. Clinician stores the JWT and sends it as `Authorization: Bearer <jwt>` on clinician endpoints.
4. `require_clinician(authorization)` calls `jwt.decode(token, jwt_secret, ["HS256"])`. Expired → 401 "Token expired." Invalid signature → 401 "Invalid token."

**Why JWT for clinicians but plain tokens for patients:**
JWTs carry an expiry claim (`exp`) that is verified without a database lookup. For clinicians who may use the same token across multiple sessions, automatic expiry after 24 hours is the right default. For patients, the session is tied to a specific intake thread and the token is naturally invalidated when the session expires — no separate expiry needed in the token itself.

**Why a single clinician password instead of per-clinician accounts:**
This is a portfolio project without a user management system. The clinician password grants access to all clinician endpoints. In a production system this would be replaced by per-clinician accounts with role-based access control (RBAC).

### 12.3 Rate limiting

`slowapi` wraps all rate limits. Patient endpoints are limited at the route level:
- `POST /start`: 10/hour (prevents session farming)
- `POST /chat`: 60/minute (prevents flooding)
- `GET /jobs/*`: 120/minute (polling is frequent)
- `POST /clinician/token`: 5/minute (prevents credential brute-force)

The limiter uses the client IP as the key. If `slowapi` is not installed, the stub implementation passes all requests through — useful in test environments without requiring the dependency.

---

## 13. FHIR R4 integration

**Files:** [app/fhir_builder.py](app/fhir_builder.py), [app/integrations/fhir_client.py](app/integrations/fhir_client.py)

### 13.1 Bundle construction

`fhir_builder.build_bundle(state_dict)` constructs a FHIR R4 Bundle (type=`document`) from the validated intake state. Resources included:

| Resource | Built from |
|---|---|
| `Patient` | identity (name, DOB, phone, address) |
| `Condition` | chief complaint + OPQRST narrative text |
| `AllergyIntolerance` | one resource per normalized allergy string |
| `MedicationStatement` | one resource per medication (name, dose, frequency) |
| `Observation` | triage risk level, visit type, rationale |

Each resource is a plain Python dict — no external FHIR library required. The bundle is serialized to JSON and stored in `reports.fhir_bundle`. DOB normalization (`_normalize_dob`) converts `MM/DD/YYYY` to FHIR-required `YYYY-MM-DD` format.

**Two-stage validation:**
1. `validate_fhir_input(state)` — pre-build: warns if name, DOB, or chief complaint are missing
2. `validate_fhir_bundle(bundle)` — post-build: checks that required resource types are present, all entries have `fullUrl` and `resource`, and resource-specific required fields are set

Both validation stages log warnings but do not block the intake flow. A structurally imperfect bundle is still saved and dispatched — the EHR can reject it and the delivery failure is tracked in `webhook_deliveries`.

### 13.2 FHIR server push

`fhir_client.push_bundle(fhir_bundle_json, thread_id)` converts the document bundle to a transaction bundle (so individual resources become searchable in the FHIR server), then POSTs to `FHIR_SERVER_URL`.

The conversion wraps each entry in a `PUT` request entry:
```json
{"request": {"method": "PUT", "url": "Patient/<id>"}}
```

`PUT` with a resource ID is a conditional create — the same patient's record can be updated on a second intake rather than creating a duplicate.

The function returns `{"ok": bool, "status": int, "error": str}` and never raises. A failed push is logged but does not affect the patient-facing response. The bundle is already saved to DB — it can be retried later via the webhook delivery system.

In Docker Compose, `FHIR_SERVER_URL` is set to `http://hapi-fhir:8080/fhir` (the HAPI FHIR R4 container). For production, point it at Azure Health Data Services, GCP Healthcare API, or Epic.

---

## 14. Webhook delivery system

**File:** [app/webhook.py](app/webhook.py)

### 14.1 Non-blocking dispatch

All three event types dispatch in daemon threads:
```python
def _dispatch_in_thread(target, kwargs):
    t = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    t.start()
```

**Patient safety rationale:** The old synchronous design meant a slow Slack response delayed the patient's 988 Lifeline message by the full retry sequence (2s + 8s + 30s = 40s worst case). For crisis and emergency alerts, the patient-facing response cannot wait for a network call. Daemon threads decouple alert delivery from the response path entirely.

### 14.2 Delivery tracking

`_post_with_retry()` is the shared delivery engine for all channels. For every delivery attempt:

1. **Idempotency check:** `SHA256(payload_body)` against previously successful deliveries for the same `thread_id + event_type`. If already delivered successfully → skip.
2. **Create row:** `webhook_deliveries` table gets a new row with status `pending`.
3. **POST with retry:** `[2s, 8s, 30s]` delay sequence. `requests.post` with a 10-second timeout.
4. **Update row:** After each attempt, update `status`, `attempts`, `last_http_status`, `last_error`, `next_retry_at`.
5. **Terminal states:** `success` (2xx HTTP), `exhausted` (all retries failed).

### 14.3 HMAC signing

The FHIR completion webhook includes a signature:
```
X-Signature: sha256=<hmac_hex>
X-Thread-Id: <session_uuid>
```

The receiving EHR verifies: `hmac.new(secret, body, sha256).hexdigest()` compared with the header value using `hmac.compare_digest`. This prevents spoofed payloads from a third party who knows the endpoint URL but not the shared secret.

### 14.4 Dead-letter recovery

Sessions where all webhook deliveries were exhausted are eligible for re-queue after 24 hours. Two recovery paths:

**On startup:** `retry_exhausted_webhooks()` queries `webhook_deliveries` for `status=exhausted` records older than `dead_letter_retry_after_hours`, resets them to `status=pending`, and re-dispatches them in background threads.

**Hourly background task:** An `asyncio` task in `main.py` runs the same recovery every 3600 seconds during normal operation:
```python
async def _dead_letter_loop():
    while True:
        await asyncio.sleep(3600)
        n = retry_exhausted_webhooks()
```

This handles the case where the downstream endpoint was down for days — records keep getting retried until they either succeed or exceed `dead_letter_max_lifetime_attempts` (default 6).

---

## 15. Observability

**File:** [app/logging_utils.py](app/logging_utils.py)

### 15.1 Structured JSON logging

Every log line is a JSON object:
```json
{
  "ts": "2026-04-08T12:00:00.000Z",
  "level": "info",
  "event": "llm_step",
  "trace_id": "e885c9d2-a91a-...",
  "request_id": "04b69baf-da80-...",
  "node": "subjective_node",
  "prompt_version": "v1.3",
  "latency_ms": 842,
  "input_tokens": 310,
  "output_tokens": 185,
  "fallback_used": false,
  "repair_used": false
}
```

Plain text logs require grep + awk to extract fields. JSON logs can be fed directly to Elasticsearch, Datadog, CloudWatch Insights, or any structured log aggregator. A single query on `thread_id` reconstructs the full session waterfall.

### 15.2 PHI masking

`mask_phi(fields)` runs automatically inside `log_event()` and `log_audit()` before any log line is emitted:

```python
_PHI_FIELDS = frozenset({
    "name", "patient_name", "dob", "date_of_birth",
    "phone", "address", "identity", "stored_identity"
})
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),   # phone
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),          # DOB
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                 # ISO date
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                 # SSN
)
```

Rule 1: any key in `_PHI_FIELDS` → entire value replaced with `"[REDACTED]"`.
Rule 2: any string value matching a phone/DOB/SSN pattern → replaced with `"[REDACTED]"`.
Rule 3: nested dicts are recursively masked.

This is intentionally conservative. Application logs never contain PII regardless of which fields the caller passes — masking is automatic, not opt-in.

### 15.3 Trace ID propagation with ContextVar

```python
_trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)

def set_trace_id(trace_id: str) -> None:
    _trace_id_ctx.set(trace_id)
```

`ContextVar` is Python's async-safe equivalent of thread-local storage. `set_trace_id(thread_id)` is called at the start of `/start` and `/chat`. All `log_event()` calls made during that request — inside LangGraph nodes, LLM calls, FHIR builders — automatically include `"trace_id": thread_id` without any argument threading.

The four context variables are:
- `trace_id` — the patient's `thread_id` (session correlation)
- `request_id` — the HTTP request UUID (request correlation)
- `job_id` — the background job ID (report generation correlation)
- `node_id` — the currently executing graph node

### 15.4 Prompt versioning

```python
PROMPT_VERSIONS: dict[str, str] = {
    "subjective":  "v1.3",
    "medications": "v1.1",
    "report":      "v1.1",
}
```

Every `log_event("llm_step")` includes `prompt_version`. When extraction quality degrades after a deployment, the logs show exactly which prompt version ran on which sessions. Compare `extraction_quality_score` distributions grouped by `prompt_version` to identify which change caused the regression. Bump minor version on wording changes; bump major version on schema changes.

### 15.5 Key events and what they mean

| Event | Level | Meaning |
|---|---|---|
| `session_started` | info | Patient session created |
| `phase_transition` | info | State machine advanced to new phase |
| `llm_step` | info | One Gemini call completed; latency, tokens, version |
| `llm_fallback_used` | warning | All LLM paths failed; hardcoded fallback used |
| `extraction_quality_retry` | info | Quality below threshold; gap-fill question sent |
| `guardrail_crisis_detected` | warning | Crisis language matched; 988 message sent |
| `guardrail_diagnosis_blocked` | warning | LLM reply contained diagnosis language; replaced |
| `emergency_escalation` | warning | Emergency phrase detected; patient routed to handoff |
| `circuit_breaker_opened` | warning | 5 consecutive LLM failures; breaker now open |
| `circuit_breaker_closed` | info | Breaker recovered after probe succeeded |
| `fhir_bundle_built` | info | Bundle generated; includes structural warnings if any |
| `report_job_done` | info | Background report generation completed |
| `report_blocked_preflight` | warning | SafetyChecker blocked report; clinician review required |
| `webhook_delivery_success` | info | Outbound delivery confirmed |
| `webhook_delivery_failed` | warning | Attempt failed; retrying |
| `phi_audit_session_complete` | info | Audit trail entry at session end |

---

## 16. The frontend

**Files:** [static/index.html](static/index.html), [static/app.js](static/app.js)

One HTML file with vanilla JavaScript. No framework, no build step, no bundler. The design choice here is deliberate for a portfolio project — it demonstrates that the complexity is in the backend, not in the frontend toolchain. Anyone reviewing the code can open `app.js` without needing to know React, Vue, or any build system.

### 16.1 Session state in memory

Three pieces of state held in JavaScript module-level variables:

```javascript
let sessionToken = null;   // Bearer token from /start; sent on every request
let threadId = null;       // Session UUID from /start
let clientMsgId = 0;       // Incrementing integer; unique per message sent
```

`sessionToken` is in memory only — not in `localStorage`, not in a cookie. If the user refreshes the page, the session is lost. This is intentional: healthcare sessions should not persist across page reloads without explicit re-authentication. The `/resume/{thread_id}` endpoint exists for applications that want to support resumable sessions with a saved token.

### 16.2 Start flow

```javascript
async function start() {
    const res = await fetch("/start", {method: "POST", body: new FormData()});
    const j = await res.json();
    threadId = j.thread_id;
    sessionToken = j.session_token;  // stored here; used on all subsequent calls
    // render j.reply as first assistant message
}
```

### 16.3 Chat loop

```javascript
async function sendMsg(msg) {
    const fd = new FormData();
    fd.append("thread_id", threadId);
    fd.append("message", msg);
    fd.append("client_msg_id", String(clientMsgId++));

    const res = await fetch("/chat", {
        method: "POST",
        body: fd,
        headers: {"Authorization": `Bearer ${sessionToken}`},
    });
    const j = await res.json();

    if (j.phase === "report_generating") {
        waitForReport(j.job_id);
    }
}
```

The `Authorization` header is set explicitly alongside the FormData body. The browser does not automatically attach session credentials to fetch requests — it must be set manually.

### 16.4 Report polling

```javascript
async function waitForReport(jobId) {
    const poll = setInterval(async () => {
        const res = await fetch(`/jobs/${jobId}`, {
            headers: {"Authorization": `Bearer ${sessionToken}`}
        });
        const j = await res.json();
        if (j.status === "done") {
            clearInterval(poll);
            loadReport(threadId);
        } else if (j.status === "failed") {
            clearInterval(poll);
            showError("Report generation failed.");
        }
    }, 2000);
}
```

Polls every 2 seconds. On `done`, fetches the report. On `failed`, shows an error with a retry button. The polling interval is chosen to be responsive (patient doesn't wait more than 2 extra seconds after generation completes) without excessive server load.

### 16.5 Clinician panel

The same `index.html` contains a clinician panel that is hidden by default. Clicking "Clinician Login" shows a password form, POSTs to `/clinician/token`, and stores the JWT:

```javascript
let clinicianToken = null;
// POST /clinician/token → clinicianToken = j.access_token
// All clinician requests send: headers: {"Authorization": `Bearer ${clinicianToken}`}
```

The clinician panel shows pending escalations, allows adding a nurse note, and resolves cases via `POST /clinician/resolve`.

### 16.6 Cache busting

`index.html` references `app.js?v=2` and `styles.css?v=2`. The query string causes browsers to treat each version as a distinct URL. When `app.js` changes, incrementing the version number forces all browsers to fetch the new file rather than serving the cached old version. Without this, browser cache can serve old JavaScript indefinitely regardless of what the server returns.

---

## 17. Router package architecture

**Directory:** [app/api/](app/api/)

The API layer is organized as a package of four router modules, each owning one concern. This replaced a single large `api.py` with all routes in one file.

| Module | Prefix | Who uses it | What it does |
|---|---|---|---|
| `patient.py` | (none) | Patient browser | `/start`, `/chat`, `/resume`, `/report`, `/jobs` |
| `clinician.py` | `/clinician` | Clinician with JWT | Escalation review, case detail, FHIR fetch, A/B experiments |
| `admin.py` | `/admin` | Ops/demo | Emergency phrase management, demo scenarios |
| `health.py` | (none) | Load balancer, monitoring | `/health`, `/ready`, `/analytics` |
| `deps.py` | — | All routers | Rate limiter, `require_session_token`, `require_clinician` |

**Design decision: shared deps.py instead of each router declaring its own auth:**
If each router re-declared `require_session_token`, changing the auth logic (e.g., adding token blacklisting) would require editing four files. With `deps.py` as the single source, one change propagates to all routes.

**Design decision: router prefixes in the router module, not in main.py:**
`router = APIRouter(prefix="/clinician")` is in `clinician.py`. This means the prefix is visible when reading the route file — you don't need to open `main.py` to know that `/token` in `clinician.py` maps to `/clinician/token`. The convention is that `patient.py` has no prefix because its routes are at the root.

**How `main.py` assembles the application:**
```python
def create_app() -> FastAPI:
    app = FastAPI(title="Clinical Intake", lifespan=lifespan)
    app.add_middleware(CorrelationMiddleware)
    app.add_middleware(CORSMiddleware, ...)
    app.mount("/static", StaticFiles(...))
    app.include_router(patient_router)
    app.include_router(clinician_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    return app

app = create_app()
```

`create_app()` is a factory function rather than module-level code because it makes the application testable — test fixtures can call `create_app()` after patching settings or database paths, without the factory running automatically on import.

---

## 18. Docker setup

**Files:** [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)

### 18.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Single stage, slim base image. `requirements.txt` is copied and installed before `COPY . .` so Docker's layer cache reuses the install layer when only application code changes (the common case). Rebuilds triggered by code-only changes skip the pip install.

### 18.2 docker-compose.yml

Two services:

**app** — the FastAPI process. Mounts `./data:/app/data` so both SQLite databases persist across container restarts. Mounts `./static:/app/static` so JavaScript/CSS changes take effect without rebuilding the image. Reads `.env` via `env_file`.

**hapi-fhir** — HAPI FHIR R4 reference server. Mounted `hapi-data` named volume with `user: root` to ensure the H2 database files are writable regardless of the host filesystem UID. Runs the healthcheck but the app dependency is `condition: service_started` (not `service_healthy`) — HAPI takes 30–90 seconds to become healthy and the app handles FHIR push failures gracefully, so blocking startup on FHIR readiness would make every `docker compose up` slow.

### 18.3 Environment variables

All secrets and environment-specific config come from `.env`. Required variables:
```
GEMINI_API_KEY=         # Gemini API key
JWT_SECRET=             # Strong random string for JWT signing
CLINICIAN_PASSWORD=     # Password for clinician panel
```

Optional:
```
SLACK_WEBHOOK_URL=      # Slack incoming webhook for alerts
COMPLETION_WEBHOOK_URL= # EHR endpoint for FHIR bundle delivery
COMPLETION_WEBHOOK_SECRET= # HMAC secret for webhook signing
FHIR_SERVER_URL=        # Direct FHIR push (set by docker-compose to hapi-fhir)
DEBUG_MODE=true         # Skip Gemini validation on startup (for CI/testing)
```

---

## 19. Design decisions index

A reference list of the significant design choices and the reasoning behind each one.

| Decision | File | Why |
|---|---|---|
| `guard_node` as centralized pre-processor | graph.py, nodes.py | Adding a new business node tomorrow can never accidentally skip safety; side effects happen in one place |
| `route_after_guard` → END on crisis, not handoff_node | graph.py | handoff_node adds a second patient message; guard already set CRISIS_RESOURCE once |
| `guard_node` NOT in `interrupt_after` | graph.py | Adding it pauses after safety check, starves the business node of the current message |
| Two-tier intent classification | nodes.py, schemas.py | Hard words are free (set lookup); ambiguous short phrases go to LLM with max_tokens=40 — replaces ever-growing keyword lists |
| LLM identity extraction via IdentityOut + validators | nodes.py, schemas.py | Regex fails on ordinal dates, mixed formats, titles, hyphenated names — normalization at schema boundary is testable |
| `None` vs `[]` sentinel for clinical fields | state.py, nodes.py | None = not yet asked; [] = asked and patient said none — enables lookahead skip logic |
| `_prescan_volunteered_clinical` lookahead | nodes.py | Patients who volunteer "no allergies, I take lisinopril" during symptoms don't have to repeat it |
| `_try_correction` shared helper | nodes.py | Go-back logic from any interactive node without duplicating regex in each node |
| Natural-language `_confirm_summary` | nodes.py | Paragraph reads like a nurse reading back notes; table looks like a form printout |
| Dosage second-attempt gate | nodes.py | Ask once warmly; accept "I don't know" on second attempt — never trap a patient in a loop |
| Report shown in chat | nodes.py | Patient gets the report immediately; "click View Report" requires a second action they may miss |
| Warm returning patient message | nodes.py | "Welcome back, Jane!" instead of a raw data comparison table; conveys that the system recognises them |
| Three `/chat` guards (circuit breaker, max turns, active job) | api/patient.py | Fail fast before graph invocation; return clear messages instead of hanging or ambiguous errors |
| `_build_resume_context` for session resume | api/patient.py | Phase-aware human-readable summary instead of raw state dump |
| `events.py` EventBus | events.py | Decouples nodes from notification channels; adding a new handler (email, pager) doesn't touch node code |
| Slack crisis alert with partial identity + message preview | webhook.py | Hospital staff can act on "Unknown patient typed: I want to hurt myself" even without identity collected |
| LangGraph for state machine | graph.py | Durable checkpointing — server restart mid-intake resumes from the same node |
| `current_phase` as a string, not enum | state.py | Trivial SQLite serialization; readable in checkpoint blobs |
| `route()` reads state, no LLM | graph.py | Flow control must be deterministic; LLM decides language, not control flow |
| `validate_node` as a separate non-interactive node | graph.py | Separates extraction from completeness enforcement; independently testable |
| Combined extraction + classification in `SubjectiveOut` | schemas.py, nodes.py | Halves LLM latency on first chief complaint message |
| All LLM calls through `run_json_step` | llm.py | Standardizes 3-level degradation; all callers get the same reliability guarantees |
| Circuit breaker with 60s recovery window | llm.py | Sustained outage: fail fast instead of queuing 40s retries per request |
| Full jitter for retry backoff | llm.py | Prevents thundering herd when multiple processes restart after an outage |
| Hardcoded fallback dicts, not generated | nodes.py | Fallback values must be reviewed and audited; runtime generation is unauditable |
| `ReportInputState` as the report boundary | schemas.py, nodes.py | Provenance: both artifacts demonstrably from validated state, never raw chat |
| SHA-256 token stored, never plaintext | sqlite_db.py | DB compromise exposes hashes, not tokens |
| `hmac.compare_digest` for token comparison | sqlite_db.py | Constant-time; prevents timing oracle attacks |
| Session tokens (per-session) vs JWTs (per-clinician) | deps.py | Session tokens: natural expiry with session. JWTs: cross-request reuse with built-in `exp` |
| PHI masking in `log_event` | logging_utils.py | Automatic; no caller can forget to sanitize before logging |
| `ContextVar` for trace ID propagation | logging_utils.py | Async-safe thread-local; no argument threading through call stacks |
| Daemon threads for webhook dispatch | webhook.py | Crisis/emergency alerts cannot wait for Slack HTTP timeouts |
| Dead-letter recovery on startup AND hourly | main.py, webhook.py | Startup handles prior-process failures; hourly loop handles extended downstream outages |
| Two SQLite databases | sqlite_db.py, graph.py | LangGraph manages checkpoints.db schema; app.db has its own migrations |
| WAL mode on both databases | sqlite_db.py, graph.py | Concurrent readers (clinician dashboard) do not block patient chat writes |
| `IntakeConfig` as nested settings | settings.py | All magic numbers in one place; overridable via env vars without code changes |
| Fail at startup for placeholder secrets | settings.py | Early failure is more honest than a 500 on first patient request |
| Singleton `get_settings()` | settings.py | One file read per process lifetime; callers always call `get_settings()` not an alias |
| `get_settings as settings` anti-pattern avoided | all callers | Aliasing the function without calling it assigns the function object — use `get_settings()` |
| Rate limits per-IP not per-session | deps.py | Prevents a single IP from flooding regardless of how many sessions they create |
| Client-generated `client_msg_id` for idempotency | patient.py | Distinguishes duplicate submission (safe to cache) from corrected retry (should process) |
| Per-request TTL enforcement, not background jobs | sqlite_db.py | Guaranteed to run; no silent cron failure; negligible overhead per request |
| `compact_snapshot` separate from checkpoint | patient.py, sqlite_db.py | Fast API reads without checkpoint deserialization overhead |
| No LLM for consent, identity, confirm routing | nodes.py | Binary decisions in healthcare require deterministic logic; probabilistic would be unsafe |
| Emergency phrases in DB, not code | sqlite_db.py | Hot-reloadable; clinicians can add phrases without a redeploy |
| `v=2` query string on static file references | index.html | Browser cache busting; version increment forces fresh fetch after JS changes |
| Router package (api/) instead of single api.py | api/ | Each concern testable and reviewable in isolation; shared deps in one place |
| No framework in the frontend | static/app.js | Backend complexity is the point; frontend simplicity makes it reviewable |
| `user: root` on HAPI FHIR container | docker-compose.yml | Named volumes can have root-owned files on some host filesystems |
| `service_started` not `service_healthy` for HAPI | docker-compose.yml | HAPI takes 90s to become healthy; app handles FHIR failures gracefully |
| Prompt version registry in `PROMPT_VERSIONS` | prompts.py | Timestamps prompt changes in logs; enables regression detection by version |
| `max_length` on all LLM output fields | schemas.py | Prevents runaway LLM text from polluting DB or generating oversized reports |
| FHIR transaction bundle (not document bundle) for server push | fhir_client.py | Transaction bundles create individual searchable/patchable resources in FHIR server |
| Two-stage FHIR validation (pre and post build) | fhir_builder.py | Pre-build catches missing input; post-build catches construction bugs |

---

## 20. Code walkthrough

This section traces the actual code for three representative paths through the system: a normal patient message, a crisis detection, and a returning patient identity confirmation. Reading these three paths covers the majority of the code that runs in production.

---

### 20.1 A normal patient message — line by line

**Scenario:** Patient is in the subjective phase and types "I've had a throbbing headache for two days, 7 out of 10, gets worse when I bend over."

**Step 1 — Browser** ([static/app.js](static/app.js))

```javascript
const fd = new FormData();
fd.append("thread_id", threadId);
fd.append("message", "I've had a throbbing headache...");
fd.append("client_msg_id", String(clientMsgId++));
const res = await fetch("/chat", {
    method: "POST", body: fd,
    headers: {"Authorization": `Bearer ${sessionToken}`}
});
```

`clientMsgId` increments on every send so each message has a unique idempotency key.

**Step 2 — CorrelationMiddleware** ([app/main.py](app/main.py):65)

Before the route handler runs, middleware generates (or reads from `X-Request-Id` header) a UUID and calls `set_request_id(req_id)`. This stores the ID in a `ContextVar` so every `log_event()` call during this request automatically includes `request_id` without needing to pass it explicitly through the call stack.

**Step 3 — Rate limiter** ([app/api/deps.py](app/api/deps.py):43)

`@limiter.limit("60/minute")` on the `/chat` route. Uses the client IP as key via `get_remote_address`. If this IP has sent more than 60 requests in the past minute, returns 429 immediately.

**Step 4 — Session token verification** ([app/api/deps.py](app/api/deps.py):46)

```python
token = authorization.removeprefix("Bearer ").strip()
if not db.verify_session_token(thread_id, token):
    raise HTTPException(status_code=401)
```

`verify_session_token` computes `SHA256(token)` and compares with the stored hash using `hmac.compare_digest` (constant-time, prevents timing oracle attacks).

**Step 5 — Input validation** ([app/api/patient.py](app/api/patient.py):225)

Message length check (1–1200 chars). Then `check_prompt_injection(message)` — regex scan for "ignore previous instructions", "you are now", "forget your training". If matched, returns a fixed neutral reply without touching the graph.

**Step 6 — Idempotency** ([app/api/patient.py](app/api/patient.py):241)

```python
request_hash = hashlib.sha256(message.encode()).hexdigest()
prev = db.get_idempotent_response(thread_id, client_msg_id)
if prev:
    if prev["request_hash"] != request_hash:
        raise HTTPException(409)  # same ID, different content — reject
    return json.loads(prev["response_json"])  # duplicate send — return cached
```

**Step 7 — Three pre-flight guards** ([app/api/patient.py](app/api/patient.py):259)

```python
if not is_llm_available():        # circuit breaker open
    raise HTTPException(503, "We're experiencing a brief technical issue...")
turn_count = len(state.get("messages") or [])
if turn_count >= settings().intake.max_session_turns:  # 30 turns
    raise HTTPException(400, "Your session has reached its maximum length...")
active_job = db.get_active_job(thread_id)
if active_job:                    # report generation already in progress
    raise HTTPException(409, "Your intake summary is currently being prepared...")
```

All three guards run before any graph invocation. If any fires, the patient gets a clear message immediately without the graph starting.

**Step 8 — Graph invocation** ([app/api/patient.py](app/api/patient.py):280)

```python
output = graph.invoke(
    {"messages": [{"role": "user", "text": message}]},
    {"configurable": {"thread_id": thread_id}}
)
```

LangGraph loads the checkpoint for this `thread_id` from `checkpoints.db`, appends the new message via `operator.add` reducer, then runs `route(state)`.

**Step 9 — guard_node** ([app/nodes.py](app/nodes.py):176)

`route()` always starts at `guard_node`. `guard_node` calls `last_user(state)` to get the patient's message, then runs `detect_crisis(user)`. For this headache message, no crisis phrases match. `has_soft_distress(user)` also returns false. `guard_node` returns `{}` (empty dict — no state changes). `route_after_guard(state)` sees no `crisis_detected` flag and calls `route(state)`, which reads `current_phase="subjective"` and returns `"subjective_node"`.

**Step 10 — subjective_node** ([app/nodes.py](app/nodes.py):800)

`_try_correction(user, state)` runs first — no correction keywords in "I've had a throbbing headache for two days." Returns `None`, continues.

`detect_emergency_red_flags(cc, opqrst, user)` runs — no emergency phrases. Continues.

`run_json_step(SubjectiveOut, ...)` is called with:
- `system=subjective_extract_system(RESPONSE_RULES)` — the OPQRST extraction prompt
- `prompt=f"PATIENT_MESSAGES=...\nCURRENT_OPQRST=..."` — conversation window + current state

Gemini returns:
```json
{
  "chief_complaint": "throbbing headache",
  "opqrst": {"onset": "two days ago", "quality": "throbbing", "severity": "7/10", "provocation": "worse when bending over", "radiation": "", "timing": ""},
  "is_complete": false,
  "reply": "How long does the headache last when it comes on, and does anything make it better?",
  "extraction_confidence": "high",
  "intake_classification": "routine_checkup",
  "classification_confidence": "high"
}
```

`SubjectiveOut.model_validate_json(cleaned)` validates the response. All fields pass. `log_event("llm_step", ..., input_tokens=310, output_tokens=185, latency_ms=840, cost_usd=0.000079)` writes to `llm_usage` table.

`score_extraction_quality(cc, opqrst)` → 0.70. Clinic threshold is 0.60 — this passes. `is_complete=False` so the node asks the follow-up question.

`_safe_reply(reply)` runs the diagnosis language filter on "How long does the headache last when it comes on, and does anything make it better?" — no diagnosis language, passes through unchanged.

Node returns `{"messages": [{"role": "assistant", "text": "How long..."}], "current_phase": "subjective", "chief_complaint": "throbbing headache", "opqrst": {...}, "extraction_quality_score": 0.70}`.

**Step 11 — State checkpoint and API response** ([app/api/patient.py](app/api/patient.py):300)

`_compact_snapshot(output)` strips the full message list down to the ~30 fields the API layer needs. `db.save_session_state(thread_id, snapshot)` writes to the `session_state` table. `db.save_message(thread_id, "assistant", reply)` stores the message in the `messages` table.

The response JSON is built, saved to `idempotency` table with this `client_msg_id`, and returned to the browser:
```json
{"reply": "How long does the headache last...", "phase": "subjective", "status": "active"}
```

---

### 20.2 Crisis detection path — line by line

**Scenario:** Patient is in the identity phase and types "I feel like ending it all."

**guard_node** ([app/nodes.py](app/nodes.py):176)

`detect_crisis("I feel like ending it all")` — checks `_CRISIS_PHRASES`. Not an exact match. Checks `_CRISIS_REGEX_PATTERNS`:
- `\bend\w*\s+my\s+life\b` — doesn't match ("ending it all" not "ending my life")
- Other patterns — no match

`has_soft_distress("I feel like ending it all")` — checks soft distress signals. Matches ("ending it all" is a known soft distress pattern).

`llm_crisis_score("I feel like ending it all")` calls Gemini with `CrisisScore` schema:
```json
{"is_crisis_risk": true, "confidence": "high", "reasoning": "Expression of desire to end one's life — genuine suicidal ideation"}
```

`score.is_crisis_risk=True` and `score.confidence="high"` → crisis confirmed via Tier 2.

```python
db.create_escalation(thread_id=thread_id, kind="crisis", payload=build_reason_trail(...))
webhook.dispatch_crisis_alert(
    thread_id=thread_id,
    patient_name="unknown patient",   # identity not yet collected
    matched_phrases=["llm_detected (high): Expression of desire to end one's life"],
    partial_identity={},              # empty — patient hadn't given identity yet
    message_preview="I feel like ending it all"
)
```

`dispatch_crisis_alert` calls `_fmt_partial_identity({})` → "Unknown — crisis occurred before identity was collected". The Slack message reads:
```
🚨 CRISIS LANGUAGE DETECTED — IMMEDIATE ATTENTION REQUIRED
Patient: Unknown — crisis occurred before identity was collected
Session: `f05da0a9-...`
Patient wrote: "I feel like ending it all"
Detected: llm_detected (high): Expression of desire to end one's life
```

`guard_node` returns:
```python
{
    "crisis_detected": True,
    "human_review_required": True,
    "current_phase": "handoff",
    "messages": [{"role": "assistant", "text": CRISIS_RESOURCE}]
}
```

`route_after_guard(state)` sees `crisis_detected=True` and `current_phase="handoff"` → returns `END` directly (not `handoff_node`). The graph finishes. The patient sees the 988 Lifeline message.

---

### 20.3 Returning patient — line by line

**Scenario:** Patient had a previous complete intake. They start a new session, consent, and type "I'm Jane Smith."

**identity_node** ([app/nodes.py](app/nodes.py):500)

`run_json_step(IdentityOut, system=identity_extract_system(), prompt="PATIENT_MESSAGE=I'm Jane Smith.")` calls Gemini. Response:
```json
{"name": "Jane Smith", "dob": "", "phone": "", "address": ""}
```

`IdentityOut._norm_name("Jane Smith")` → "Jane Smith" (already Title Case). `dob`, `phone`, `address` are empty strings — that's fine for partial extraction.

Node merges into state: `identity["name"] = "Jane Smith"`. Other fields still empty. Since not all four fields are collected yet, the node asks: "Thanks Jane — what's your date of birth?"

(Three more turns collect DOB, phone, address.)

When all four fields are populated: `db.get_stored_identity_by_name("Jane Smith")` finds a record:
```python
stored = {"name": "Jane Smith", "dob": "1990-03-15", "phone": "4125550199", "address": "123 Main St"}
```

Node returns:
```python
{
    "stored_identity": stored,
    "messages": [{"role": "assistant", "text":
        "Welcome back, Jane Smith! Your phone on file is 4125550199. Address: 123 Main St. "
        "Does everything still look right, or has anything changed? (reply 'yes' to keep, 'no' to update)"}],
    "current_phase": "identity_review",
}
```

**identity_review_node** ([app/nodes.py](app/nodes.py):640)

Patient replies "yes that's all correct."

`_classify_intent("yes that's all correct", state)`:
- `len("yes that's all correct".split()) = 5` — under the 8-word LLM threshold
- Lowercase: "yes that's all correct" — not in `_HARD_YES` (exact match only)
- Not in `_HARD_NO`
- No correction regex match
- Tier 2 LLM: Gemini classifies → `{"intent": "confirm", "correcting_section": "none"}`

`intent.intent == "confirm"` → returns:
```python
{
    "identity": stored,           # use the verified stored record
    "identity_status": "verified",
    "needs_identity_review": False,
    "messages": [{"role": "assistant", "text": "Thanks — I'll keep what's on file. What brings you in today?"}],
    "current_phase": "subjective",
}
```

No escalation created. Identity verified. Session advances to symptom collection.

---

### 20.4 Key code invariants

These properties hold throughout the codebase and are worth knowing before making changes:

**`current_phase` is the only routing key.** The graph's `route()` function reads only `state["current_phase"]`. No node can skip another by manipulating anything else. If you add a new phase, add it to `route()` and to `interrupt_after` if it requires patient input.

**Nodes return dicts, not new state objects.** LangGraph merges the returned dict with existing state. A node that returns `{"current_phase": "subjective"}` only updates that one field — it doesn't erase everything else. The only field that accumulates instead of replacing is `messages`, because of the `operator.add` reducer defined in `IntakeState`.

**All LLM output goes through Pydantic before use.** No node reads raw LLM text. Every call goes through `run_json_step` → `schema.model_validate_json` → typed model. This means LLM output that exceeds `max_length`, uses wrong types, or invents extra keys always fails validation and goes to repair or fallback before any state is updated.

**PHI never appears in logs.** `log_event()` calls `mask_phi()` unconditionally before writing. Any key named `name`, `dob`, `phone`, `address`, `identity`, or `stored_identity` is redacted. Any string value matching phone/DOB patterns is redacted. Callers don't need to sanitize — the masking is automatic.

**Fallback dicts are in the caller, not in `run_json_step`.** `run_json_step` only knows to fall back to `schema.model_validate(fallback)`. It doesn't know what the fallback values should be. Each node authors its own fallback that makes sense for its phase — subjective fallback has `is_complete=False`, medications fallback has `medications=[]`. This keeps node policy visible at the call site.

**`None` vs `[]` for clinical fields.** `allergies`, `medications`, `pmh`, `recent_results` start as `None` in initial state. `None` means "not yet asked." `[]` means "asked and patient reported none." The clinical_history lookahead only pre-populates `None` fields. The `validate_node` only checks that allergies were asked (i.e., not `None`) before allowing confirm. Never set a clinical field to `None` after it has been asked.
