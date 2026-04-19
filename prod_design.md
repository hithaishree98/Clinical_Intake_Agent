# Production System Design: Clinical Intake AI Agent

> A complete guide to understanding what this system does, how it works today, where it breaks under real traffic, and how to scale it from a single clinic to a hospital network — explained from the ground up.

---

## Table of Contents

1. [What Does This System Do?](#1-what-does-this-system-do)
2. [Who Uses It and How Many?](#2-who-uses-it-and-how-many)
3. [Current Architecture: The Full Picture](#3-current-architecture-the-full-picture)
4. [The Request Lifecycle: What Happens When a Patient Chats](#4-the-request-lifecycle-what-happens-when-a-patient-chats)
5. [The AI Brain: LangGraph State Machine](#5-the-ai-brain-langgraph-state-machine)
6. [The Database Layer: What Is Stored and Why](#6-the-database-layer-what-is-stored-and-why)
7. [The LLM Layer: Gemini + Reliability Patterns](#7-the-llm-layer-gemini--reliability-patterns)
8. [Safety and Guardrails](#8-safety-and-guardrails)
9. [Clinician-Facing Features](#9-clinician-facing-features)
10. [What Will Break in Production](#10-what-will-break-in-production)
11. [Scale Tiers: Small to Huge](#11-scale-tiers-small-to-huge)
12. [Production Changes Required](#12-production-changes-required)
13. [Observability: Knowing When Things Go Wrong](#13-observability-knowing-when-things-go-wrong)
14. [Security Gaps and Fixes](#14-security-gaps-and-fixes)
15. [The Target Production Architecture](#15-the-target-production-architecture)

---

## 1. What Does This System Do?

This is an **AI-powered clinical intake assistant** for hospitals and clinics. Think of it as the clipboard of questions a nurse hands you before the doctor walks in — but automated, intelligent, and delivered over a chat interface.

### The patient experience, step by step:

```
Patient arrives at kiosk or opens link on phone
            |
            v
    [CONSENT] — Patient reads and agrees to data collection
            |
            v
    [IDENTITY] — Bot collects name, date of birth, phone, address
                 Bot checks these against the hospital's EHR (mock, in this code)
                 Flags mismatches for nurse review
            |
            v
    [SUBJECTIVE] — Bot asks about the chief complaint:
                   "What brings you in today?"
                   Then deep-dives with OPQRST:
                   - Onset: When did it start?
                   - Provocation: What makes it better/worse?
                   - Quality: Describe the pain (sharp, dull, burning?)
                   - Radiation: Does it spread anywhere?
                   - Severity: Rate 1-10
                   - Timing: Constant or comes and goes?
            |
            v
    [CLINICAL HISTORY] — Allergies, current medications, past medical history,
                         recent lab results
            |
            v
    [REPORT GENERATION] — AI generates a structured clinical note
                          + FHIR R4 Bundle (healthcare data standard)
                          + Notifies clinician via Slack
                          + Sends FHIR bundle to EHR via webhook
            |
            v
    [CLINICIAN REVIEW] — Doctor/nurse sees the report before walking in
```

Throughout the conversation, the system:
- Detects **emergency language** ("chest pain", "can't breathe") → immediate escalation
- Detects **crisis language** (self-harm, suicidal ideation) → immediate escalation + resource display
- Classifies the visit type (emergency, routine, mental health, specialist referral)
- Scores the quality of the collected information
- Blocks report generation if critical information is missing

---

## 2. Who Uses It and How Many?

### User Types

| User Type | How They Use It | Expected Volume |
|-----------|----------------|-----------------|
| **Patients** | Chat interface (web browser, kiosk) | Most of the load |
| **Clinicians** | Read reports, resolve escalations, view analytics | Low volume, high importance |
| **Admins** | Manage emergency phrases, reset demo | Very low |

### Volume Estimates by Deployment Size

**Small Clinic** (1-2 doctors, walk-in clinic)
- 30-80 patients/day
- Peak: 10 concurrent at lunch hour
- ~1 intake takes 5-15 minutes of conversation
- ~10-30 LLM calls per intake (each node calls Gemini once or twice)

**Medium Hospital** (50-bed hospital, multiple departments)
- 300-800 patients/day
- Peak: 80-150 concurrent
- Multiple kiosks running simultaneously

**Large Hospital Network** (multi-site, 500+ beds)
- 5,000-20,000 patients/day
- Peak: 1,000-3,000 concurrent
- Multiple geographic locations

**Understanding Concurrency**

This is the most important metric. "500 patients/day" does not mean 500 at once. A typical pattern:
- Intakes happen in clusters (morning rush 8-10am, post-lunch 1-3pm)
- Each chat session holds the conversation open for ~10 minutes
- Assume 20% of daily patients are concurrent at peak

So 800 patients/day → 160 concurrent active conversations at peak.

---

## 3. Current Architecture: The Full Picture

```
Internet
    |
    v
[Single Docker Container]
    |
    |-- FastAPI app (uvicorn, single worker)
    |       |
    |       |-- /start, /chat, /resume, /report, /jobs (patient-facing)
    |       |-- /clinician/* (protected by JWT)
    |       |-- /admin/*, /analytics (admin)
    |
    |-- LangGraph Graph (compiled once at startup, held in memory)
    |       |
    |       |-- consent_node
    |       |-- identity_node
    |       |-- identity_review_node
    |       |-- subjective_node
    |       |-- validate_node
    |       |-- clinical_history_node
    |       |-- report_node
    |       |-- handoff_node
    |       |-- confirm_node
    |
    |-- GeminiClient (singleton, held in memory)
    |       |-- Circuit breaker (in-memory state)
    |       |-- Retry logic (exponential backoff with jitter)
    |
    |-- SQLite: app.db (all application data)
    |       |-- sessions
    |       |-- messages
    |       |-- reports
    |       |-- escalations
    |       |-- jobs
    |       |-- session_state
    |       |-- idempotency
    |       |-- llm_failure_log
    |       |-- webhook_deliveries
    |       |-- mock_ehr
    |       |-- emergency_phrases
    |
    |-- SQLite: checkpoints.db (LangGraph conversation state)
    |
    |-- Static files: /static (index.html, app.js, styles.css)
    |
    |-- SlowAPI rate limiter (in-memory counters)

External Dependencies:
    |-- Google Gemini API (gemini-2.0-flash)
    |-- Slack (optional webhook)
    |-- EHR system (optional FHIR webhook)
```

### Key Design Decisions (and their tradeoffs)

**LangGraph for the conversation state machine**
- Good: Each conversation phase is a node. State is automatically checkpointed. Very easy to add new phases.
- Bad: The graph is compiled once and held as a global. The checkpoint store is SQLite. LangGraph's SQLite checkpoint is not designed for high concurrency.

**FastAPI BackgroundTasks for report generation**
- Good: Report generation (which calls Gemini) doesn't block the user's response. Patient gets "report is generating" immediately.
- Bad: BackgroundTasks run in the same process. If the server crashes, the job is lost and never retried. No persistent queue.

**SlowAPI for rate limiting**
- Good: Very simple to add `@limiter.limit("60/minute")` to any endpoint.
- Bad: Counters are stored in-memory. If you run 3 instances, each has its own counter. A user can hit 60 requests/minute per instance = 180/minute total. Useless for horizontal scaling.

**Single threading lock for database**
- `_db_lock = threading.Lock()` in `sqlite_db.py` wraps every database operation
- Good: Prevents SQLite corruption from concurrent writes
- Bad: Serializes ALL database operations. At 100 concurrent users, every DB read/write queues behind the lock. This will be your first bottleneck.

---

## 4. The Request Lifecycle: What Happens When a Patient Chats

Let's trace a single `POST /chat` request from network packet to response. Understanding this is critical for knowing where things slow down under load.

```
1. Patient types "My chest hurts" and hits Send
   |
   v
2. Browser sends: POST /chat
   body: thread_id=<uuid>, message="My chest hurts", client_msg_id=<uuid>
   |
   v
3. FastAPI receives the request
   |-- SlowAPI checks rate limit (60/minute per IP, in-memory counter)
   |-- Validates: message not empty, not > 1200 chars, client_msg_id not > 128 chars
   |
   v
4. check_prompt_injection(message) — regex scan for injection attempts
   If flagged → return safe response immediately (no LLM call)
   |
   v
5. Idempotency check: db.get_idempotent_response(thread_id, client_msg_id)
   |-- Acquires _db_lock
   |-- Queries SQLite: SELECT FROM idempotency WHERE thread_id=? AND key=?
   |-- Releases _db_lock
   If found → return the cached response (no LLM call). This handles duplicate
   requests (network retry, double-click, browser refresh during submit).
   |
   v
6. Session lookup: db.fetch_one("SELECT thread_id, status FROM sessions WHERE thread_id=?")
   |-- Acquires _db_lock → query → releases lock
   If not found → 404
   |
   v
7. Load previous phase from session_state
   |-- Another DB query with lock
   |
   v
8. graph.invoke({"messages": [{"role": "user", "text": "My chest hurts"}]}, config)
   |
   |  This is where the magic (and most of the time) happens.
   |  LangGraph looks up the checkpoint for this thread_id from checkpoints.db
   |  Determines current phase (e.g., "subjective")
   |  Routes to subjective_node
   |
   |  subjective_node:
   |    a. detect_crisis("My chest hurts") — keyword scan, instant
   |    b. detect_emergency_red_flags("My chest hurts") — keyword scan, instant
   |       "chest" matches emergency phrase → emergency_flag = True
   |    c. If emergency: create_escalation() → DB write → dispatch_emergency_alert()
   |       → Slack webhook (HTTP POST with retry, runs inline here)
   |    d. run_json_step() → GeminiClient.generate_text()
   |       → HTTP call to Google Gemini API (this is the slow part, ~1-3 seconds)
   |       → Parse JSON response → validate against Pydantic schema
   |       → If parse fails: retry with repair prompt (another Gemini call)
   |       → If still fails: use hardcoded fallback
   |    e. validate_llm_response() — scan for diagnosis language
   |    f. classify_intake() → another Gemini call (if first subjective message)
   |    g. score_extraction_quality() — deterministic scoring
   |    h. Returns updated state dict
   |
   |  LangGraph writes updated checkpoint to checkpoints.db
   |  (acquires SQLite WAL lock)
   |
   v
9. Back in api.py: db.save_session_state() → DB write (app.db)
   |
   v
10. Phase check: if phase == "report" → create a background job
    db.create_job() → DB write
    background_tasks.add_task(run_report_job, thread_id, job_id)
    (Job is queued but not started yet — it starts after response is sent)
    |
    v
11. db.save_message(thread_id, "user", message) → DB write
    db.save_message(thread_id, "assistant", reply) → DB write
    db.set_session_status() → DB write
    db.save_idempotent_response() → DB write
    |
    v
12. Return JSON: {"reply": "...", "status": "active", "phase": "subjective"}
    |
    v
13. AFTER RESPONSE: background_tasks execute run_report_job()
    (Only if phase == "report")
    → Another graph.invoke() → More Gemini calls → save_report() → webhook dispatch
```

**Total time for a typical chat turn:** 1.5-4 seconds (dominated by Gemini API call)
**Total DB operations per chat:** ~7-9 (each one acquires and releases the global lock)
**Total Gemini calls per chat turn:** 1-3 (primary + possibly repair + possibly classify)

---

## 5. The AI Brain: LangGraph State Machine

### What is LangGraph?

LangGraph is a framework for building stateful, multi-step AI agents as directed graphs. Think of it like a flowchart where each box is a Python function ("node") and the arrows between boxes are determined by the current conversation state.

### The Phase State Machine

```
START
  |
  v (route() function decides based on current_phase)
  |
  +---> [consent_node] ----+
  |                        |
  +---> [identity_node] <--+
  |         |
  |         v (if needs_identity_review=True)
  +---> [identity_review_node]
  |         |
  |         v (once identity is done)
  +---> [subjective_node]
  |         |
  |         v (when symptom data collected)
  +---> [validate_node] (agentic, non-interactive — runs without waiting for user)
  |         |
  |         v (if validation passes)
  +---> [clinical_history_node]
  |         |
  |         v (when allergies/meds/PMH done)
  +---> [report_node] ---------> END (report saved, job done)
  |
  +---> [handoff_node] ---------> END (safety block: go to clinician)
  |
  +---> [confirm_node] (confirmation step)
  |
  END
```

### How State Persists Between Messages

This is the clever part. Each conversation is identified by a `thread_id` (UUID). LangGraph stores a full snapshot of the `IntakeState` TypedDict in `checkpoints.db` after every node runs.

When the patient sends the next message:
1. `graph.invoke()` reads the checkpoint from SQLite using `thread_id`
2. Merges the new message into the state
3. Routes to the correct node based on `current_phase`
4. Runs the node
5. Saves the updated state back to SQLite

This means if the server restarts mid-conversation, the patient can resume exactly where they left off — because the full state is in the database, not in memory.

**What's in the state (IntakeState)?**
- `current_phase` — where we are in the flow
- `identity` — name, DOB, phone, address collected from patient
- `stored_identity` — what the EHR has on file for this patient
- `chief_complaint` — the main reason for the visit
- `opqrst` — the structured symptom breakdown (6 fields)
- `allergies`, `medications`, `pmh`, `recent_results` — clinical history
- `triage` — AI-assessed risk level, emergency flags, red flags
- `crisis_detected`, `human_review_required`, `safety_score` — safety signals
- `intake_classification` — emergency/routine/specialist/mental_health/pediatric
- `extraction_quality_score` — how completely was OPQRST captured (0.0-1.0)

### The Agentic Capabilities

The system goes beyond a simple scripted chatbot with four "agentic" features:

**1. Intake Classification** — After the patient states their chief complaint, the LLM classifies the visit type. This determines how aggressively to probe for information (ED mode vs. clinic mode) and feeds into the safety score.

**2. Dynamic Follow-up Selection** — Instead of asking the same questions in the same order every time, the LLM selects which follow-up question is most clinically relevant given what it already knows. A patient who mentioned they're diabetic gets a different follow-up than one who mentioned trauma.

**3. Extraction Quality Scoring** — After collecting OPQRST, the system scores how complete the data is (0.0-1.0). If below threshold (0.75 for ED, 0.60 for clinic), it generates a targeted gap-fill question and asks the patient to elaborate. The retry counter increments. This feeds into the safety score.

**4. Validation Gate** — Before transitioning phases, `validate_node` runs non-interactively to check whether required fields are actually populated. It can loop back and ask the patient to complete missing information rather than silently generating an incomplete report.

---

## 6. The Database Layer: What Is Stored and Why

There are two SQLite databases:

### app.db — Application Data

| Table | What It Stores | Why |
|-------|---------------|-----|
| `sessions` | thread_id, status (active/done/escalated), timestamps | Track which sessions exist and their state |
| `messages` | All user and assistant messages per session | Full conversation history for clinician review |
| `reports` | Generated clinical notes, FHIR bundles, risk level | What the doctor actually reads |
| `escalations` | Emergency and crisis escalations with full reason trail | Safety audit trail, nurse resolution workflow |
| `jobs` | Async report generation jobs with status | Decouples report generation from the chat response |
| `session_state` | Compact snapshot of IntakeState per session | Faster than re-reading the checkpoint for API responses |
| `idempotency` | SHA-256 of (thread_id, client_msg_id) → cached response | Prevents duplicate processing if patient re-sends |
| `llm_failure_log` | Every LLM parse failure, repair attempt, fallback | Analytics: which node fails most, error rates |
| `webhook_deliveries` | Every outbound webhook attempt with HTTP status | Audit trail: did the FHIR bundle reach the EHR? |
| `mock_ehr` | Simulated patient records (name, DOB, allergies etc.) | Demo: simulates identity verification against real EHR |
| `emergency_phrases` | Keywords that trigger emergency escalation | Configurable at runtime without code changes |

### checkpoints.db — LangGraph Conversation State

This is LangGraph's internal store. It holds the full serialized `IntakeState` after each node execution, keyed by `(thread_id, checkpoint_id)`. This is what makes conversation resumption possible after server restart.

**The separation exists for a reason:** `app.db` is queried by API endpoints. `checkpoints.db` is queried exclusively by LangGraph. Mixing them would mean LangGraph's internal schema changes break your application schema.

### How Writes Are Serialized

Every write to `app.db` goes through this pattern:

```python
_db_lock = threading.Lock()  # ONE lock for the entire application

def exec_one(q: str, p: tuple = ()) -> None:
    def _exec():
        with _db_lock:      # <-- Acquires global lock
            c = conn()
            c.execute(q, p)
            c.commit()      # Flush to disk immediately
    _retry_db_operation(_exec)  # Retry 3x on "database locked" errors
```

WAL (Write-Ahead Logging) mode is enabled on both databases:
- WAL allows concurrent reads while a write is in progress
- Without WAL, a write would lock out ALL reads
- With WAL, readers see the last committed state while a write happens

Even with WAL, the Python-level lock serializes everything through a single thread. This is intentional to prevent race conditions but is the primary scalability ceiling.

---

## 7. The LLM Layer: Gemini + Reliability Patterns

### The Three-Level Degradation System

Every LLM call in this system goes through `run_json_step()` which implements three levels of graceful degradation:

```
Level 1: PRIMARY CALL
  → Send prompt to Gemini
  → Parse JSON from response
  → Validate against Pydantic schema
  → If all good: use the result
  |
  If parse/validation fails but LLM responded (garbage JSON):
  |
  v
Level 2: REPAIR CALL
  → Send a corrective prompt: "Your output was invalid. Here's the error.
     Here are the required keys. Return ONLY a JSON object."
  → Try parse/validate again
  → If good: use repaired result
  |
  If still fails (or LLM returned an error):
  |
  v
Level 3: HARDCODED FALLBACK
  → Use a safe default dict defined per-node
  → Log the failure to llm_failure_log
  → Session continues (patient is never stuck due to LLM failure)
```

### The Circuit Breaker

The `CircuitBreaker` class in `llm.py` prevents cascading failures when Gemini is down:

```
CLOSED (normal)
  → Every request passes through to Gemini
  → If 5 consecutive failures: transition to OPEN

OPEN (Gemini is down)
  → All requests fail immediately WITHOUT calling Gemini
  → After 60 seconds: transition to HALF_OPEN

HALF_OPEN (testing recovery)
  → ONE request is allowed through
  → If success: back to CLOSED
  → If failure: back to OPEN, reset 60-second timer
```

**Why this matters for production:** Without a circuit breaker, if Gemini is down:
- Every request tries to call Gemini
- Each call times out after 15 seconds (the LLM timeout)
- 100 concurrent users = 100 threads all blocked for 15 seconds
- Server runs out of threads, starts refusing new connections
- Your patients can't complete intake even if the rest of the system is fine

With the circuit breaker, after 5 failures the system immediately serves the fallback response to all subsequent requests until Gemini recovers.

### The LLM Timeout Problem

```python
def _call() -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(self.client.models.generate_content, ...)
        resp = future.result(timeout=timeout)  # timeout=15 seconds
```

The Google Gemini SDK ignores HTTP timeouts, so this wraps every call in a `ThreadPoolExecutor` to enforce a hard timeout. The problem: every LLM call creates a new thread. Under 100 concurrent users, each making a call that takes 3 seconds, you have ~300 threads created and destroyed per second. This is expensive and will degrade performance.

### Retry Strategy: Full Jitter

When a transient error occurs, the retry delay is:
```
sleep = random.uniform(0, min(cap, base * 2^attempt))
```

This is "full jitter" — the delay is random between 0 and the ceiling. Why random? If 50 clients all fail at the same moment and all retry at exactly `base * 2^attempt` seconds, they all hammer Gemini at the same time again. Randomizing the retry window spreads the load.

---

## 8. Safety and Guardrails

### Layer 1: Input Validation (instant, before any LLM call)

- Message length cap: 1200 characters
- Prompt injection detection: regex scan for patterns like "ignore previous instructions", "system prompt", "act as"
- If injection detected: safe canned response, no LLM call, no escalation (doesn't reward the attempt)

### Layer 2: Crisis and Emergency Detection (instant, keyword-based)

Runs inside every conversation node before the LLM call:

- `detect_crisis()` — scans for self-harm and suicidal language using phrase list from `emergency_phrases` table
- `detect_emergency_red_flags()` — scans for medical emergency phrases ("chest pain", "can't breathe", "stroke")

If triggered:
1. Log the event
2. Write to `escalations` table with full reason trail
3. Dispatch Slack alert (synchronously in the request path)
4. Display resource message to patient (crisis line, emergency instruction)
5. Set `crisis_detected=True` or `emergency_flag=True` in state (feeds into report)

These are deliberately synchronous and rule-based (not LLM-based) because:
- Speed: the patient needs the crisis response NOW, not in 3 seconds
- Reliability: keyword detection never has an API outage

### Layer 3: LLM Output Filtering

`validate_llm_response()` scans every LLM-generated message before it's sent to the patient:
- Blocks diagnosis language: "you have", "you likely have", "consistent with", "sounds like a case of"
- Replaces with: "I've noted your symptoms. The clinician will review everything when they see you."

This prevents the AI from making medical diagnoses, which is a regulatory and liability issue.

### Layer 4: Pre-Report Safety Check (SafetyChecker)

Before generating the clinical report, `SafetyChecker.compute()` runs a weighted scoring rubric:

**Hard blocks** (these alone prevent report generation):
- Chief complaint missing: +35 points
- Patient name missing: +30 points
- Clinical history incomplete (allergies/meds/PMH not collected): +25 points

**Soft review flags** (these raise the score but don't block alone):
- Emergency flag active: +50 points
- Crisis detected: +40 points
- Identity unverified: +20 points
- Identity mismatch: +15 points
- Extraction quality below threshold: +20 points
- Extraction was retried: +10 points
- ED mode baseline: +10 points (all ER visits are inherently higher risk)

**Threshold:** If score >= 50, the report is flagged `pending_review=True` (a clinician should double-check). If hard blocks fire, report generation is prevented entirely and the session goes to `handoff_node`.

This is scored separately from the LLM — it's deterministic Python code that always runs the same way, making it auditable and testable.

---

## 9. Clinician-Facing Features

### Authentication

A single shared clinician password via `POST /clinician/token`. Returns a JWT that expires in 24 hours. The JWT is verified with HS256 using `JWT_SECRET` from the environment.

**Current limitation:** There is only ONE password for ALL clinicians. There is no per-clinician identity, no RBAC (role-based access control), no audit trail of which clinician viewed which patient.

### Clinician Endpoints

- `GET /clinician/pending` — all unresolved escalations
- `GET /clinician/case/{thread_id}` — full case: messages, report, escalations, safety summary
- `POST /clinician/resolve` — mark an escalation resolved with a nurse note
- `GET /report/{thread_id}/fhir` — FHIR R4 Bundle (machine-readable, for EHR import)
- `GET /clinician/webhooks` — audit log of all outbound webhook deliveries
- `GET /analytics` — operational metrics for the last 7 days

### The FHIR Bundle

FHIR (Fast Healthcare Interoperability Resources) is the international standard for healthcare data exchange. When an intake completes, the system generates a FHIR R4 Bundle containing:
- `Patient` resource — identity
- `Condition` resource — chief complaint and OPQRST
- `AllergyIntolerance` resource — one per allergy
- `MedicationStatement` resource — one per medication
- `Observation` resource — triage risk level

This bundle is:
1. Stored in the `reports` table as JSON
2. Sent via HMAC-signed POST to the configured EHR endpoint
3. Available via `GET /report/{thread_id}/fhir` for direct EHR pull

### Async Report Generation

When the conversation ends (phase transitions to "report"), the API does NOT block waiting for the report to generate. Instead:
1. A `job` record is created in the `jobs` table with status `queued`
2. The job is handed to FastAPI's `BackgroundTasks`
3. The patient gets a response immediately: `{"phase": "report_generating", "job_id": "..."}`
4. The frontend polls `GET /jobs/{job_id}` every 2-3 seconds
5. When the job is `done`, the frontend fetches `GET /report/{thread_id}`

This matters because report generation involves multiple Gemini calls (report text, FHIR bundle construction) and could take 5-15 seconds. Blocking the patient's response for that long is unacceptable UX.

---

## 10. What Will Break in Production

This section is the most important. Here is every bottleneck and failure mode, ordered by how quickly they will be hit.

---

### CRITICAL: SQLite Cannot Scale Horizontally

**What it is:** SQLite is a file-based database. It lives on the filesystem of the server running the application. It is not a server — it is a library that reads and writes to a file.

**Why it breaks:** If you run two instances of this application (for load balancing or redundancy), both try to write to the same SQLite file. Two containers cannot share a SQLite file safely — you need a network filesystem (NFS, EFS), which adds latency and doesn't solve the fundamental concurrency problem.

Even on a single instance, the Python-level `_db_lock` serializes all database operations through a single thread. At 50 concurrent users, you will see lock contention. At 100, you will see `sqlite3.OperationalError: database is locked` errors (the retry logic delays but doesn't fully prevent this).

**When it breaks:** At roughly 30-50 concurrent users for the checkpoint database, 80-100 for app.db.

**The fix:** Replace SQLite with PostgreSQL for both databases. LangGraph supports PostgreSQL checkpointers (`langgraph-checkpoint-postgres`). This single change unlocks horizontal scaling.

---

### CRITICAL: In-Memory Rate Limiter Breaks on Multiple Instances

**What it is:** SlowAPI stores rate limit counters in the application's memory.

**Why it breaks:** If you run 3 instances behind a load balancer:
- Instance 1 sees 40 requests from patient A
- Instance 2 sees 30 requests from patient A
- Instance 3 sees 20 requests from patient A
- None of them individually hit the limit of 60/minute
- Patient A has made 90 requests — 50% over the limit — and was never throttled

**When it breaks:** The moment you add a second instance, which you will need for redundancy.

**The fix:** Use Redis as the backend for SlowAPI. `slowapi` supports `RedisStore`. All instances share one counter per IP.

---

### CRITICAL: Background Tasks Have No Persistence

**What it is:** FastAPI's `BackgroundTasks` run the report generation job in a thread in the same process.

**Why it breaks:** If the server crashes (or is killed for a deployment) while a report job is running:
- The `jobs` table has a row with status `queued` or `running`
- The actual background thread is dead
- The job will never complete
- The patient's report is never generated
- The clinician will never see it

The code has a partial mitigation: `mark_stale_jobs_failed()` marks jobs that have been `running` for more than 10 minutes as `failed`. This at least makes the failure visible, and clinicians can manually retry. But the report is still lost unless manually retried.

**When it breaks:** Any time you do a rolling deployment or the server crashes under load.

**The fix:** Use a real job queue. Celery with Redis as the broker is the standard Python solution. The job is written to Redis, picked up by a separate Celery worker process, and completes independently of the web server. If the worker crashes, the job is re-queued automatically.

---

### CRITICAL: Single LLM Client and API Key

**What it is:** One `GeminiClient` singleton with one API key. No fallback LLM.

**Why it breaks:**
- Google Gemini has rate limits (requests per minute and tokens per minute per project/key)
- If you hit the rate limit, ALL conversations fail simultaneously
- There is no fallback to GPT-4, Claude, or another model
- The circuit breaker helps by failing fast, but the patient experience is still broken

**When it breaks:** At medium-to-large scale. At 100 patients/day with 20 LLM calls each = 2,000 LLM calls/day. Fine. At 5,000 patients/day = 100,000 calls/day. Very dependent on your API quota tier.

**The fix:**
- Multi-key rotation: rotate between N API keys, round-robin
- Provider fallback: if Gemini returns 429 or error, try OpenAI or Claude
- Semantic caching: similar questions get similar answers; cache responses by embedding similarity (using Redis + a vector store)

---

### HIGH: Webhook Retries Block Background Threads

**What it is:** In `webhook.py`, the retry logic uses `time.sleep()`:

```python
if attempt > 1:
    delay = _RETRY_DELAYS[min(attempt - 2, len(_RETRY_DELAYS) - 1)]
    time.sleep(delay)  # blocks the thread for 2s, 8s, or 30s
```

**Why it breaks:** This runs in the FastAPI background task thread. A single failing webhook call with 3 retries blocks a thread for up to `2 + 8 + 30 = 40 seconds`. Under load, with multiple failing webhook deliveries, you can exhaust the server's thread pool.

**When it breaks:** When the EHR webhook endpoint is slow or down (the most likely time you need it most).

**The fix:** Move webhook delivery to the Celery job queue alongside report generation. The thread is freed immediately and the retry logic runs in a dedicated worker.

---

### HIGH: No Health Check Endpoint

**What it is:** Load balancers (AWS ALB, nginx, Kubernetes) need to know if an instance is healthy before sending it traffic. They send periodic requests to a health check URL.

**Why it breaks:** Without a `/health` endpoint:
- A dead instance keeps receiving traffic
- The load balancer can't automatically route around failures
- You can't do safe rolling deployments (bring up new version, verify it's healthy, then drain old version)

**The fix:** Add:
```python
@app.get("/health")
def health():
    # Check DB connection, check LLM client initialized
    return {"status": "ok"}
```

---

### HIGH: CORS Wildcard

**What it is:** In `settings.py`:
```python
cors_allowed_origins: list[str] = ["*"]
```

This allows ANY website on the internet to make requests to your API from a browser.

**Why it breaks:** This is not a crash issue — it's a security issue. A malicious website could embed your chat in their page and make requests on behalf of visiting users. In a healthcare context, this could be used to phish patients.

**The fix:** Set `CORS_ALLOWED_ORIGINS=["https://yourhospital.com"]` in production.

---

### MEDIUM: Unauthenticated Session Resume

**What it is:** `GET /resume/{thread_id}` requires only knowing the thread_id (a UUID).

**Why it breaks:** A UUID is not secret. If a thread_id is shared (e.g., in a URL bar visible to someone else at a kiosk), anyone who knows it can read the patient's session. There is no second factor — no patient authentication.

**When it breaks:** It's a privacy/HIPAA risk that exists from day 1.

**The fix:** Issue a short-lived, single-use resume token that is sent to the patient's phone via SMS/email, separate from the thread_id. The resume endpoint requires both the thread_id and the token.

---

### MEDIUM: No Database Migrations

**What it is:** The code does schema upgrades via manual `ALTER TABLE` checks in Python:
```python
if "fhir_bundle" not in existing:
    c.execute("ALTER TABLE reports ADD COLUMN fhir_bundle TEXT")
```

**Why it breaks:** This works for adding columns but:
- Cannot rename columns safely
- Cannot change column types
- No version tracking (how do you know which migrations have run?)
- If the startup migration fails, the server fails to start

**The fix:** Use Alembic (the standard Python migration tool). Every schema change is a numbered migration file. Alembic tracks which ones have been applied.

---

### MEDIUM: No Observability

**What it is:** Logs are `logging.basicConfig(level=logging.INFO, format="%(message)s")` — plain text to stdout. Events are structured JSON via `log_event()` but they go to stdout and are never aggregated.

**Why it breaks:** In production, you cannot SSH into containers and tail logs. Containers are ephemeral. You need:
- A log aggregation system (Datadog, Elasticsearch, CloudWatch)
- Metrics (request rate, error rate, latency percentiles, LLM call latency)
- Alerting (if error rate > 5% for 2 minutes, page someone)
- Distributed tracing (which Gemini call slowed this specific patient's session?)

**When it breaks:** The first time something goes wrong at 2am and you have no idea why.

---

### LOW: Single Clinician Password (No RBAC)

**What it is:** All clinicians share one password. There is no audit trail of which clinician accessed which patient.

**Why it breaks (compliance):** HIPAA requires access controls and audit logs. If there is a data breach, you cannot determine who accessed what.

**The fix:** Individual clinician accounts, stored in a `users` table (hashed passwords with bcrypt). JWT tokens include the `sub` (subject) field identifying the clinician. Every sensitive action logs `(clinician_id, action, thread_id, timestamp)`.

---

### LOW: Static Files Served by FastAPI

**What it is:** `app.mount("/static", StaticFiles(directory=...))` serves HTML, CSS, and JavaScript from the application server.

**Why it breaks:** This puts static file serving load on your application server. At scale, your Gemini-calling, state-machine-running Python process is also sending 200KB of JavaScript to every browser. It's wasteful and adds latency.

**The fix:** Put a CDN (CloudFront, Cloudflare) in front of static assets. The app server only handles API requests.

---

## 11. Scale Tiers: Small to Huge

### Tier 1: Small Clinic (30-80 patients/day, <10 concurrent)

**Current system works as-is, with config hardening.**

Constraints still holding:
- SQLite fine for this load (WAL mode + single instance)
- Single uvicorn process with async handlers is sufficient
- FastAPI BackgroundTasks work fine at low volume

Minimum production hardening:
- Change default JWT_SECRET and CLINICIAN_PASSWORD (currently breaks at startup — good)
- Set CORS_ALLOWED_ORIGINS to your domain
- Enable HTTPS (TLS termination at nginx or load balancer)
- Mount `/app/data` to a persistent volume (not container ephemeral storage)
- Set up daily SQLite backup to S3: `sqlite3 app.db ".backup 'backup.db'"`
- Add `/health` endpoint
- Set up log forwarding to CloudWatch or similar

Estimated infrastructure cost: $30-80/month (one t3.small or equivalent)

---

### Tier 2: Medium Hospital (300-800 patients/day, 50-150 concurrent)

**Must replace SQLite. Must add Redis. Must separate background jobs.**

The specific breaking points at this tier:
- SQLite lock contention visible above ~50 concurrent → replace with PostgreSQL
- Rate limiter ineffective at >1 instance → Redis-backed SlowAPI
- Background tasks lost on deploy → Celery workers
- LLM API rate limits may be hit → API key rotation

Architecture changes:
```
[Load Balancer (nginx or AWS ALB)]
          |
    +-----+-----+
    |             |
[App Instance 1] [App Instance 2]
    |             |
    +------+------+
           |
    [PostgreSQL]   [Redis]
                       |
               [Celery Workers (2-4)]
                       |
               [PostgreSQL (same DB, separate connection pool)]
```

Infrastructure:
- 2 app instances (t3.medium, 2 vCPU, 4GB RAM each)
- 1 PostgreSQL instance (db.t3.medium, or Amazon RDS)
- 1 Redis instance (cache.t3.micro, or ElastiCache)
- 2 Celery worker instances (t3.small each)
- nginx as load balancer (or AWS ALB)

Estimated cost: $300-600/month

---

### Tier 3: Large Hospital Network (5,000-20,000 patients/day, 500-2000 concurrent)

**Full distributed system. Kubernetes. Read replicas. Caching. Multiple LLM providers.**

At this scale, every component needs to handle failure independently:

```
[CloudFront CDN] ─── static assets (JS/CSS/HTML)
        |
[AWS ALB or nginx cluster]
        |
[Kubernetes cluster]
        |── [App Pods x10] ─────────────────── API servers
        |── [Celery Worker Pods x20] ─────────── background jobs
        |── [Redis Cluster] ──────────────────── rate limiting, caching, job queue
        |
[RDS PostgreSQL Primary] ─── writes
        |
[RDS PostgreSQL Read Replica x2] ── read-heavy queries (analytics, report reads)
        |
[Gemini API]  [OpenAI API]  [Claude API]  ── LLM providers with fallback
        |
[Datadog / Grafana] ──────────────────────── observability
        |
[PagerDuty] ──────────────────────────────── alerting
```

Additional components needed:
- **LLM Gateway** (LiteLLM or custom): routes LLM calls, implements load balancing across API keys and provider fallback
- **Semantic Cache** (Redis + embedding model): cache LLM responses for near-identical questions, dramatically reducing API calls and cost
- **Kubernetes HPA** (Horizontal Pod Autoscaler): automatically scales app pods based on CPU/memory/request rate
- **Database connection pooling** (PgBouncer or pgpool): PostgreSQL has a max connection limit; connection poolers multiplex thousands of app connections into a small pool
- **Secrets management** (AWS Secrets Manager or HashiCorp Vault): API keys, JWT secrets, DB passwords — NOT in .env files in containers
- **Queue monitoring** (Flower for Celery): see how many jobs are queued, which are failing, worker throughput

Estimated cost: $3,000-15,000/month (depends heavily on LLM API costs)

---

## 12. Production Changes Required

### Change 1: Replace SQLite with PostgreSQL

**Files to change:** `graph.py`, `sqlite_db.py`, `settings.py`

In `graph.py`:
```python
# Current (SQLite):
cp = sqlite3.connect(settings.checkpoint_db_path, ...)
checkpointer = SqliteSaver(cp)

# Production (PostgreSQL):
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
conn = psycopg.connect(settings.database_url)
checkpointer = PostgresSaver(conn)
```

The `sqlite_db.py` module needs to be replaced with SQLAlchemy (the standard Python ORM) targeting PostgreSQL. This eliminates the `_db_lock` entirely — PostgreSQL handles concurrent access natively with MVCC (Multi-Version Concurrency Control).

**What MVCC means:** In PostgreSQL, readers never block writers and writers never block readers. Each transaction sees a snapshot of the database at the time it started. This is fundamentally different from SQLite's writer-locks-everything model.

---

### Change 2: Redis for Rate Limiting

**File to change:** `api.py`

```python
# Current:
limiter = Limiter(key_func=get_remote_address)

# Production:
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis

redis_client = redis.from_url(settings.redis_url)
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
)
```

All instances now share the same counter. One Redis instance handles millions of rate limit checks per second with sub-millisecond latency.

---

### Change 3: Celery for Background Jobs

**New files:** `celery_app.py`, `tasks.py`

```python
# celery_app.py
from celery import Celery
app = Celery("intake", broker=settings.redis_url, backend=settings.redis_url)

# tasks.py
from .celery_app import app as celery_app

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_report_task(self, thread_id: str, job_id: str):
    try:
        run_report_job(thread_id, job_id)
    except Exception as exc:
        raise self.retry(exc=exc)
```

In `api.py`, replace:
```python
# Current:
background_tasks.add_task(run_report_job, thread_id, job_id)

# Production:
from .tasks import generate_report_task
generate_report_task.delay(thread_id, job_id)
```

Now if the web server dies, the job is still in Redis. A Celery worker will pick it up. If the worker dies mid-job, Celery re-queues it (configurable with `acks_late=True`).

---

### Change 4: Health Check Endpoints

**File to change:** `api.py`

```python
@app.get("/health")
async def health():
    """Kubernetes/load balancer liveness probe."""
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Kubernetes readiness probe — fails if DB or LLM is not reachable."""
    try:
        # Check DB
        db.fetch_one("SELECT 1")
        # Check LLM (use circuit breaker state, don't make a real call)
        from .llm import _breaker
        if _breaker.state == "open":
            raise RuntimeError("LLM circuit breaker is open")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(503, detail=str(e))
```

`/health` is the liveness probe — if this fails, Kubernetes restarts the pod.
`/ready` is the readiness probe — if this fails, the load balancer stops sending traffic to this pod (but doesn't restart it).

---

### Change 5: LLM Provider Abstraction and Key Rotation

**File to change:** `llm.py`

The `GeminiClient` should become a `LLMClient` interface with multiple implementations. LiteLLM is an open-source library that provides a unified API across 100+ LLM providers:

```python
import litellm

class LLMGateway:
    def __init__(self, providers: list[str], api_keys: dict):
        self.providers = providers  # ["gemini/gemini-2.0-flash", "gpt-4o", "claude-3-5-sonnet"]
        self.keys = api_keys
        self._current = 0

    def generate(self, *, system: str, prompt: str, **kwargs) -> LLMResult:
        for provider in self._get_provider_order():
            try:
                response = litellm.completion(
                    model=provider,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    **kwargs
                )
                return LLMResult(ok=True, text=response.choices[0].message.content)
            except litellm.RateLimitError:
                continue  # try next provider
            except Exception as e:
                continue
        return LLMResult(ok=False, text="", error="all_providers_exhausted")
```

---

### Change 6: Structured Logging and Metrics

**File to change:** `logging_utils.py`

```python
# Current: prints JSON to stdout
# Production: send to log aggregator + increment Prometheus metrics

import structlog
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("intake_requests_total", "Total requests", ["endpoint", "status"])
LLM_LATENCY = Histogram("llm_call_duration_seconds", "LLM call latency", ["provider", "node"])
ACTIVE_SESSIONS = Gauge("intake_active_sessions", "Currently active intake sessions")

logger = structlog.get_logger()

def log_event(event: str, level: str = "info", **kwargs):
    log_func = getattr(logger, level, logger.info)
    log_func(event, **kwargs)
    # Also increment relevant counter
    if event == "chat_done":
        REQUEST_COUNT.labels(endpoint="/chat", status="success").inc()
    elif event == "chat_error":
        REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
```

Add a Prometheus metrics endpoint:
```python
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

### Change 7: Database Migrations with Alembic ✅ IMPLEMENTED

Alembic is installed and wired up. A baseline migration (`alembic/versions/001_baseline_schema.py`) captures all 14 application tables as they existed before version tracking was added:

```python
# alembic/versions/001_baseline_schema.py
def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS sessions (
        thread_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT, updated_at TEXT, session_token_hash TEXT
    )""")
    # ... all 14 tables with IF NOT EXISTS for zero-downtime first run

def downgrade():
    op.execute("DROP TABLE IF EXISTS sessions")
    # ... reverse dependency order
```

The CI pipeline runs `alembic upgrade head` as the first step before starting the app. Every future schema change is a versioned migration file. The `alembic_version` table tracks exactly which migrations have been applied, so it's impossible for app code and database schema to drift out of sync between environments.

**Why Alembic over the previous `CREATE TABLE IF NOT EXISTS` approach:** The old approach had no version tracking. `ALTER TABLE` changes were applied by hand-written Python checks at startup. When a column was added in one branch but not another, the discrepancy was invisible until a query failed at runtime on a production server. Alembic makes the schema history an explicit, auditable, reversible record.

**Existing deployments:** Run `alembic stamp 001` once to mark the existing database as already at baseline without running migrations against it.

---

### Change 8: Secrets Management

Remove all secret defaults from `settings.py`. In production:
- JWT_SECRET → AWS Secrets Manager (rotated every 90 days)
- CLINICIAN_PASSWORD → replaced with proper user auth (see Change 9)
- GEMINI_API_KEY → Secrets Manager, injected at container startup
- Database password → IAM database authentication (no static password)

---

### Change 9: Proper Clinician Authentication

Replace the single shared password with individual accounts:

```sql
CREATE TABLE clinician_users (
    user_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'nurse',  -- 'nurse' | 'doctor' | 'admin'
    pw_hash     TEXT NOT NULL,                  -- bcrypt
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

CREATE TABLE clinician_audit_log (
    log_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES clinician_users(user_id),
    action      TEXT NOT NULL,     -- 'view_case' | 'resolve_escalation' | 'view_report'
    thread_id   UUID,
    ip_address  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

JWT tokens now include `{"sub": "user_uuid", "role": "nurse", "exp": ...}`. The `require_clinician` dependency extracts the user identity and logs every access.

---

## 13. Observability: Knowing When Things Go Wrong

### The Three Pillars of Observability

**Logs** — What happened, and when, with context
**Metrics** — How often is it happening, how fast, are rates changing
**Traces** — For a specific request, which functions ran, how long did each take

### Key Metrics to Track

| Metric | Alert Threshold | Why |
|--------|----------------|-----|
| `llm_call_duration_p99` > 8s | Page on-call | Patients are waiting too long |
| `llm_error_rate` > 10% | Alert | Gemini is degraded |
| `circuit_breaker_state = open` | Page on-call | Gemini is down |
| `report_job_failure_rate` > 5% | Alert | Reports not reaching clinicians |
| `db_query_duration_p99` > 500ms | Alert | Database is slow (connection pool exhausted?) |
| `active_sessions` spike | Alert | Possible DDoS or load test |
| `emergency_escalations_per_hour` spike | Alert | Unusual patient volume or test data contamination |
| `webhook_exhausted_count` > 0 | Alert | FHIR bundles not reaching EHR |

### Dashboard Panels

1. **Patient Flow** — sessions started vs. completed per hour. Drop in completion = something broke mid-flow.
2. **LLM Health** — calls/min, error rate, latency p50/p95/p99, circuit breaker state
3. **Escalation Volume** — emergency and crisis escalations over time. A spike could mean a real incident or a data quality problem.
4. **Report Job Status** — queued/running/done/failed over time
5. **Webhook Delivery Status** — success rate to EHR and Slack. Failed deliveries mean clinicians miss data.

### Distributed Tracing

In a multi-service architecture (web server + Celery workers), you need to trace a request across process boundaries. OpenTelemetry is the standard:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("intake.api")

@app.post("/chat")
async def chat(...):
    with tracer.start_as_current_span("chat.request") as span:
        span.set_attribute("thread_id", thread_id)
        span.set_attribute("phase", current_phase)
        # ...
        with tracer.start_as_current_span("llm.gemini_call"):
            result = get_gemini().generate_text(...)
```

Now in Jaeger or Datadog APM, you can click on any slow request and see exactly which Gemini call took 4 seconds, which DB query took 200ms, etc.

---

## 14. Security Gaps and Fixes

### HIPAA Compliance Considerations

This system handles Protected Health Information (PHI) — names, dates of birth, medical conditions, medications. Under HIPAA:

1. **Encryption at rest** — All databases and storage must be encrypted. AWS RDS enables this with a checkbox. SQLite files on disk must be encrypted (use dm-crypt/LUKS, or encrypted EBS volumes).

2. **Encryption in transit** — All HTTP must be HTTPS (TLS 1.2+). This applies to: patient-to-server, server-to-Gemini, server-to-Slack, server-to-EHR. The current code sends data to Gemini over HTTPS (handled by the SDK). Slack webhooks and FHIR webhooks are HTTPS. The patient-facing server must terminate TLS.

3. **Minimum necessary access** — The patient endpoint should not be able to access the clinician database tables. Currently, everything runs in the same process with the same DB connection. At scale, consider a separate read-only DB user for patient-facing endpoints.

4. **Audit logging** — Every access to PHI must be logged with who, what, when, from where. Currently only the application-level `log_event()` exists. Need clinician action logging (Change 9).

5. **BAA with Gemini** — Google offers a Business Associate Agreement for Vertex AI (not the standard Gemini API). For a HIPAA-covered entity, patient data MUST NOT be sent to an API that doesn't have a signed BAA. This is a critical compliance item — the current code uses the standard Gemini API, which may not qualify. Evaluate: Google Vertex AI, AWS Bedrock (Claude), Azure OpenAI — all offer HIPAA BAAs.

6. **Data retention** — HIPAA requires retaining medical records for at least 6 years. There is currently no data retention policy or archival mechanism. Old sessions should be archived to cold storage (S3 Glacier) and purged from the database after a defined retention period.

### Current Security Wins (what the code does right)

- HMAC-SHA256 signing on FHIR webhooks
- JWT for clinician auth with expiry
- Idempotency keys prevent duplicate processing
- Input length limits prevent buffer-type attacks
- Prompt injection detection (regex-based)
- Diagnosis language filtering
- Password validation at startup (refuses to start with defaults)
- Rate limiting on all patient-facing endpoints, including `/resume`
- `bandit` static analysis runs in CI to catch common Python security issues (hardcoded secrets, subprocess injection, use of `eval`, etc.)
- `pip-audit` dependency scanning runs in CI to catch known CVEs in the dependency tree before they reach production
- PHI masking in `logging_utils.py` recursively traverses nested dicts and lists so medication names in structured log payloads are masked — not just top-level identity fields
- Session-state snapshot uses an exclusion-list approach rather than a whitelist, so new `IntakeState` fields persist automatically without a developer remembering to add them to a list

**Why static analysis in CI rather than just code review:** A reviewer might not catch that `subprocess.run(user_input)` is a shell injection or that a newly added dependency has a CVE from three months ago. Automated tools catch these consistently every push, before code is merged. In a system that handles PHI, the cost of a missed vulnerability is far higher than the cost of a CI step that takes 10 seconds.

---

## 15. The Target Production Architecture

### For 500-2,000 Concurrent Users (Medium Hospital)

```
                    PATIENTS                    CLINICIANS
                       |                            |
              [HTTPS / TLS termination]
                       |
              [AWS ALB Load Balancer]
              /health checks every 30s
                    |        |
         [App Pod 1]          [App Pod 2]        (Kubernetes, auto-scaled)
         FastAPI +             FastAPI +
         uvicorn               uvicorn
              |                    |
              +--------+-----------+
                       |
              [Redis Cluster]
              - Rate limit counters (SlowAPI)
              - Job queue (Celery broker)
              - Session cache (optional)
                       |
              [Celery Workers x4]
              - Report generation jobs
              - Webhook delivery jobs
              - FHIR bundle construction
                       |
              [PostgreSQL Primary]  ←────── writes
                       |
              [PostgreSQL Replica]  ←────── reads (analytics, report queries)

External Services:
    [Vertex AI / Gemini] ←── LLM calls (HIPAA BAA signed)
    [Slack]              ←── emergency/crisis alerts
    [EHR System]         ←── FHIR bundle delivery

Observability:
    [Datadog / Grafana]  ←── metrics, logs, traces, alerting
    [PagerDuty]          ←── on-call alerting

CI/CD:
    [GitHub Actions]     ←── test, build Docker image, push to ECR
    [ArgoCD]             ←── deploy to Kubernetes, run Alembic migrations
```

### The Migration Path (in order)

If you're taking this from the current state to production, do it in this order — each step is independently deployable and adds value without requiring the next:

1. **Harden configuration** (1 day) — Set real secrets, CORS origins, HTTPS, persistent volume for SQLite. Ship to production safely for small clinics.

2. **Add `/health` and `/ready` endpoints** (2 hours) — Enables proper load balancer integration and Kubernetes.

3. **Add structured logging + basic Prometheus metrics** (1 day) — Observability before you need it.

4. **Replace SQLite with PostgreSQL** (3-5 days) — Biggest change. Unblocks everything else. Test thoroughly.

5. **Add Redis + fix rate limiting** (1 day) — Enable second instance immediately after.

6. **Replace BackgroundTasks with Celery** (2-3 days) — Durable job processing.

7. **Add Alembic** (1 day) — Safe schema evolution.

8. **Implement per-clinician auth + audit logging** (3-5 days) — HIPAA compliance.

9. **LLM provider abstraction + key rotation** (2-3 days) — Reliability at scale.

10. **Move to Kubernetes** (1-2 weeks) — Auto-scaling, self-healing, rolling deployments.

---

## Key Numbers to Internalize

| Metric | Current Value | Why It Matters |
|--------|--------------|----------------|
| LLM timeout | 15 seconds | Maximum time a patient waits per message |
| Rate limit: /chat | 60/minute per IP | ~1 message per second per patient (fine) |
| Rate limit: /start | 10/hour per IP | Prevents session farming |
| Max message length | 1,200 characters | Prevents token abuse |
| Max retries (LLM) | 3 attempts | With full jitter backoff |
| Circuit breaker threshold | 5 failures | Then 60 seconds recovery |
| Report job stale timeout | 10 minutes | Job marked failed if stuck |
| Max report retries | 3 (1 original + 2) | Clinician-only retry |
| Webhook retry delays | 2s, 8s, 30s | EHR must be eventually consistent |
| JWT expiry | 24 hours | Clinician re-auth daily |
| Safety score threshold | 50 points | Above = human review required |
| Emergency escalation | Real-time | No buffering, no batching |

---

*This document reflects the system as of April 2026. Change 7 (Alembic) is now implemented. The architecture described in Section 15 is a target state requiring the remaining changes in Section 12. The current codebase is production-ready for small clinics with configuration hardening; medium-to-large deployments require database and job queue changes before going live.*
