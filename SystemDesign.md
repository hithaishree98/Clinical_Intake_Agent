# Architecture and Design Decisions

## High Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP (REST, FormData)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Rate Limiter│  │  JWT Auth    │  │ Idempotency Layer  │ │
│  │  (slowapi)  │  │  (clinician) │  │  (thread+msg_id)   │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│                    Background Task Queue                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                     │
│                                                              │
│  route() ──► consent_node                                    │
│          ──► identity_node                                   │
│          ──► identity_review_node                            │
│          ──► subjective_node ──► [red flag → handoff_node]   │
│          ──► clinical_history_node                           │
│          ──► confirm_node                                    │
│          ──► report_node                                     │
│          ──► handoff_node (emergency terminal)               │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌─────────────────────────────────────┐
│   Gemini 2.0     │   │          SQLite (WAL mode)           │
│   Flash (LLM)    │   │                                      │
│                  │   │  app.db            checkpoints.db    │
│  extract.py      │   │  ├─ sessions       └─ LangGraph      │
│  (regex only,    │   │  ├─ messages           snapshots     │
│   no LLM)        │   │  ├─ reports                         │
│                  │   │  ├─ escalations                      │
│  fhir_builder.py │   │  ├─ jobs                            │
│  (pure Python,   │   │  ├─ mock_ehr                        │
│   no LLM)        │   │  ├─ idempotency                     │
└──────────────────┘   │  ├─ session_state                   │
                       │  └─ emergency_phrases               │
                       └─────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   Outbound Integrations      │
│   Slack · FHIR webhook       │
└──────────────────────────────┘
```

## Why LangGraph here?

I had used LangChain in a previous RAG project and wanted to explore something that could handle more complex workflows, that's what led me to LangGraph.

But before just picking it up, I was confused as to what makes it different from LangChain or even plain if/else, and whether it was genuinely the right fit for this project or just a more complicated way to do the same thing.

Here's what I figured out.

LangChain is built for workflows that run start to finish without stopping. You give it a task, it chains multiple LLM steps together and returns a result. No waiting for user input in the middle.

That works well for something like this: user uploads a document → LLM summarises it → LLM extracts key points → LLM formats a report → done. The whole thing runs start to finish automatically, no human input needed in the middle.

If/else can handle simple conversations but falls apart the moment we have multiple stages. We need to manually track which phase the patient is in on every single message, build a decision tree with 7 branches, and figure out what happens when the server restarts mid-intake, which without state persistence means the patient loses everything and starts over. Letting someone go back and correct a previous answer would mean writing all that backtracking logic yourself.

LangGraph is specifically helpful here - multi-stage conversations where you pause after each stage, wait for the human, and resume exactly where you left off. Each phase is its own isolated node. State is checkpointed to the database automatically after every step. If the connection drops after the allergy phase, the patient resumes at medications, not the beginning.

For a clinical intake that's collecting allergy and medication data, that reliability isn't optional. If a patient completes the allergy phase and the server crashes before the next step saves, those allergies cannot be lost. That's not a convenience argument, it's a patient safety one.

One thing LangGraph makes easy to enforce is that the LLM has no control over the flow. It extracts information inside a phase but the decision of what phase comes next, when a phase is complete, and when to escalate is all deterministic code. That separation is important in a clinical context where you can't have a model deciding when enough has been collected.

## System design concepts in this project

**Circuit Breaker over Exponential Backoff**

I had used exponential backoff for LLM call failures before.

In exponential backoff instead of retrying immediately it waits 1 second, then 2 then 4 then 8. The idea is to give API time to recover before sending requests again. If we have multiple clients sending request for this API, all the requests go through the full retry sequence of waiting, retrying, waiting, retrying before getting the fallback. But in case there's a Gemini outage then every request is still considered with complete retries with backoff before getting a response.

The circuit breaker solves this. It tracks consecutive Gemini failures across all requests. After 5 failures it stops sending requests entirely and returns fallback immediately for requests after that. After 60 seconds it allows one request through, if that succeeds it detects the external service is working again and allows requests as usual else it closes again.

In production with multiple workers this matters more. Without a circuit breaker every worker independently retries every failing request. With one, the pattern is detected quickly and all workers stop retrying until the service recovers.

This project runs as a single-process SQLite app, so the circuit breaker doesn't have multiple workers to protect. I built it this way because the pattern is the same regardless of scale, and if this moved to Gunicorn with 4 workers behind a load balancer, it would work without changes. The alternative was to leave it out and retrofit it later, which in practice means it never gets added until the first outage.

**Retry with Full Jitter**

For transient errors (429, timeout, 503) the system retries with jitter. It waits for a random amount of time between 0 and the cap before retrying.

Without jitter, if 50 patients hit a rate limit at the same moment, all 50 retry at the same moment causing a second rate limit. Randomness spreads them out so they stop hitting the API simultaneously.

**Per-Request LLM Timeout**

The circuit breaker handles failures, but a slow response is not a failure. If Gemini takes 45 seconds to respond, the circuit breaker sees nothing wrong. The patient is just staring at a spinner.

Each LLM call has a 15-second timeout enforced via a thread pool. If Gemini doesn't respond in time, it's treated as a transient error and the retry/fallback logic kicks in. The patient gets a fallback response instead of waiting indefinitely.

The google-genai SDK doesn't reliably respect HTTP-level timeouts (it overrides the httpx client timeout internally), so the timeout is enforced externally using `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=15)`.

**Graceful Degradation**

Three layers inside run_json_step:

If Gemini responded and the JSON is valid — return real data.

If Gemini responded but the JSON failed validation — send a repair prompt back to Gemini naming the exact error and showing it its own bad output. One attempt only.

If Gemini didn't respond at all, or repair also failed — use the hardcoded fallback. This preserves whatever data was already collected in state, returns a safe generic question to the patient, and keeps is_complete = False so the phase doesn't accidentally advance.

If the failure happens at report generation specifically, the report still generates directly from state data without Gemini. Allergies and medications are always included and clearly marked.

**Dual State Persistence**

State is stored in two places: LangGraph checkpoints (checkpoints.db) and a session_state table (app.db). This looks redundant but solves different problems.

LangGraph checkpoints are opaque blobs. They're serialized snapshots of the full graph state, and the only way to read them is to replay through LangGraph's checkpointer API. That works for resuming a conversation, but it's slow and awkward for the API layer that just needs to know "what phase is this session in?" to return it in a response.

The session_state table stores a compact snapshot of the fields the API actually needs — current_phase, identity, triage, clinical_step, etc. When `GET /resume/{thread_id}` is called, it reads one row instead of deserializing a checkpoint. During normal flow, the phase comes from the graph output in the `/chat` response; the session_state table is the fallback for resumption after a crash.

If I had to pick one, LangGraph checkpoints are the source of truth for conversation continuity. The session_state table is a read-optimized projection for the API layer.

**Background Report Generation**

Report generation runs as a background task with a job queue (queued → running → done → failed) instead of blocking the HTTP response.

The report step makes an LLM call that can take 3-8 seconds. In a synchronous flow, the patient's browser would be waiting on that response with the connection held open. With a single user that's fine. But with 10 patients finishing intake around the same time, 10 worker threads are blocked on LLM calls, and new requests start queueing.

The background job approach returns immediately with a job_id, and the frontend polls `/jobs/{job_id}` until it's done. The worker thread is freed up to handle other requests. If the report generation fails, the job status shows the error instead of the patient getting a generic 500.

This is also why the job table tracks status and error. If a report fails at 2 AM, a clinician can see it failed and why, instead of just having a missing report with no explanation.

**Idempotency Layer**

Every chat message includes a client_msg_id and a SHA256 hash of the message content. If the same client_msg_id arrives twice for the same thread, the server returns the cached response instead of processing it again.

This handles a few real scenarios: the patient double-clicks send, the browser retries on a timeout, or a mobile client on flaky connection sends the same request twice. Without idempotency, a double-send during the allergy phase could process "penicillin" twice, or worse, advance the phase twice and skip medications entirely.

The SHA256 hash is there to catch a different edge case: client_msg_id reuse. If a client reuses an ID with different content (bug or tampering), the server returns 409 instead of silently returning stale data.

**Outbound Notifications**

Two notification channels, each demonstrating a different integration pattern:

Slack is a simple JSON POST to an Incoming Webhook. It's the most common pattern for internal team alerts. Emergency escalations, crisis detections, and intake completions all post to the configured Slack channel so the care team gets immediate visibility without polling the dashboard.

The FHIR webhook is a different pattern entirely. It's a cryptographically signed payload delivery. The FHIR R4 Bundle is POSTed with an HMAC-SHA256 signature in the X-Signature header. The receiving system (an EHR, a data pipeline, a compliance logger) can verify the signature to confirm the payload wasn't tampered with in transit and actually came from this system. This is how real healthcare integrations work — you can't just POST patient data to an endpoint without authentication and integrity verification.

Both are fire-and-forget. A webhook failure is logged but never blocks the patient flow. If Slack is down, the patient still completes their intake.

**Guardrails Architecture**

Every patient message passes through multiple safety layers before and after the LLM sees it. The order matters.

```
Patient message arrives
    │
    ▼
1. Prompt injection check (extract.py)
   Regex scan for "ignore previous instructions", "you are now a..."
   Blocked messages get a neutral response, never reach the LLM
    │
    ▼
2. Crisis / self-harm detection (extract.py)
   Phrase matching for "want to die", "kill myself", etc.
   Returns 988 Lifeline info, creates escalation, does NOT terminate session
    │
    ▼
3. Emergency red flag detection (extract.py)
   Phrase matching with negation awareness ("no chest pain" ≠ "chest pain")
   Triggers immediate handoff to 911 + clinician notification
    │
    ▼
4. LLM processes the message (llm.py)
   Gemini extracts structured data from natural language
    │
    ▼
5. Diagnosis language filter (llm.py)
   Regex scan on LLM output for "you have", "consistent with", etc.
   If triggered, replaces LLM reply with a safe generic response
   Legal necessity — intake assistants cannot diagnose
    │
    ▼
Safe response delivered to patient
```

Layer 1 runs before any clinical logic because injected prompts shouldn't interact with the system at all.

Layer 2 runs before emergency detection because the response is different. A crisis gets the 988 Lifeline and a compassionate message. An emergency gets "call 911." Ordering them wrong would send a suicidal patient a 911 message instead of crisis resources.

Layer 3 runs before the LLM because the emergency check is deterministic and faster. If someone says "I'm having a seizure", there's no reason to wait for Gemini to extract OPQRST fields before escalating.

Layer 5 runs after the LLM because it's guarding against the LLM's own output. The prompt tells Gemini not to diagnose, but LLMs don't always follow instructions. The regex filter is the safety net.

The emergency phrases (layer 3) are stored in the database, not hardcoded. Clinicians can add or remove phrases through the admin API without a redeploy. If a new drug interaction creates a new emergency pattern, it can be added immediately.

## What I'd change for production

This project is built as a single-process app with SQLite. That's appropriate for a demo and for proving the architecture works end to end. Here's what would change if this needed to handle real patient load.

**Database: SQLite → PostgreSQL**

SQLite is single-writer. With WAL mode and busy_timeout it handles moderate concurrency, but under real load (50+ concurrent intakes) writes would bottleneck. PostgreSQL handles concurrent writes natively, supports row-level locking, and doesn't need the threading lock or retry wrapper that SQLite requires.

The LangGraph checkpointer would switch from SqliteSaver to PostgresSaver (langgraph ships both). The application DB queries are standard SQL and would migrate with minimal changes.

**Task queue: BackgroundTasks → Celery + Redis**

FastAPI's BackgroundTasks runs in the same process. If the process crashes, queued tasks are lost. Celery with Redis gives durable task queues, retries with backoff, dead-letter handling, and worker scaling independent of the web process.

Report generation would become a Celery task. The job table already tracks status, so the frontend polling logic wouldn't change at all.

**Circuit breaker: in-memory → Redis-backed**

The current circuit breaker tracks failures in a Python object. It works within a single process but if there are 4 Gunicorn workers, each has its own breaker with its own failure count. Worker A might have the breaker open while workers B, C, D are still hammering a dead API.

A Redis-backed breaker shares state across all workers. After 5 total failures (not 5 per worker), every worker stops sending requests simultaneously. The recovery probe also coordinates so only one worker tests the API, not all four.

**Horizontal scaling**

With PostgreSQL and Redis in place, the web layer becomes stateless. Multiple FastAPI instances behind a load balancer, each connecting to the same database and the same Redis. Session affinity isn't needed because all state lives in the database.

The LLM timeout and circuit breaker patterns are already designed for this. The idempotency layer already uses the database, so it works across instances without modification.

**HTTPS and authentication hardening**

The current setup uses HTTP with JWT for clinician auth. Production would add TLS termination (nginx or a cloud load balancer), CSRF tokens on form submissions, rate limiting per authenticated user instead of per IP, and token refresh rotation instead of a 24-hour expiry.

Patient sessions would also need authentication once real PHI is involved. The current model (anonymous sessions with a UUID) is acceptable for intake but not for accessing or modifying existing medical records.
