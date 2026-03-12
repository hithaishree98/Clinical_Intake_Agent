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
┌──────────────────────────────────┐
│   Outbound Integrations          │
│   ntfy.sh · Discord · Slack      │
│   FHIR webhook   (on completion) │
└──────────────────────────────────┘
```

## Why Langgraph here?

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

**Circuit Breaker over and Exponential fallback** 

I had used exponential fallback for LLM call failures before.

In exponential backoff instead of retrying immediately it waits 1 second, then 2 then 4 then 8. The idea is to give API time to recover before sending requests again. If we have multiple clients sending request for this API, all the requests go through the full retry sequence of waiting, retrying, waiting, retrying before getting the fallback. But in case there's a Gemini outage then every request is still considered with complete retries with backoff before getting a response.

The circuit breaker solves this. It tracks consecutive Gemini failures across all requests. After 5 failures it stops sending requests entirely and returns fallback immediately for requests after that. After 60 seconds it allows one request through, if that succeeds it detects the external service is working again and allows requests as usual else it closes again. 

In production with multiple workers this matters more. Without a circuit breaker every worker independently retries every failing request. With one, the pattern is detected quickly and all workers stop retrying until the service recovers.

**Retry with Full Jitter**

For transient errors(429, timeout, 503) the system retries with jitter. It waits for a random amount of time between 0 and the cap before retrying.

Without jitter, if 50 patients hit a rate limit at the same moment, all 50 retry at the same moment causing a second rate limit. Randomness spreads them out so they stop hitting the API simultaneously.

**Graceful Degradation**

Three layers inside run_json_step:

If Gemini responded and the JSON is valid — return real data.

If Gemini responded but the JSON failed validation — send a repair prompt back to Gemini naming the exact error and showing it its own bad output. One attempt only.

If Gemini didn't respond at all, or repair also failed — use the hardcoded fallback. This preserves whatever data was already collected in state, returns a safe generic question to the patient, and keeps is_complete = False so the phase doesn't accidentally advance. 

If the failure happens at report generation specifically, the report still generates directly from state data without Gemini. Allergies and medications are always included and clearly marked.

