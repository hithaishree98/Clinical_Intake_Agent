# System Design

The system is a conversational intake agent. The patient types or speaks, the system collects identity, symptoms, allergies, medications, and history through natural conversation, triages urgency in real time, and outputs a clinician note and FHIR R4 bundle.

The architectural principle: the LLM handles language understanding and extraction. A fixed LangGraph state machine controls flow, phase transitions, and safety checks. The model has no ability to skip a phase, route around a validation gate, or suppress an escalation. Those are all code.

---

## Intake flow

Each session goes through these phases in order:

**Consent** — shown before any data is collected. Patient must explicitly agree; declining ends the session with no data retained. Configurable — can be disabled if consent is handled at registration.

**Identity** — patient provides name, DOB, phone, and address in free text. LLM extracts and normalises into a typed schema. Patient ID is derived from SHA-256(name|dob) and the record is checked for a prior visit. After three failed extraction attempts, the system directs the patient to the front desk.

**Identity review** — for returning patients: stored details shown alongside what was just provided, patient chooses keep or update; a discrepancy creates a nurse-review escalation. For new patients: extracted details read back, patient confirms or corrects; correction routes back to identity.

**Symptom assessment** — LLM extracts chief complaint and full OPQRST in one call. At the same time it classifies the visit: `emergency_visit`, `routine_checkup`, `specialist_referral`, `mental_health`, or `pediatric`. A deterministic quality scorer evaluates completeness (0–1 scale); below threshold it asks a targeted gap-fill for the specific missing field. After two retries the session advances regardless — the patient is never stuck.

**Clinical history** — sequences through allergies → medications → PMH → recent results. Questions adapt to the visit classification: mental health explicitly asks for psychiatric medications and supplements, pediatric is parent-addressed throughout, emergency uses short urgent phrasing. Returning patients get "anything new?" — "no" preserves the prior record. Mid-sequence corrections route to the named step without losing other answers.

**Confirm** — natural-language paragraph summary of everything collected. Patient confirms or names what to change. Corrections route directly to the named section — identity, symptoms, or a specific history step — without resetting the rest. Session doesn't advance until confirmed.

**Report generation** — LLM generates a plain-text clinician note. System builds a FHIR R4 bundle (Patient, Condition, AllergyIntolerance, MedicationStatement, Observation). Both saved to the database. Slack notification sent, FHIR bundle posted to the configured webhook and pushed to the EHR server in background threads so delivery latency doesn't slow the patient-facing response. Patient memory upserted for next visit.

---

## Safety and escalation

`guard_node` runs before every node on every message. There is no code path that bypasses it.

**Emergency detection** matches each message against a configurable phrase list loaded from the database with a 60-second TTL — clinicians can add phrases without a server restart. Matching applies negation guards ("I don't have chest pain"), historical guards ("I used to have chest pain years ago"), and reactivation detection ("it stopped and then came back"). A match routes to `handoff_node`, disables chat input, and dispatches a Slack alert.

**Crisis detection** is two-tier. Tier 1 is keyword and regex — runs always, zero cost. Tier 2 is a lightweight LLM scorer that runs only when Tier 1 finds soft distress signals but no definitive keyword ("no point", "hopeless", "burden to everyone"). Confirmed crisis routes to `handoff_node` with the 988 Lifeline message but chat input stays enabled — the patient can keep typing. Once a crisis or emergency flag is set on a session it cannot be cleared.

| Kind | Trigger | Session state | Slack alert |
|---|---|---|---|
| `emergency` | Emergency phrase detected | Ends, input disabled | Yes |
| `crisis` | Suicidal/self-harm language | Continues, input enabled | No |
| `identity_review` | Returning patient details mismatch | Continues | No |
| `human_review` | SafetyChecker score above threshold | Report flagged, not blocked | No |

**SafetyChecker preflight** runs before report generation. Hard blocks prevent the note from being written at all: missing chief complaint, missing patient name, incomplete clinical history. Review signals raise the score without blocking alone: active emergency flag, crisis detected, identity unverified, low extraction quality. Reports above the review threshold include `X-Pending-Review: true` on the FHIR endpoint so a consuming EHR knows it needs clinician sign-off before acting on the data.

**Output guardrails** run after every LLM reply. Six regex patterns catch diagnosis language ("you have X", "consistent with", "diagnos\*"). Any match replaces the entire reply with a safe response and logs the event — the model cannot route around this.

**validate_node** is a silent routing gate between phase transitions. It checks required fields are actually populated before the session advances. Either it passes and routes forward, or routes back to the source phase with a targeted gap message. Never shown to the patient.

---

## Patient experience

**Corrections anywhere** — a patient can change any previously given answer at any phase, including from the confirm screen. Corrections route directly to the named step without resetting anything else. If the patient says "go back" without specifying what, the system shows a short menu.

**Returning patients** — the prior summary (allergies, conditions, medications, last five complaints, crisis flags) is injected into the identity review prompt. Returning patients get a warm acknowledgment, not a raw comparison table. Clinical history questions are phrased as "anything new?" — "no" preserves the prior record. Crisis flags never drop between visits.

**Voice** — the mic button sends audio to `POST /transcribe` (Groq Whisper), which returns a transcript. A hallucination filter rejects empty transcripts or ones matching known Whisper hallucination phrases ("Thanks for watching"). The browser then posts the transcript to `POST /chat` — voice goes through the exact same pipeline as typed input. On identity and clinical history turns a confirm strip appears so the patient can review before submitting, since Whisper commonly mistranscribes phone numbers and medication names.

**Quick replies** at binary-gate phases (consent, identity review, confirm) — the API response includes labelled buttons. Selections bypass intent classification and go directly to the graph.

**Quality retry loop** — if OPQRST completeness is below threshold, the system asks one targeted gap-fill for the specific missing field. After two retries the session advances. 

**Dosage follow-up** — if medications are given without dosage, the system asks once with warm phrasing. If the patient says they don't know, accepted on the second attempt. No third ask.

---

## Architecture

When a message comes in it hits the FastAPI layer first. Before the graph is invoked:
- Session token checked
- Prompt injection filter (regex blocks "ignore previous instructions", "you are now a", etc.)
- Circuit breaker checked — if open, returns a "try again shortly" message without touching the LLM
- Max session turns checked
- Per-session cost cap checked against cumulative LLM spend for the thread

If all pass, `graph.invoke()` is called. Every message enters through `guard_node` first, then routes to the current phase node based on `current_phase` in state. After each patient-facing node the graph pauses (LangGraph `interrupt_after`) and the API returns the reply. State is checkpointed to `checkpoints.db` after every node.

**LLM pipeline** — three levels of degradation per call so the session always continues:

```
Level 1: Primary call → JSON extract → Pydantic schema validation
              ↓ (validation fails)
Level 2: Repair call — sends the exact validation error back to the model
              ↓ (repair also fails, or primary call failed entirely)
Level 3: Hardcoded fallback dict — session continues, failure logged
```

The circuit breaker tracks consecutive LLM failures. After five it opens; `/chat` returns a brief message without spending a token. After 60 seconds it moves to HALF_OPEN, lets one probe through, closes on success.

Every LLM call goes through the `LLMProvider` interface. Current implementation is `GeminiProvider` (Gemini Flash). Swapping backends means implementing one class and calling `set_provider()`. The circuit breaker, retry logic, cost accounting, and logging all work against the interface, not the implementation.

**Storage** — two SQLite files in WAL mode:
- `app.db` — sessions, messages, reports, escalations, patient memory, LLM usage, webhook deliveries, idempotency cache, emergency phrases, prompt experiments
- `checkpoints.db` — LangGraph's internal graph state only. Kept separate because LangGraph's internal schema changes with library upgrades and shouldn't require coordinating with application migrations.

---

## Design decisions

**Fixed state machine, not an agent** — LangGraph gives a fixed, auditable node sequence. The LLM operates inside each node and extracts structured information from what the patient said. It has no ability to decide which node runs next, skip phases, or override safety checks. Every routing decision is Python code — testable and auditable in a way that prompt-based routing is not.

**Intake classification in the same call as symptom extraction** — visit type is extracted in the same LLM call as the chief complaint, not a separate call. This drives two things downstream: clinical history question phrasing per (step, classification) pair, and the OPQRST completeness threshold (0.75 for ED, 0.60 for clinic).

**Deterministic OPQRST quality scoring** — completeness is a weighted formula, no LLM involved. Each field has an assigned weight; the scorer returns a 0–1 value. Gap-fill questions are built deterministically from the first missing field in priority order. The threshold is easy to audit and the retry path has no LLM cost.

**Two-tier detection for intent and crisis** — same pattern for both. Deterministic first pass (regex/keyword) handles obvious cases at zero cost. LLM brought in only when the first pass is inconclusive. Safety-critical logic stays in code.

**Cross-visit memory with field-level merge rules** — memory merges per field type rather than a whole replace: allergies and conditions union across visits (they don't go away), medications replace each visit (patients start and stop them), last five chief complaints as a rolling list, crisis flags never drop.

**System-prompt caching** — per-schema-type cache registry in `GeminiProvider`, 55-minute client TTL, SHA-256 hash invalidation on prompt change. None of the current prompts clear Gemini's 2,048-token minimum so the code path is built but not yet active.

**Session resumption** — no cookies. `/start` returns a `thread_id` (UUID) and a `session_token` (random 64-char hex). The browser stores both in localStorage. Every `/chat` call sends `thread_id` in the form body and `session_token` in the Authorization header. State is never held in memory — after every node LangGraph writes the full graph state to `checkpoints.db`. On the next `/chat` call it passes the thread_id back and LangGraph picks up exactly where it left off. A server restart between two patient messages is transparent.

`GET /resume/{thread_id}` is for browser refreshes — reads from `app.db`, not checkpoints.db, and returns the current phase and a short context message without invoking the graph.

**Idempotent outbound webhooks** — SHA-256 hash of the payload as the idempotency key so the same FHIR bundle is never delivered twice. Exhausted deliveries go to a dead-letter record for manual replay.

**Permanent vs transient error classification** — the circuit breaker distinguishes permanent errors (bad API key, auth failure) from transient ones (timeout, 503). Permanent errors fast-fail immediately and open the breaker — retrying an auth failure is pointless and burns time. Transient errors retry with exponential backoff and random jitter before failing. This prevents the breaker from opening on a single network hiccup while still catching a genuinely broken provider.

**Report node runs synchronously** — `patient.py` checks `if phase == "report"` and runs `_run_report_inline()` in the same `/chat` request rather than backgrounding it. The patient gets the complete clinician note back in the same response that triggered report generation. Backgrounding it would require either polling or a server-push mechanism.

**Skip-ahead after corrections in clinical history** — `_next_clinical_step_needed()` walks the step order forward from the corrected step and returns the first step whose state field is still `None`. It only re-asks unanswered steps. This is what prevents re-asking medications after a patient corrects allergies — the answered fields are still in state and are skipped over.

**Incoming `/chat` idempotency — key plus hash** — `/chat` takes a `client_msg_id` from the browser. On a cache hit, it also compares a SHA-256 hash of the message body. If the same `client_msg_id` arrives with a different message, it returns 409 rather than silently returning the cached response for the wrong message. A client can safely retry on network timeout; it cannot use the same key to substitute a different message.

**Repair call only fires on parse failure, not API failure** — if the LLM API call fails entirely (timeout, error), repair is skipped and the session goes straight to the hardcoded fallback. Repair only runs when the model returned a response but the JSON failed schema validation. Trying to repair an empty or error response wastes a token and always fails.

**Guard node once-set behaviour** — when `crisis_detected` is already `true` in state, guard_node skips re-running detection and immediately returns the safety message and routes to handoff. Detection logic runs once — the first time the flag is set — and subsequent messages are a simple state read. A later message cannot clear the flag or route around it.

---

## Operations

**Health endpoints:**
- `GET /health` — always 200, for process supervisors
- `GET /ready` — 503 if SQLite is unreachable, the graph failed to compile, or the circuit breaker is open. Use this for load balancer readiness checks.

All log events pass through `log_event()` which runs PHI redaction before writing to stdout. Names, DOBs, and phone numbers are replaced with `[REDACTED]`.

**Admin panel** (`/admin/*`, requires clinician JWT):
- `GET /admin/analytics` — LLM cost today, average per session, primary/repair/fallback rates, cache hit rate, circuit breaker state, last 7 days
- `GET/POST/DELETE /admin/emergency-phrases` — live phrase management; changes take effect within 60 seconds via the TTL cache, no server restart needed
- `GET /admin/webhooks` — webhook delivery log, per-entry retry counts and HTTP status
- `POST /admin/demo/reset` — wipe all session data and re-seed mock EHR patients
- `GET/POST/PATCH /admin/experiments` — prompt A/B experiment management

**Clinician dashboard** (`/clinician/*`, requires clinician JWT):
- `POST /clinician/token` — exchange the clinician password for a short-lived JWT; only endpoint that doesn't require a prior token
- `GET /clinician/pending` — all unresolved escalations across all sessions
- `POST /clinician/resolve` — mark resolved, attach nurse note, session returns to active
- `GET /clinician/case/{thread_id}` — full transcript, clinician note, all escalations with reasons and safety score
- `GET /clinician/report/{thread_id}/fhir` — FHIR R4 bundle as `application/fhir+json`; includes `X-Pending-Review: true` header if the report needs clinician sign-off before EHR ingestion

---

## Limitations and future work

**What the system cannot do:**
- No physical examination, vital signs, diagnostic tests, or prescriptions
- Does not provide a diagnosis — output is clinical description language only
- SQLite is single-instance; contention begins at ~50 concurrent sessions
- Standard Gemini API is not HIPAA BAA eligible — Vertex AI required for a covered deployment
- Drug name normalisation (RxNorm) is stubbed — strips whitespace only
- Single clinician password, no role separation (nurse, physician, admin)

**Production path:**
- Replace SQLite with PostgreSQL, swap `SqliteSaver` for LangGraph's `PostgresSaver`
- Vertex AI for HIPAA eligibility
- Wire `normalize_drug_name()` to the RxNorm API
- Send session links via SMS or email so resumption works across devices
- Role-based access with scoped JWT claims
- Appointment scheduling integration so the clinician note is available before the visit
