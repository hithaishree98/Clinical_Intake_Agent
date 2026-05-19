# System Design

## The main idea

The LLM handles language — extracting structured data from what a patient says in natural speech. A LangGraph state machine controls everything else: which phase runs, when a phase ends, what constitutes a valid transition, and when to escalate. The model has no ability to decide what happens next. Every routing decision is Python code.

The reason for this split: LLMs are good at understanding "I've had a throbbing headache since this morning, maybe a 7 out of 10" and turning it into structured OPQRST fields. They are bad at reliably enforcing that allergies were collected before the confirmation screen, or at detecting mid-conversation that a patient just described a stroke. Keeping those responsibilities separate means each part can be tested and audited independently.

---

## How a message flows through the system

When a patient sends a message:

1. The FastAPI layer validates the session token, checks for prompt injection (`check_prompt_injection()` — four regex patterns for "ignore previous instructions", "you are now a", etc.), checks the per-session cost cap, and checks whether the LLM circuit breaker is open.
2. If all checks pass, `graph.invoke()` is called with the current `thread_id`.
3. LangGraph loads the full graph state from `checkpoints.db` and routes to `guard_node` first.
4. `guard_node` runs crisis and emergency detection. If nothing fires, it routes to the current phase node based on `current_phase` in state.
5. The phase node runs its LLM call(s), updates state, and returns.
6. LangGraph checkpoints the updated state to `checkpoints.db` and pauses (the graph uses `interrupt_after` on every patient-facing node).
7. The FastAPI layer reads the assistant reply from state and returns it.

There is no conversation state held in memory. Every message starts from a checkpoint read and ends with a checkpoint write.

---

## Intake phases

Sessions go through these phases in order. A session cannot skip a phase or move backward unless the patient explicitly requests a correction.

### Consent

Shown before any data is collected. The patient must explicitly agree — "yes", "sure", "I consent" all work. "No" or "I don't want to" ends the session immediately with no data retained. Intent classification (`_classify_intent`) handles the full range of natural-language responses; there's no keyword list here.

Consent can be disabled in settings if it's handled at registration.

### Identity

The patient provides name, DOB, phone, and address in free text. The LLM extracts and normalises into a typed schema — it handles "March fifteenth, eighty-five" and "15/3/1985" equally. Only fields not yet in state get filled, so the patient can give all four at once or one at a time across turns.

DOB validation runs after extraction: future dates and impossible ages get an immediate re-ask with a specific error. If all four fields still aren't collected after `max_identity_attempts` (default 3), the session directs the patient to the front desk.

A stable `patient_id` is derived from SHA-256(name + DOB) once both fields are present. This is what links visits together for returning patients.

### Identity review

For returning patients: the system shows details on file and asks the patient to confirm or update. "Keep" → stored details used, session continues. "Update" → an `identity_review` escalation is created for nurse follow-up, and the session continues with the patient's version.

For new patients: the system reads back what it extracted and asks the patient to confirm or correct. Correction routes back to the identity phase to re-collect.

Both branches use `_classify_intent` — it handles "yes that's right", "keep it", "looks good", "no update please" without a keyword list.

### Symptom assessment

One LLM call extracts chief complaint and full OPQRST simultaneously and also classifies the visit: `emergency_visit`, `routine_checkup`, `specialist_referral`, `mental_health`, or `pediatric`. That classification drives question phrasing in clinical history and the completeness threshold for the quality gate.

A deterministic quality scorer evaluates OPQRST completeness (0–1 scale). Threshold is 0.75 for ED mode, 0.60 for clinic. If below threshold, the system asks a targeted gap-fill for the first missing field. After two retries (`max_quality_retries = 2`) the session advances regardless — the patient is never stuck here.

If the LLM flags its own extraction as low confidence (`extraction_confidence = "low"`), a deterministic gap-fill replaces the LLM's generated question for that turn. This prevents the model from asking something sensible-sounding when it clearly didn't understand what the patient said.

### Clinical history

Sequences through four steps: allergies → medications → past medical history → recent labs/imaging. The step order is fixed but the system skips ahead past steps already collected — so if a patient corrects their allergies from the confirm screen, it only re-asks allergies and jumps back to confirm without re-asking medications, PMH, and results.

All four steps use LLM extraction to parse lists from natural speech. "Penicillin and I think latex too" becomes two separate allergy entries; "heart attack in 2019 and gallbladder out in 2021" becomes two separate PMH entries rather than one long string.

For medications specifically: if a medication name is extracted but frequency is missing, the system asks a follow-up once with warm phrasing. If the patient doesn't know, it's accepted on the second attempt. No third ask.

For optional steps (allergies, PMH, recent results), intent classification (`_classify_intent`) detects decline responses — "no", "nope", "I don't have any", "nothing comes to mind" — before sending the message to the LLM list extractor. A clear decline accepts empty immediately and moves on; the LLM is not called.

Questions adapt to the visit classification: mental health explicitly asks about psychiatric medications and supplements; pediatric addresses the parent throughout; emergency uses shorter, more urgent phrasing.

### Confirm

The system shows a natural-language summary of everything collected. The patient confirms or says what to change. Corrections route directly to the named step — "I need to change my allergies" routes to the allergies step, not back to the start. `_classify_intent` handles confirm; `_try_correction` handles routing to the right step. The session doesn't advance until the patient explicitly confirms.

### Report generation

The LLM generates a plain-text clinician note. A FHIR R4 bundle is built from the same validated state (Patient, Condition, AllergyIntolerance, MedicationStatement, Observation resources). If the LLM report fails, a deterministic template generates the note instead — the session always completes.

The note is saved to `app.db`. A Slack notification is sent. The FHIR bundle is dispatched to the configured webhook (HMAC-signed) and, if `FHIR_SERVER_URL` is set, pushed directly to the EHR server. Both the webhook and EHR push are best-effort — failure doesn't block the patient from getting a response.

Patient memory is upserted: allergies and conditions union across visits, medications replace, the last five chief complaints roll forward, and crisis flags never drop.

---

## Safety

`guard_node` runs before every node on every message. There is no code path that bypasses it.

### Crisis detection

Two-tier. Tier 1 is a phrase/regex scan — runs always, zero cost. Tier 2 is a lightweight LLM scorer that runs unconditionally on every message that Tier 1 doesn't catch. If the LLM returns `is_crisis_risk=True` with `high` or `medium` confidence, the session escalates. `low` confidence is logged as a soft distress signal but doesn't escalate.

The reason Tier 2 runs on every message rather than just "suspicious" ones: a soft-distress phrase list to gate the LLM was removed. A patient saying "I feel like dying, I don't know" matches no crisis phrase but the LLM classifies it correctly. A gate based on phrases would have missed it.

Once `crisis_detected` is set to `True` in state, it cannot be cleared. Subsequent messages return the safety response immediately without re-running detection.

### Emergency detection

A configurable phrase list loaded from the database with a 60-second TTL — phrases can be added via the admin panel without a server restart. Matching applies negation guards ("I don't have chest pain"), historical guards ("had chest pain years ago"), and reactivation detection ("stopped but came back").

**There is no LLM fallback for emergency detection.** Phrases not in the list won't fire. This is a known gap — "significant difficulty breathing" would miss if "significant" isn't how the list is phrased. Adding an LLM tier matching the crisis architecture is the right fix.

| Kind | Trigger | Input after detection |
|---|---|---|
| `crisis` | Suicidal/self-harm language (LLM or phrase) | Stays enabled — patient can keep typing |
| `emergency` | Emergency phrase match | Disabled — session ends |
| `identity_review` | Returning patient details mismatch | Normal — escalation is background |
| `human_review` | SafetyChecker score above threshold | Normal — report flagged for clinician sign-off |

### SafetyChecker preflight

Runs before report generation. Hard blocks prevent the note from being written: missing chief complaint, missing patient name, clinical history marked incomplete. Review signals raise the score without blocking alone — active emergency flag, crisis detected, identity unverified, OPQRST completeness below threshold. Reports above the review threshold get `X-Pending-Review: true` on the FHIR endpoint so a consuming EHR knows it needs sign-off.

### Output guardrails

Every LLM reply goes through `validate_llm_response()` before the patient sees it. Six regex patterns catch diagnosis language ("you have X", "consistent with", "this indicates"). A match replaces the entire reply with a safe response and logs the event.

---

## Storage

Two SQLite files in WAL mode:

**`app.db`** — everything the application owns: sessions, messages, reports, escalations, LLM usage, webhook delivery log, dead-letter records, patient memory summaries, emergency phrases, prompt experiment assignments, idempotency cache.

**`checkpoints.db`** — LangGraph's internal graph state only. Kept separate because LangGraph's internal schema changes with library upgrades and shouldn't require coordinating with application migrations. This is what enables session resumption — every node write is checkpointed here, so a server restart mid-intake is transparent to the patient.

SQLite connections are per-thread, opened lazily on first use. `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=10000` are set on every connection. This handles concurrent reads fine; concurrent writes start contending around 50 sessions. PostgreSQL is the path forward when you need scale.

---

## LLM pipeline

Every LLM call goes through `run_json_step()` which handles the full degradation chain:

```
Level 1 — Primary call: prompt → LLM → JSON parse → Pydantic validation
           ↓ (parse fails or validation fails)
Level 2 — Repair call: sends the validation error back to the model for one retry
           ↓ (repair also fails, or primary call failed entirely)
Level 3 — Hardcoded fallback dict: session continues, failure logged
```

Repair only runs on parse/validation failures — if the API call itself fails (timeout, auth error), the fallback is used immediately. Trying to repair an empty response wastes a token and always fails.

The circuit breaker (`CircuitBreaker` in `app/llm/circuit_breaker.py`) wraps all LLM calls. After 5 consecutive failures it opens; `/chat` returns a brief error without spending a token. After 60 seconds it moves to half-open, lets one probe through, and closes on success. Both thresholds are configurable in `settings.py`.

The `LLMProvider` interface (`app/llm/base.py`) abstracts the backend. The current implementation is `GeminiProvider` (Gemini Flash). Swapping backends means implementing the interface and calling `set_provider()` — the circuit breaker, degradation logic, cost accounting, and token tracking all sit above the interface and work unchanged.

---

## Design decisions worth explaining

**Two SQLite files.** LangGraph's checkpoint schema changes with library upgrades. If it's in `app.db` alongside application migrations, a library upgrade becomes a database migration problem. Keeping them separate means LangGraph can be upgraded by dropping and recreating `checkpoints.db` — you lose in-flight sessions but not reports or patient data.

**Repair call only on parse failure, not API failure.** This is easy to get wrong. If the API times out, there's no response to repair — a repair call will also time out. The repair call is only useful when the model returned something but it failed schema validation. The runner checks `llm_ok` explicitly before deciding whether to attempt repair.

**Idempotency on `/chat` uses key plus body hash.** The browser sends a `client_msg_id` with each message. On a cache hit, the system also compares a SHA-256 hash of the message body. Same `client_msg_id` + different body → 409. This lets the browser safely retry on network timeout without worrying about a different message being associated with the same key accidentally.

**Classification in the same call as symptom extraction.** Visit type (emergency/routine/specialist/mental health/pediatric) is extracted in the same LLM call as chief complaint and OPQRST, not a separate call. The classification drives question phrasing downstream (clinical history questions are different for mental health vs pediatric) and sets the OPQRST completeness threshold. Doing it in one call keeps latency down and ensures the classification always has the symptom context.

**The guard_node "once-set" behaviour.** When `crisis_detected` is already `True`, `guard_node` skips re-running detection and returns the safety response immediately. Detection runs exactly once — the first time the flag is set. This is important: it means a patient cannot send a later message that somehow routes around the flag, and it means repeated safety responses don't make additional escalation entries.

**Permanent vs transient error classification in the circuit breaker.** Auth failures (bad API key, 403) open the breaker immediately and don't retry — retrying a bad API key is pointless and burns time. Timeouts and 503s use exponential backoff with jitter before counting as a failure. This prevents a single network hiccup from opening the breaker while still catching a genuinely down provider.

---

## Known gaps and pending work

- Emergency detection needs an LLM tier matching the crisis architecture (phrase list + LLM fallback, not phrase list alone)
- `identity_review_node` has no retry cap — a patient sending garbage responses will loop forever; it needs a `review_attempts` counter with a graceful exit
- Clinical history steps (allergies, meds, PMH, results) have no per-step retry cap; a patient giving persistently unclear responses is stuck
- System-prompt caching is implemented in `GeminiProvider` but not active — current prompts don't clear Gemini's 2,048-token minimum for cache eligibility
- RxNorm normalisation is stubbed
