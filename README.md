# Clinical AI Intake

A conversational intake agent for clinical settings. The patient types or speaks, the system collects their identity, symptoms, allergies, medications, and history through natural conversation, triages urgency in real time, and outputs a clinician note + FHIR R4 bundle.

> **Demo**: _add 30–60s screen recording — chest pain emergency → escalation → clinician dashboard → FHIR bundle_

## Problem

Clinical intake today is a paper form or a dropdown tablet. Neither produces structured output an EHR can actually ingest, and neither can detect mid-conversation that a patient is describing a cardiac event.

I wanted to fix that but not by just handing everything to an LLM. An LLM alone will skip allergies, mark intake complete while required fields are missing, and treat an escalation as a routine follow-up. In healthcare those aren't acceptable failure modes. So the architecture is specifically designed to prevent them: the LLM handles language understanding and extraction. A fixed state machine controls flow, phase transitions, and safety checks.

## Architecture Overview

I used LangGraph to build a fixed state machine, each phase of intake is a separate node with a single job. The LLM runs inside each node but has no control over flow, phase transitions, or safety checks. Those are all code.

```
Browser / Voice
    ↓
FastAPI — rate limiting · authentication · input validation
         · cost cap · circuit breaker
    ↓
LangGraph state machine — checkpointed to SQLite after every node
    ├── guard_node — crisis + emergency screen, runs before every node
    ├── consent → identity → identity_review → subjective
          → clinical_history → confirm → report
    └── handoff_node — reached on crisis or emergency; directs to 988 or 911
    ↓
SQLite — app.db (sessions, reports, escalations, LLM usage, webhooks, patient memory)
       — checkpoints.db (LangGraph graph state, kept separate)
    ↓ background threads
Slack alerts · HMAC-signed FHIR webhook
```

## Flow

Patient opens the app and clicks New Session
         ↓
guard_node — runs on every message, regardless of which phase intake is in
  Crisis detection: keyword/regex first, LLM only if inconclusive
  Emergency detection: phrase list with negation and context guards
  If crisis detected → 988 Lifeline message, session ends
  If emergency detected → 911/ER message, clinician notified via Slack, session ends
  Otherwise → continues to the current phase
         ↓
Consent
  Patient is shown an AI disclosure before any data is collected
  Must explicitly agree to continue, declining ends the session
         ↓
Identity
  The patient provides 4 fields: name, DOB, phone, address
  LLM extracts and normalises into a typed schema (IdentityOut)
  Name looked up in database to detect returning patients
         ↓
Identity review
  Returning patient: system shows details on file, patient confirms or requests update
    Update → identity_review escalation created, session continues with patient's version
    Keep → stored info used, no escalation
  New patient: system reads back what it extracted, patient confirms or corrects
    Correction → routes back to Identity to re-collect
         ↓
Subjective
  LLM extracts chief complaint + full OPQRST
  Also classifies intake type: emergency/routine/specialist/mental health/pediatric
  Quality gate: if OPQRST completeness below threshold (0.75 ED / 0.60 clinic), retries up to 2 times with a deterministic gap-fill question
         ↓
Validation gate (transparent — runs without pausing the patient)
  Checks required fields before allowing phase transitions
         ↓
Clinical history
  Collects Allergies → Medications → Past medical history → Recent labs
  Questions adapt to intake classification (pediatric/spealist/mental health meds ask differently)
  Dosage follow-up: asks with warm phrasing; accepts no answer on second attempt and moves on
         ↓
Confirm
  Natural-language paragraph summary of everything collected
  Patient can correct any section — type "I need to change my allergies" and the system
  routes back to that step without losing other collected data
  Must explicitly confirm before proceeding
         ↓
Report generation
  LLM generates plain text clinician note
  FHIR R4 Bundle built from the same validated state
  Full report shown directly in chat
  Both saved to database
  Slack notification sent to clinician channel
  FHIR Bundle posted to configured webhook URL (HMAC-signed)
  FHIR Bundle pushed directly to HAPI/Azure/Epic if FHIR_SERVER_URL is set
         ↓
Clinician receives complete note in the portal


Clinician opens the portal
  Go to /dashboard and enter the clinician password in the auth bar at the top, then click Auth to load case notes
  Click View Escalations to see all flagged cases
  Click an escalation to populate the resolve form, add a nurse note, and click Resolve


## Technical decisions 

**Guard node runs before every node, not just at entry.**
Crisis and emergency detection aren't a filter at the front door. Guard_node is wired into the graph so it runs before consent, identity, subjective, history — every step. Once a session is flagged it stays flagged; all subsequent messages return the safety response immediately regardless of which phase intake is in.

**Two-tier crisis detection to keep LLM calls cheap.**
The first pass is keyword/regex — handles obvious cases at zero cost. The LLM only comes in when the first pass finds nothing but soft distress signals are present (hopelessness, burden language, passive ideation). Same pattern for intent classification.

**Three levels of LLM degradation so the session never crashes.**
Primary call → Pydantic validation. If that fails but the model responded, a repair call sends the exact validation error back to the model in a "REPAIR REQUIRED" prompt. If that also fails, a hardcoded fallback dict. The patient always gets a response.

**OPQRST completeness is scored deterministically, not by the LLM.**
Each field has a weight (chief complaint 0.25, onset 0.20, severity 0.20, etc). Score below 0.75 for ED or 0.60 for clinic and the system asks a targeted gap-fill question — no LLM involved. After two failed quality checks it switches to an "I have X, still need Y" summary so the patient isn't stuck in a loop.

**Circuit breaker on the LLM so load spikes don't cascade.**
After 5 consecutive failures the circuit opens and /chat returns 503 without spending a token. After 60 seconds it half-opens and lets one probe through. A successful probe closes it; a failure resets the timer.

**Correction routing to specific fields, not just sections.**
If a patient says "I want to change my medications", they go directly to the medications step, not the top of clinical history. The system recognizes section-level corrections via regex across identity, symptom, and history sub-fields, and routes back to the right step without losing anything else collected.

**Safety preflight before report generation.**
Before generating a clinician note, a weighted scoring pass runs over the session state. Hard blocks — missing chief complaint, incomplete clinical history — prevent the report from being written entirely and create an escalation record so the clinician knows why. Review signals (active emergency, identity mismatch, low extraction quality) raise the score and can flag the case for mandatory clinician review.

**Cross-visit memory with field-level merge rules.**
Allergies and conditions union across visits. Medications replace each visit. Last five chief complaints kept as a rolling list. Crisis flags are never dropped. On the next visit this summary is injected into prompts so returning patients aren't re-asked things the system already knows.

**Session resumption after restart.**
LangGraph checkpoints the full graph state to SQLite after every node. If the server restarts mid-intake, the next request with the same thread_id picks up from exactly the interrupted node — the patient doesn't start over. This works as long as checkpoints.db is on a persistent volume, which is why it's kept separate from app.db rather than bundled together.

**Prompt caching via Gemini's CachedContent API — infrastructure built, not yet active.**
The infrastructure is there — cache registry, TTL, hash invalidation on prompt changes, fallback to inline if creation fails. What's not active yet: Gemini requires a 2,048-token minimum and none of the current prompts clear it. In future if we add more examples  that activates caching on the node that runs most often.

**Outbound webhook idempotency via payload hash.**
A SHA-256 hash of the payload is used as an idempotency key for outbound FHIR webhooks so the same notification is never delivered twice. Failed deliveries retry with exponential backoff; exhausted deliveries go to a dead-letter table for manual replay.

**Voice hallucination filter on Whisper output.**
Whisper occasionally hallucinates YouTube caption phrases ("thanks for watching", "[music]") on silent or low-audio clips. These are filtered out before the transcript reaches the chat pipeline, and the patient gets a "didn't catch that" prompt instead.

**Medication name spelling correction.**
The LLM silently corrects common misspellings ("lisonopril" → "lisinopril", "metfornim" → "metformin"). Unrecognizable names trigger a re-prompt rather than storing garbage in the record.

## Quick start

Requires Docker and Docker Compose.

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Yes | |
| `JWT_SECRET` | Yes | any strong random string |
| `CLINICIAN_PASSWORD` | Yes | for dashboard auth |
| `SLACK_WEBHOOK_URL` | No | Slack alerts on escalations |
| `GROQ_API_KEY` | Voice only | Whisper transcription |

```sh
docker compose up --build
```

- App: http://localhost:8000
- HAPI FHIR server: http://localhost:8080 (takes ~90s on first boot)
- Clinician dashboard: `/dashboard` → enter clinician password → click Auth
- Admin panel: linked from dashboard

## Data

- **app.db** (SQLite WAL): sessions, messages, reports, escalations, LLM usage, webhooks, patient memory
- **checkpoints.db**: LangGraph conversation state — this is what enables session resumption after a server restart
- **Dead-letter table**: exhausted webhook deliveries stored here for manual replay

All log events run through a PHI redaction pass before writing to stdout — names, DOBs, and phone numbers are replaced with `[REDACTED]` before any log aggregation sees them.

## Tech stack

FastAPI · LangGraph · Pydantic v2 · Google Gemini API · SQLite WAL · FHIR R4 · JWT · HMAC-SHA256 · Docker · pytest · slowapi · bandit · pip-audit · Slack · Chart.js
