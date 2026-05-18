# Clinical AI Intake

A conversational intake agent for clinical settings. The patient types or speaks, the system collects their identity, symptoms, allergies, medications, and history through natural conversation, triages urgency in real time, and outputs a clinician note + FHIR R4 bundle.


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
```
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
```

## Key Features

- **Real-time safety detection** — crisis and emergency phrases checked before every node, not just at intake entry. Once a session is flagged it stays flagged; all subsequent messages return the safety response immediately.
- **Session resumption** — LangGraph checkpoints the full graph state after every node. A server restart mid-intake is transparent to the patient.
- **Cross-visit memory** — allergies and conditions union across visits; medications replace each visit; crisis flags never drop. Returning patients aren't re-asked things already on file.
- **Voice intake** — mic button sends audio to Groq Whisper; transcript flows through the same chat pipeline with a hallucination filter before it reaches the patient flow.
- **Clinician dashboard** — view all escalations, resolve with nurse notes, pull full session transcript and FHIR bundle per case.
- **Three-level LLM degradation** — primary call → repair on validation failure → hardcoded fallback. The patient always gets a response.
- **FHIR R4 output** — clinician note + bundle with Patient, Condition, AllergyIntolerance, MedicationStatement, and Observation resources, built from validated structured state.

## Data

- **app.db** (SQLite WAL): sessions, messages, reports, escalations, LLM usage, webhooks, patient memory
- **checkpoints.db**: LangGraph conversation state — this is what enables session resumption after a server restart
- **Dead-letter table**: exhausted webhook deliveries stored here for manual replay

All log events run through a PHI redaction pass before writing to stdout — names, DOBs, and phone numbers are replaced with `[REDACTED]` before any log aggregation sees them.

## Architecture

Full node descriptions, design decisions, and pipeline: [SystemDesign.md](SystemDesign.md)

## Testing

Manual test scenarios, test coverage, and eval categories: [TestingAndEvals.md](TestingAndEvals.md)


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


## Tech stack

FastAPI · LangGraph · Pydantic v2 · Google Gemini API · SQLite WAL · FHIR R4 · JWT · HMAC-SHA256 · Docker · pytest · slowapi · bandit · pip-audit · Slack · Chart.js
