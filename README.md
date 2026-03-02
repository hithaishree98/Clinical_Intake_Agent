# Production-Safe Clinical AI Workflow

A state-machine-driven patient intake agent for clinical environments where AI failure has real consequences.

The core design principle: **LLM handles language, state machine handles control.** The AI extracts what patients say — but it cannot skip mandatory intake fields, override clinical logic, or advance the workflow without deterministic validation at every step.

---

## What it does

A patient opens the app and is guided through a structured clinical intake:

1. **Identity** — name, DOB, phone, address (deterministic extraction + EHR lookup)
2. **Symptom collection** — chief complaint + OPQRST via LLM extraction
3. **Emergency triage** — red-flag phrase detection with negation handling (e.g. "no chest pain" does not trigger)
4. **Clinical history** — allergies, medications (dose/freq/last taken), PMH, recent labs
5. **Confirm** — patient reviews a summary and can correct any section
6. **Report** — clinician note generated as plain text + FHIR R4 Bundle

When complete, the clinician gets a structured note and a FHIR R4 Bundle ready for EHR import. Emergency escalations fire a Slack alert and route to a clinician review queue immediately.

---

## Architecture

```
Patient (browser)
      │  HTTP
      ▼
FastAPI  ──► LangGraph state machine  ──► Gemini (LLM extraction only)
      │            │
      │            └── SQLite WAL (crash-safe checkpointing)
      │
      ├── /report/{id}        plain text clinician note
      ├── /report/{id}/fhir   FHIR R4 Bundle (application/fhir+json)
      ├── /clinician/pending  escalation queue (JWT-protected)
      └── /jobs/{id}          async report generation status
```

**Why a state machine instead of pure LLM?**
In regulated environments, "the AI decides what happens next" is not acceptable. Every phase transition is a deterministic code path. The LLM is sandboxed to extraction — it fills fields, never drives flow.

---

## Safety features

| Feature | How it works |
|---|---|
| Pydantic validation | Every LLM output is schema-validated before it touches state |
| SQLite WAL | Crash recovery — interrupted intakes resume exactly where they left off |
| Emergency escalation | Red-flag detection fires before LLM sees the message; routes to clinician queue |
| Idempotent chat | client_msg_id + request hash prevents double-processing on retry |
| JWT clinician auth | /clinician/* routes require a signed token (24hr expiry) |
| Rate limiting | 10 sessions/hr, 60 chat messages/min per IP via slowapi |
| Audit trail | Every message, state snapshot, and escalation written to SQLite |

---

## FHIR R4 output

After intake completes, `GET /report/{thread_id}/fhir` returns a FHIR R4 Bundle containing:

- `Patient` — identity (name, DOB, phone, address)
- `Condition` — chief complaint + OPQRST as clinical note
- `AllergyIntolerance` — one resource per allergy
- `MedicationStatement` — one resource per medication (dose, frequency, last taken)
- `Observation` — triage risk level and visit type

```json
{
  "resourceType": "Bundle",
  "type": "document",
  "entry": [
    { "resource": { "resourceType": "Patient", ... } },
    { "resource": { "resourceType": "Condition", ... } },
    ...
  ]
}
```

This output is drop-in compatible with EHR systems like Epic and Cerner — no transformation required on the receiving end.

---

## Running locally

```bash
# 1. Clone and install
git clone https://github.com/hithaishree98/Clinical_Intake_Agent.git
cd Clinical_Intake_Agent

# 2. Set environment variables
cp .env.example .env
# Edit .env: add GEMINI_API_KEY, JWT_SECRET, CLINICIAN_PASSWORD

# 3. Run with Docker Compose
docker compose up

# App at http://localhost:8000
# Clinician portal: POST /clinician/token to authenticate
```

**Note:** Not deployed due to free-tier memory constraints. All functionality verified locally via Docker and pytest.

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Tech stack

FastAPI · LangGraph · Pydantic v2 · Gemini · SQLite WAL · FHIR R4 · JWT · Docker · pytest · slowapi
