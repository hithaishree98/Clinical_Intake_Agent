# Clinical AI Workflow

A state-machine-driven conversational agent that guides patients through structured clinical intake, extracts information from natural language, triages urgency, and outputs a clinician note and FHIR R4 health record.

Built on a core principle: LLM handles language, state machine handles control.

The AI extracts what patients say. It cannot skip phases, override clinical logic, or advance the workflow without deterministic validation at every step.

## Problem Context

Most clinical intake today is a paper form or a dropdown-based tablet app. Both share the same flaw: patients don't speak in structured fields.

A patient says "it's been hurting on and off since Tuesday, worse when I breathe in." A dropdown gives them "chest pain: yes/no." Information gets lost, fields are left blank, and clinicians spend time reformatting notes. And a static form can't detect mid-conversation that a patient is describing a medical emergency and flag it immediately.

This project bridges that gap by converting natural patient language into structured clinical documentation while continuously monitoring for urgency, without forcing patients to adapt to the system.

## Why this approach

In a clinical setting, an LLM alone cannot be allowed to decide what to ask, when intake is complete, or whether something is serious. A model might skip allergies, treat escalations as routine, or mark intake complete while key fields are missing. In healthcare, these are not acceptable failure modes.

The architecture of this system is specifically designed to prevent those failure modes. The LLM handles language understanding and extracting structure from natural text.

Everything else — what to ask, what order, what constitutes an emergency, when to escalate — is deterministic code.

## What it does

- Full conversational intake covering identity, chief complaint, OPQRST symptom assessment, allergies, medications, past medical history, and recent labs.
- A centralised `guard_node` runs crisis detection before every single business node. If a patient mentions self-harm in any phase, they get the 988 Lifeline response regardless of how far along they are. Identity is never a prerequisite for a safety response.
- Two-tier crisis detection: keyword/regex fires first at zero latency; a second LLM classifier handles soft distress signals ("I can't take it anymore") that don't match exact phrases.
- Two-tier intent classification replaces hardcoded yes/no word lists. Exact matches are free. Ambiguous short messages ("I think so", "not really") go to a lightweight LLM call (max_tokens=40, temperature=0.0).
- LLM-based identity extraction with Pydantic schema validators: name normalised to Title Case, DOB to ISO 8601 from any format including "March 3rd 1992", phone to 10 digits stripping +1.
- Emergency detection that runs in the subjective node. If a patient mentions chest pain or a seizure, they get immediate escalation regardless of what else they said. Emergency phrases are stored in the database; clinicians can add or remove them without a redeploy.
- Patients confirm AI-assisted intake before any data is collected.
- Identity verification against an EHR record. Returning patients get a warm acknowledgment with their details on file. Discrepancies are flagged for nurse review.
- Layer-2 patient memory (`app/memory.py`): after each completed intake, allergies and chronic conditions are unioned into a cross-visit summary, medications are replaced with the current list, and the last five chief complaints are kept. On the next visit, this context is injected into LLM prompts so the model doesn't ask a returning patient with known penicillin allergy to list their allergies from scratch.
- `None` vs `[]` sentinel distinction: clinical fields start as `None` (not yet asked) rather than `[]` (asked, none reported), so lookahead skipping works correctly when patients volunteer information early.
- If the server goes down mid-intake, the patient resumes exactly where they left off, with a context-aware summary of what was collected.
- Three guards in `/chat` before any graph invocation: circuit breaker open, max session turns reached (30), active report job already running.
- All tunable knobs in `IntakeConfig`: quality thresholds, cost caps, LLM pricing constants, identity max attempts, session TTL. A single place to A/B test or adjust operational parameters without grepping the codebase.
- Generates a structured clinician note and a FHIR R4 Bundle. Both are shown in the chat on completion and sent to the clinician team.
- Operations dashboard at `/dashboard` showing session KPIs, escalation breakdown, LLM health, circuit breaker state, and API cost — auto-refreshes every 30 seconds.
- Clinician portal for reviewing and resolving escalations.
- Slack notifications for emergencies, crisis language (includes what the patient typed and whatever identity was collected), and completed intakes. FHIR Bundle posted via HMAC-signed webhook on completion.
- Dead-letter webhook recovery: exhausted deliveries are re-queued on startup and hourly so a transient downstream outage doesn't drop records permanently.
- Database schema versioned with Alembic. Every schema change is a migration file with an `upgrade()` and `downgrade()`. CI runs `alembic upgrade head` before deploy so schema drift between environments is impossible.

## How it works

```
Patient opens the app and clicks New Session
         ↓
guard_node (runs before every step below)
  Two-tier crisis detection on every message
  Keyword/regex Tier 1 → immediate 988 Lifeline response
  Soft distress Tier 2 → LLM classifier confirms or clears
         ↓
Consent
  Patient is shown an AI disclosure before any data is collected
  Must explicitly agree to continue — declining ends the session
  Two-tier intent classification: "I suppose" → LLM → confirm
         ↓
Identity
  LLM extraction with schema-level normalisation (IdentityOut)
  Name → Title Case, DOB → ISO 8601, phone → 10 digits
  Name looked up in EHR
    Returning patient → warm acknowledgment with details on file
    New patient → confirms details before continuing
    Discrepancy → escalated for nurse review
         ↓
Symptom collection (subjective)
  Emergency check on every message (before LLM)
  LLM extracts chief complaint + full OPQRST in one call
  Also classifies intake type: emergency/routine/specialist/mental health/pediatric
  Quality gate: score < threshold → deterministic gap-fill question, up to 2 retries
         ↓
Validation gate (silent node)
  Checks required fields before allowing phase transitions
  Fails gracefully with targeted patient-facing messages
         ↓
Clinical history
  Lookahead: if patient volunteered allergies/meds/PMH earlier, those steps skip
  Allergies → Medications → Past medical history → Recent labs
  Questions adapt to intake classification (pediatric meds ask differently)
  Dosage follow-up: asks once with a warm phrasing, accepts "I don't know" on second attempt
         ↓
Confirm
  Natural-language paragraph summary — not a raw data table
  Patient can correct any section; system routes back to that phase
  Must explicitly confirm before proceeding
         ↓
Report generation
  Safety preflight: blocks if chief complaint, name, or clinical history missing
  LLM generates plain text clinician note
  FHIR R4 Bundle built from the same validated state
  Full report shown directly in chat (not just "click here")
  Both saved to database
  Slack notification sent to clinician channel
  FHIR Bundle posted to configured webhook URL (HMAC-signed)
  FHIR Bundle pushed directly to HAPI/Azure/Epic if FHIR_SERVER_URL is set
         ↓
Clinician receives complete note in the portal
```

## Setup

Docker and Docker Compose installed on your machine.

Set these variables in `.env` (copy from `.env.example`):
- `GEMINI_API_KEY`
- `JWT_SECRET`
- `CLINICIAN_PASSWORD`
- `SLACK_WEBHOOK_URL` (optional)

```
docker compose up --build
```

The app starts at `http://localhost:8000`. A HAPI FHIR R4 reference server starts at `http://localhost:8080` (takes ~90 seconds to boot the first time).

## Using the app

**As a patient:**

- Click New Session
- Read the AI disclosure and type yes to begin
- Answer the agent's questions naturally — no need to worry about formatting
- If you need to correct something, say so and the agent routes back to that section
- Review the full summary at the end and confirm
- The clinician note appears in the chat and is sent to the care team

**As a clinician:**

- Open `http://localhost:8000/dashboard` for the operations dashboard
- Enter your password in the Clinician Access section of the patient UI sidebar and click Auth to load case notes
- Click View Escalations to see all flagged cases
- Click an escalation to populate the resolve form, add a nurse note, and click Resolve

**Admin API (all require clinician token):**

- `GET /analytics` — operational metrics for the last 7 days including LLM cost
- `GET /analytics/summary` — dashboard summary (KPIs only)
- `POST /demo/reset` — wipes session data and re-seeds mock EHR patients
- `GET /admin/emergency-phrases` — lists active emergency phrases
- `POST /admin/emergency-phrases` — adds a new phrase, takes effect within 60 seconds
- `DELETE /admin/emergency-phrases` — removes a phrase
- `GET /clinician/webhooks` — outbound webhook delivery log
- `GET /clinician/case/{thread_id}` — full session detail: transcript, escalations, safety score

## Tech stack

FastAPI · LangGraph · Pydantic v2 · Google Gemini 2.5 Flash Lite (google-genai) · SQLite WAL · Alembic · FHIR R4 · JWT · HMAC-SHA256 · Docker · pytest · slowapi · bandit · pip-audit · Slack · Chart.js
