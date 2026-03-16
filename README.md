# Clinical AI Workflow

A state-machine-driven conversational agent that guides patients through structured clinical intake, extracts information from natural language, triages urgency, and outputs a clinician note and FHIR R4 health record.

Built on a core principle: LLM handles language, state machine handles control. 

The AI extracts what patients say, it cannot skip phases, override clinical logic, or advance the workflow without deterministic validation at every step.

## Problem Context

Most clinical intake today is a paper form or a dropdown-based tablet app. Both share the same flaw that patients don't speak in structured fields.

A patient says for example "it's been hurting on and off since Tuesday, worse when I breathe in". In this case a dropdown gives them "chest pain: yes/no". Information gets lost, fields are left blank, and clinicians spend time reformatting notes. And a static form can't detect mid-conversation that a patient is describing a medical emergency and flag it immediately.

This project bridges that gap by converting natural patient language into structured clinical documentation while continuously monitoring for urgency, without forcing patients to adapt to the system.

## Why this approach

In a clinical setting, an LLM alone cannot be allowed to decide what to ask, when intake is complete or whether something is serious. A model might skip allergies, treat escalations as routine or mark intake complete while key fields are missing. In healthcare, these are not acceptable failure modes.

The architecture of this system is specifically designed to prevent those failure modes. The LLM handles language understanding and extracting structure from natural text. 

Everything else like what to ask, what order, what constitutes an emergency, when to escalate is deterministic code. 

## What it does

- Full conversational intake covering identity, chief complaint, OPQRST symptom assessment, allergies, medications, past medical history, and recent labs.
- Emergency detection that runs before the LLM during the symptom collection phase. If a patient mentions chest pain or a seizure, they get immediate escalation regardless of what else they said.
- Patients confirm AI-assisted intake before any data is collected.
- Identity verification against an EHR record where discrepancies are flagged for nurse review automatically.
- If the server goes down mid-intake, the patient resumes exactly where they left off.
- Generates a structured clinician note and a FHIR R4 Bundle compatible with FHIR-compliant EHR systems.
- Clinician portal for reviewing and resolving escalations.
- Slack notifications for emergencies, crisis language, and completed intakes. FHIR Bundle posted via HMAC-signed webhook on completion.

## How it works

```
Patient opens the app and clicks New Session
         ↓
Consent
  Patient is shown an AI disclosure before any data is collected
  Must explicitly agree to continue — declining ends the session
         ↓
Identity
  Agent asks for name, date of birth, phone, address
  Extracted using regex. No LLM or no transformation of identity data
  Name looked up in EHR. If record found, patient asked to confirm or update
  Discrepancy → escalation flagged for nurse review
         ↓
Symptom collection
  Emergency check runs on every message
  "Chest pain", "seizure", "can't breathe" → immediate escalation
  Crisis check also runs. If self-harm phrases → 988 Lifeline response
  If clear → LLM extracts chief complaint and OPQRST fields
  Agent asks follow-up questions until all key fields are captured
         ↓
Clinical history
  Allergies → Medications → Past medical history → Recent labs
  Each step is sequential so none can be skipped
         ↓
Confirm
  Patient reviews full summary of everything collected
  Can correct any section, routes back to that phase
  Must explicitly confirm before proceeding
         ↓
Report generation
  LLM generates plain text clinician note
  FHIR R4 Bundle built from the same state data
  Both saved to database
  Slack notification sent to clinician channel
  FHIR Bundle posted to configured webhook URL (HMAC-signed)
         ↓
Clinician receives complete note and FHIR Bundle
```

## Setup Instructions

Docker and Docker Compose installed on your machine

Set these variables in .env
- GEMINI_API_KEY
- JWT_SECRET
- CLINICIAN_PASSWORD
- SLACK_WEBHOOK_URL

### docker compose up --build

## Using the app

As a patient:

- Click New Session
- Read the AI disclosure and type yes to begin
- Answer the agent's questions naturally, no need to worry about formatting
- If you need to correct something, say so the agent will route back to that section
- Review the full summary at the end and confirm
- The clinician note generates automatically

As a clinician:

- Enter your password in the Clinician Access section of the sidebar and click Auth
- Click Pull Clinician Note to load the report for the current session
- Click View Escalations to see all flagged cases
- Click an escalation to populate the resolve form
- Add a nurse note and click Resolve

Admin API (all require clinician token):

- GET /analytics — operational metrics for the last 7 days
- POST /demo/reset — wipes session data and re-seeds mock EHR patients
- GET /admin/emergency-phrases — lists active emergency phrases
- POST /admin/emergency-phrases — adds a new phrase, takes effect immediately
- DELETE /admin/emergency-phrases — removes a phrase
  
## Tech stack

FastAPI · LangGraph · Pydantic v2 · Google Gemini (google-genai) · SQLite WAL · FHIR R4 · JWT · HMAC-SHA256 · Docker · pytest · slowapi · Slack
