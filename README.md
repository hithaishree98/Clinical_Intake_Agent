# Clinical AI Workflow

A conversational patient intake system for clinical environments.

The agent guides patients through a structured intake conversation, extracts clinical information from natural language, and generates a formatted clinician note along with a FHIR R4 Bundle ready for EHR import.

## Problem Context

Most clinical intake today is either a paper form or a rigid dropdown-based tablet app. Both have the same problem that patients don't speak in structured fields. 

The gap between how patients naturally describe symptoms and how clinical documentation needs to be structured. This leads to errors like information gets lost, fields get left blank, clinicians spend time reformatting instead of treating.

## Why this approach

In a clinical setting, an LLM alone cannot be allowed to decide what to ask, when intake is complete, or whether something is serious.

A model might skip allergies, treat escalations as routine or mark intake complete while key fields are missing. These risks are unacceptable in healthcare.

The architecture of this system is specifically designed to prevent those failure modes. The LLM handles language understanding and extracting structure from natural text. 

Everything else like what to ask, what order, what constitutes an emergency, when to escalate is deterministic code. 

## What it does

- Full conversational intake covering identity, chief complaint, OPQRST symptom assessment, allergies, medications, past medical history, and recent labs.
- Emergency detection that runs before the LLM on every message. If a patient mentions chest pain or a seizure, they get immediate escalation regardless of what else they said.
- Identity verification against an EHR record where discrepancies are flagged for nurse review automatically
- Session crash recovery, if the server goes down mid-intake. The patient resumes exactly where they left off on their next message.
- Structured clinician note generated at completion
- FHIR R4 Bundle output directly compatible FHIR-compliant EHR systems.
- Clinician portal for reviewing and resolving escalations.
- Slack notifications for emergencies, identity mismatches, and completed intakes.
- Configurable emergency phrases via admin API (when clinical protocols change)

## How it works

```
Patient opens the app and clicks New Session
         ↓
Identity phase
  Agent asks for name, date of birth, phone, address
  Extracted using regex — no LLM, no silent transformation of identity data
  Name looked up in EHR — if record found, patient asked to confirm or update
  Discrepancy → escalation flagged for nurse review
         ↓
Symptom collection
  Red flag check runs on every message BEFORE the LLM
  "Chest pain", "seizure", "can't breathe" → immediate handoff, Slack alert
  If clear → LLM extracts chief complaint and OPQRST fields
  Agent asks follow-up questions until onset, severity, and chief complaint are captured
         ↓
Clinical history
  Allergies → Medications (LLM extraction) → Past medical history → Recent labs
  Each step is sequential — none can be skipped
         ↓
Confirm
  Patient sees full summary of everything collected
  Can correct any section — routes back to that phase
  Must explicitly confirm before proceeding
         ↓
Report generation
  LLM generates plain text clinician note
  FHIR R4 Bundle built from the same state data
  Both saved to database
  Slack notification fired
  Outbound webhook posted to configured URL
         ↓
Clinician receives complete note and FHIR Bundle
```

## Setup Instructions

Docker and Docker Compose installed on your machine

Set these variables in .env
- GEMINI_API_KEY
- JWT_SECRET
- CLINICIAN_PASSWORD
- SLACK_URL

### docker compose up --build

## Using the app

As a patient:

- Click New Session
- Answer the agent's questions naturally — don't worry about formatting
- At the end, review the summary and confirm
- The clinician note generates automatically

As a clinician:

- Enter your password in the Clinician Access section of the sidebar
- Click View Pending Escalations to see flagged cases
- Click an escalation to populate the resolve form
- Add a nurse note and resolve it

## Tech stack

FastAPI · LangGraph · Pydantic v2 · Gemini · SQLite WAL · FHIR R4 · JWT · Docker · pytest · slowapi
