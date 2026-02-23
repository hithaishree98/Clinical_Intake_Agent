# Clinical_Intake_Agent

This project is a workflow-based clinical intake assistant built using FastAPI, LangGraph, and SQLite.

Instead of building a generic medical chatbot, I designed this as a controlled intake system that safely collects patient information step-by-step, performs deterministic triage checks, and generates a structured clinician-ready report.

## A typical intake process requires:

- Collecting identity information

- Capturing symptoms in a standardized format

- Recording medications and allergies

- Detecting urgent red flags

- Confirming correctness

- Generating a usable report

- Escalating risky cases for clinician review

## Application Workflow

### The intake follows clear phases:

### Identity Phase

Collects name, DOB, phone, and address.

Asks only for missing fields.

### Subjective Phase

Captures chief complaint.

Extracts OPQRST symptom details using structured model prompts.

### Medications & Allergies Phase

Extracts medication lists and allergy information.

Uses deterministic parsing first, LLM normalization second.

### Confirmation Phase

Shows a structured summary.

Asks for confirmation.

Allows corrections before proceeding.

### Report Phase

Generates a structured clinician-ready report asynchronously.

Stores it for retrieval.

### Escalation (If Needed)

If red-flag symptoms are detected, the case is escalated.

Clinicians can review the case and resolve it.

## How a User Would Use It

- The user starts a session.

- The assistant asks for basic identity information.

- The user describes symptoms in natural language.

- The assistant extracts structured details.

- The assistant asks one clarifying question at a time.

- The user confirms the summary.

- A final report is generated.

- If urgent symptoms were detected, the case is escalated.

The experience feels conversational, but under the hood it is controlled by a structured workflow.

## Key Features

Deterministic red-flag triage before LLM reasoning

Schema-validated structured extraction

Idempotent chat handling (safe retries)

Session checkpointing and resumability

Asynchronous report generation

Escalation + clinician review workflow

Persistent audit trail
