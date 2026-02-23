
# System Design & Architecture

The system has four main layers:

- API Layer (FastAPI)

- Workflow Layer (LangGraph state machine)

- Business Logic Nodes

- Persistence Layer (SQLite + WAL)

- LLM Integration Layer

Everything revolves around a thread_id, which represents a single intake session.

## The intake is modeled as a state machine:

### identity → subjective → meds → confirm → report → done

Each node:

Receives structured state

Updates missing fields

Returns next routing decision

The graph uses SQLite checkpointing to persist state between turns.

# Node-Level Design

## Identity Node 

Collects the minimum identity details needed for intake and ensures they’re clean + consistent.

This is handled by Python function. No hallucination risk because parsing is deterministic. Normalization reduces downstream inconsistencies.

Fallback:

- If fields missing, ask exactly one clarifying question.

## Triage Node

Runs urgent symptom detection and routes away from normal intake if risk is detected.

Python script handles rule/pattern based detection, routing decision: “continue intake” vs “escalate” and escalation DB write: insert row into escalations

Fallback:

- If text is ambiguous but matches risk pattern → bias safe (escalate or clarify)

- If escalation created, flow doesn’t proceed to report

## Subjective/OPQRST Node

Transforms unstructured symptom description into structured clinical fields (OPQRST).

Here Python script:
- decides whether OPQRST is “complete enough” or which field is missing

- decides whether a clarifying question is needed (exactly one)

- validates + merges extracted fields into state

- doesn’t overwrite stable fields unless user explicitly corrects

Here LLM handles:

- Structured extraction call via run_json_step(...)

- Outputs JSON matching schema

Safety Measure followed:

- Strict prompt rules:

JSON only

no diagnosis / no advice

don’t invent missing info

- Schema validation (Pydantic): rejects malformed output

- Merge rules: safe update into state

Fallback:

- If LLM output is invalid JSON, retry once with stricter instruction

- If still invalid, ask a deterministic clarifying question instead

- If user gives incomplete symptoms, ask one missing OPQRST dimension

## Medications + Allergies Node

Captures meds and allergies as structured lists.

Python handles:

- extract_allergies_simple(...): pulls obvious allergies from text

- extract_list_simple(...): parses comma/line separated med lists

- detects “no meds” / “no allergies” using is_no, is_ack

- preserves previously correct lists

LLM handles:

- Used selectively for normalization / cleanup when text is messy.

- converting “metformin 500mg twice daily” into structured form

- separating combined lists (“ibuprofen, no allergies, sometimes Zyrtec”)

Safety measures:

- Prompt forbids inventing medications/allergies

- Schema validation ensures output is a list in correct format

- Conservative merging: don’t overwrite valid meds list with weaker output

Fallback:

- If extraction unclear, ask one question:

“Are you currently taking any medications?”

- If LLM JSON invalid, retry once, else fall back to deterministic question flow

## Confirmation Node 

Summarizes captured data and asks patient to confirm before report generation.

Python handles:

- builds a summary string from structured fields

- detects user response:

is_yes(...)

is_no(...)

is_ack(...)

- routes:

yes, proceed to report phase

no, enter edit path and ask what to correct

Safety measures:

- Graph interrupt ensures report cannot generate before confirmation

- Confirmation parsing is deterministic (not LLM-based)

Fallback:

If user response is unclear, ask exactly one clarification:

“Should I generate the report, or would you like to change something?”

## Report Generation Node (Async)

Creates the final clinician-ready report from structured state.

Python handles:

- schedules async job in jobs table

- updates job status: queued → running → done/failed

- persists final report to reports table

- ensures failures are visible (not silent)

LLM handles:

- Generates the report text (summary) using a report-specific prompt

- Input is mostly structured fields (identity, OPQRST, meds, allergies, PMH)

Safety measures:

- Prompt constraints:

no diagnosis / no medical advice

“summarize what patient reported”

don’t invent facts

Fallback:

- If report LLM call fails, job marked failed with error

- clinician can still view structured data + chat log

- system doesn’t lose session state

## Escalation / Clinician Review Path

Supports human-in-the-loop review when risk is detected.

Python handles:

- creates escalation record with reason and status

- clinician endpoints:

list pending

open case view (chat + snapshot + report)

resolve escalation

Safety measures:

Clear boundary: the system escalates instead of improvising

Auditability: escalation reason is persisted

Fallback:

Even if report generation fails, clinician can review the case from stored messages/snapshots


