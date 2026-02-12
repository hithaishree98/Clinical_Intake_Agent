# System Design

The system is organized into clear layers:

## API Layer (FastAPI)

Handles:

Incoming user messages

Session retrieval

Routing to the workflow engine

Returning structured responses

Viewing reports and escalations

This separation keeps HTTP concerns separate from workflow logic.

## Workflow Engine (LangGraph State Machine)

This is the core controller.

Each intake phase is implemented as a node that:

Reads current state

Extracts structured data

Updates state

Determines the next phase

Using a state machine guarantees controlled progression and predictable behavior.

## Deterministic Safety Layer

Before the LLM is called, the system runs emergency detection using:

Regex patterns

Keyword matching

Red-flag phrase checks

If emergency symptoms are detected:

Session status becomes escalated

Phase switches to handoff

An escalation record is created

Chat progression stops

Emergency handling is deterministic by design.

## LLM Layer (Structured Extraction Only)

The LLM is used for:

Extracting chief complaint

Parsing onset and severity

Extracting medications and allergies

Updating structured patient state

It does not control routing or safety decisions.

The LLM acts as a structured data parser.

## Validation & Normalization Layer

LLM outputs are never trusted blindly.

Every response must:

Return valid JSON

Match a strict Pydantic schema

Pass validation

After validation, data is normalized:

Phone numbers cleaned using regex

Risk levels mapped to controlled enums

Confidence scores clamped to valid ranges

Duplicate medications removed

Visit types standardized

Normalization prevents inconsistent data from breaking logic or storage.

## Retry & Fallback Mechanisms

LLM calls are wrapped in:

Exponential backoff retry logic

Maximum retry limits

Graceful fallback prompts

If parsing fails:

The system retries

Or asks the user for clarification

This prevents transient failures from corrupting the workflow.

## Idempotency & Replay Protection

The system protects against duplicate processing.

If a message is replayed (network retry or browser refresh):

The system detects duplicate message ID

Returns the stored response

Does not progress the workflow again

This prevents:

Duplicate reports

Double escalations

State corruption

## Persistence Layer (SQLite)

SQLite stores:

Sessions

Messages

Escalations

Reports

Mock EHR records

Configured with:

WAL mode for concurrency

Busy timeout to avoid locking errors

SQLite was chosen for simplicity in a prototype. For production, PostgreSQL would be used.

## Safety Principles

This system follows key safety principles:

Deterministic emergency detection

Strict schema validation of AI output

Controlled phase transitions

Data normalization before storage

Retry + fallback handling

Idempotent message processing

AI is allowed to assist — but never to control safety-critical decisions.

## Scalability Considerations:

This is a prototype demo. If scaling this system for prod:

Replace SQLite with PostgreSQL - PostgreSQL supports true concurrent writes, better indexing, and horizontal scaling, making it more suitable for production workloads with multiple users.

Add Redis for Session Caching - Redis can cache active session state to reduce database load and improve response times for high-traffic, real-time conversations.

Introduce Background Job Queues - Job queues allow heavy tasks like report generation or LLM retries to run asynchronously without blocking user responses.

Implement Encryption - Encryption ensures sensitive patient data is protected both while stored in the database.

Add Audit Logging and RBAC - Audit logs provide traceability for medical actions, and role-based access control ensures only authorized users can view or modify sensitive data.

Add Observability and Metrics Tracking - Monitoring metrics like escalation rates, retry counts, and completion times helps detect failures early and improve system reliability at scale.
