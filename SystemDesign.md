
# Architecture

User
↓
FastAPI API Layer
↓
LangGraph Workflow Engine
↓
SQLite (WAL mode) Persistence
↓
LLM (Extraction + Report)
↓
Clinician Escalation Interface

## Clinical intake has clear phases:

identity → symptoms → medications → confirm → report → done

## Using LangGraph allowed me to define:

Explicit phases

Routing logic

Interrupt points

Controlled transitions

## I intentionally separated safety logic from model reasoning.

Before any LLM call:

The system runs deterministic red-flag checks

If high-risk patterns are detected, escalation is triggered

The normal intake flow is paused

This ensures urgent routing decisions are not dependent on model interpretation.

