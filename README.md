# Clinical_Intake_Agent

This project is a workflow-based clinical intake assistant built using FastAPI, LangGraph, and SQLite.

Instead of building a generic medical chatbot, I designed this as a controlled intake system that safely collects patient information step-by-step, performs deterministic triage checks, and generates a structured clinician-ready report.

A typical intake process requires:

Collecting identity information

Capturing symptoms in a standardized format

Recording medications and allergies

Detecting urgent red flags

Confirming correctness

Generating a usable report

Escalating risky cases for clinician review
