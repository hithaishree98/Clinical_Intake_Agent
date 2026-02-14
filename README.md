# Clinical_Intake_Agent

An AI-powered, state-driven clinical intake system designed to safely collect structured patient information, detect emergency conditions deterministically, and generate clinician-ready reports.
This project demonstrates how to responsibly integrate LLMs into a safety-sensitive workflow using deterministic control, validation, normalization, and fallback strategies.

## Why I Built This

Most AI chat systems are free-form and conversational. That approach works for general chat, but in healthcare intake, it can be unsafe.
If a patient mentions something like chest pain or difficulty breathing, the system cannot rely on probabilistic interpretation. It must respond correctly every single time.

So instead of building just a chatbot, I built a controlled intake workflow system where the goal was to combine AI flexibility with engineering-level reliability.

## What This System Does

The Clinical Intake Agent guides a patient through a structured intake flow:

1)Collects identity information

2)Verifies identity against stored records

3)Captures chief complaint and symptom details

4)Collects allergies, medications, and medical history

5)Detects emergencies immediately

6)Generates a structured intake report

7)Creates clinician review escalations when needed

The final output is a structured report ready for clinician review.

## How It Works 

### The system is built like a guided intake form with intelligence layered on top.

Instead of letting AI decide what to ask next, the system moves through clearly defined phases:

identity

identity_review

subjective

clinical_history

done

handoff (for emergency)

### Each phase is controlled by a state machine (implemented using LangGraph). This ensures:

No skipped steps

No repeated questions

No infinite loops

No accidental phase jumps

 

