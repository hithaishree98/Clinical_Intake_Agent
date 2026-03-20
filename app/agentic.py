"""
agentic.py — Agent-level decision helpers.

Implements the four agentic capabilities that elevate the system from a
deterministic state machine to a true agent:

  1. Intake classification   — classify visit type from first signals
  2. Dynamic follow-up       — select the most relevant next question
  3. Extraction quality      — score OPQRST completeness and build gap-fill questions
  4. Validation messaging    — targeted messages when validate_node blocks a transition
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .llm import run_json_step
from .schemas import ClassificationOut, FollowUpStrategyOut
from .prompts import classification_system, followup_strategy_system


# ---------------------------------------------------------------------------
# Feature 1: Intake classification
# ---------------------------------------------------------------------------

_VALID_CLASSIFICATIONS = {
    "emergency_visit",
    "routine_checkup",
    "specialist_referral",
    "mental_health",
    "pediatric",
}


def classify_intake(mode: str, cc: str, user_text: str) -> Tuple[str, str, dict]:
    """
    Classify the intake type from mode, chief complaint, and first user message.

    Returns (classification, confidence, meta).
    The caller should inspect meta["fallback_used"] to decide whether to log a failure.
    Falls back to ("routine_checkup", "low") on any LLM failure so that
    downstream logic always has a valid classification.
    """
    prompt = f"MODE={mode}\nCHIEF_COMPLAINT={cc}\nUSER_TEXT={user_text}"
    obj, meta = run_json_step(
        system=classification_system(),
        prompt=prompt,
        schema=ClassificationOut,
        fallback={"intake_classification": "routine_checkup", "confidence": "low", "rationale": ""},
        temperature=0.2,
    )
    out = obj.model_dump()
    classification = out.get("intake_classification") or "routine_checkup"
    if classification not in _VALID_CLASSIFICATIONS:
        classification = "routine_checkup"
    confidence = out.get("confidence") or "low"
    return classification, confidence, meta


# ---------------------------------------------------------------------------
# Feature 2: Dynamic follow-up strategy selection
# ---------------------------------------------------------------------------

def select_followup(intake_classification: str, mode: str, cc: str, opqrst: Dict[str, str]) -> Tuple[str, dict]:
    """
    Select the most clinically relevant follow-up question given current state.

    Returns (question, meta).
    question is "" when the fallback fired so the caller falls back to the LLM's
    own reply from the extraction step.
    The caller should inspect meta["fallback_used"] to decide whether to log a failure.
    """
    prompt = (
        f"INTAKE_CLASSIFICATION={intake_classification}\n"
        f"MODE={mode}\n"
        f"CHIEF_COMPLAINT={cc}\n"
        f"OPQRST_CURRENT={opqrst}"
    )
    obj, meta = run_json_step(
        system=followup_strategy_system(intake_classification, mode),
        prompt=prompt,
        schema=FollowUpStrategyOut,
        fallback={"priority_fields": [], "next_question": "", "rationale": ""},
        temperature=0.3,
    )
    return (obj.model_dump().get("next_question") or "").strip(), meta


def adapt_clinical_question(step: str, classification: str) -> str:
    """
    Return a classification-adapted clinical history question for `step`,
    or "" to use the static default question.
    """
    if step == "meds" and classification == "pediatric":
        return (
            "What medications or vitamins is the patient currently taking? "
            "Include the name, dose, how often, and when last taken. "
            "Include any children's vitamins or OTC medications. If none, say 'none'."
        )
    if step == "pmh" and classification == "mental_health":
        return (
            "Any past medical conditions, surgeries, or prior mental health diagnoses "
            "or treatments? If none, say 'none'."
        )
    if step == "results" and classification == "mental_health":
        return (
            "Any recent lab tests, imaging, or psychiatric evaluations since your last visit? "
            "If none, say 'none'."
        )
    return ""


# ---------------------------------------------------------------------------
# Feature 3: Extraction quality scoring and gap-fill questions
# ---------------------------------------------------------------------------

def score_extraction_quality(cc: str, opqrst: Dict[str, str]) -> float:
    """
    Score OPQRST completeness on a 0.0–1.0 scale.

    Weights reflect clinical importance:
      - chief_complaint: 0.25  (prerequisite for everything)
      - onset:           0.20  (when it started — critical)
      - severity:        0.20  (how bad — critical for triage)
      - quality:         0.10
      - timing:          0.10
      - provocation:     0.075
      - radiation:       0.075
    A score >= 0.65 means cc + onset + severity are all present, which is
    the practical minimum for a useful intake note.
    """
    score = 0.25 if (cc or "").strip() else 0.0
    field_weights = {
        "onset":       0.20,
        "severity":    0.20,
        "quality":     0.10,
        "timing":      0.10,
        "provocation": 0.075,
        "radiation":   0.075,
    }
    for field, weight in field_weights.items():
        if (opqrst.get(field) or "").strip():
            score += weight
    return round(min(score, 1.0), 3)


def build_gap_fill_question(cc: str, opqrst: Dict[str, str], classification: str) -> str:
    """
    Build a targeted gap-fill question for the most critical missing OPQRST field.
    Deterministic — no LLM call, keeping the retry path fast.
    Priority order: severity → onset → quality → timing → radiation → provocation.
    """
    subject = cc or "symptom"
    priority = ["severity", "onset", "quality", "timing", "radiation", "provocation"]
    for field in priority:
        if not (opqrst.get(field) or "").strip():
            if field == "severity":
                return f"How severe is the {subject} on a scale of 0 to 10?"
            if field == "onset":
                return f"When did the {subject} start?"
            if field == "quality":
                return f"How would you describe the {subject}? (e.g., sharp, dull, burning, pressure)"
            if field == "timing":
                return f"Is the {subject} constant, or does it come and go?"
            if field == "radiation":
                return f"Does the {subject} spread or radiate anywhere?"
            if field == "provocation":
                return f"What makes the {subject} better or worse?"
    return f"Can you tell me anything else about the {subject}?"


# ---------------------------------------------------------------------------
# Feature 4: Validation gap message builder
# ---------------------------------------------------------------------------

def build_validation_gap_message(errors: List[str], cc: str, mode: str) -> str:
    """
    Produce a patient-facing message explaining what information is still needed
    before the phase transition is allowed.
    """
    parts = []
    if "chief_complaint" in errors:
        parts.append("I still need to know what brought you in today.")
    if "opqrst_incomplete" in errors or "severity_required" in errors:
        subject = cc or "symptom"
        if mode == "ed":
            parts.append(
                f"For an emergency visit I need the severity (0–10) and when the {subject} started."
            )
        else:
            parts.append(
                f"I need a bit more detail about your {subject} — when it started and how severe it is."
            )
    if "allergies_not_collected" in errors:
        parts.append("I still need to ask about your allergies before we continue.")
    if "extraction_confidence_low" in errors:
        subject = cc or "symptom"
        parts.append(
            f"I want to make sure I captured your {subject} details correctly — "
            "could you briefly confirm when it started and how severe it is?"
        )
    if "clinical_history_incomplete" in errors:
        parts.append("There are a few clinical history questions we haven't finished yet.")
    return " ".join(parts) if parts else "I need a bit more information before we continue."
