"""
agentic.py — Agent-level decision helpers.

  1. Intake classification   — embedded in SubjectiveOut (single extraction call)
  2. Extraction quality      — score OPQRST completeness and build gap-fill questions
  3. Validation messaging    — targeted messages when validate_node blocks a transition
  4. Clinical question tuning — adapt questions by intake classification

Follow-up question selection is handled by the extraction LLM's own `reply` field,
which is already prompted to ask exactly one relevant question when is_complete=False.
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Clinical history question tuning by intake classification
# ---------------------------------------------------------------------------

def adapt_clinical_question(step: str, classification: str) -> str:
    """
    Return a classification-adapted clinical history question for `step`,
    or "" to fall back to the static default question.
    """
    if step == "meds" and classification == "pediatric":
        return (
            "What medications or vitamins is the patient currently taking? "
            "Please include the name, dose, how often, and when last taken — "
            "including any children's vitamins or OTC medications. If none, just say 'none'."
        )
    if step == "pmh" and classification == "mental_health":
        return (
            "Thank you for sharing. Do you have any past medical conditions, surgeries, "
            "or prior mental health diagnoses or treatments I should know about? "
            "If none, just say 'none'."
        )
    if step == "results" and classification == "mental_health":
        return (
            "Have you had any recent lab tests, imaging, or psychiatric evaluations "
            "since your last visit? If none, just say 'none'."
        )
    return ""


# ---------------------------------------------------------------------------
# Extraction quality scoring and gap-fill questions
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
# Validation gap message builder
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
