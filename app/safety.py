"""
safety.py — Pre-report safety checks and structured escalation reason trail.

Two responsibilities:

  1. SafetyChecker.compute(state)
     Scores the session against a weighted risk rubric and enforces hard-block
     rules before report_node is allowed to generate the clinician note.
     Returns a PreflightResult with ok=False whenever any hard block fires.

  2. build_reason_trail(kind, state, ...)
     Produces a standardised, clinician-readable payload for every
     db.create_escalation() call.  Clinicians see *why* a case was escalated,
     not just that it was.

Why a dedicated module:
  nodes.py decides *when* to escalate; safety.py decides *what to record*.
  Keeping them separate means the scoring logic can be unit-tested in isolation
  (see evals/run_evals.py  human_review_threshold category).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import IntakeState


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

REVIEW_THRESHOLD: float = 50.0
"""
Sessions whose safety_score >= REVIEW_THRESHOLD are flagged for human review.
Hard-block violations (missing required fields) always block *regardless* of score.
"""


# ---------------------------------------------------------------------------
# PreflightResult
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    ok: bool                              # False → block report generation
    blocking_reasons: List[str]           # hard-block violations (empty = none)
    review_reasons: List[str]             # soft flags that raised the score
    safety_score: float                   # weighted sum (0–∞, threshold = 50)
    review_required: bool                 # score >= threshold OR hard block fired

    @property
    def all_reasons(self) -> List[str]:
        return self.blocking_reasons + self.review_reasons


# ---------------------------------------------------------------------------
# SafetyChecker
# ---------------------------------------------------------------------------

class SafetyChecker:
    """
    Scores a session and enforces hard-block rules before report generation.

    Score weights (additive):
      Hard blocks — fires a blocking_reason AND adds to score:
        chief_complaint_missing        +35
        patient_name_missing           +30
        clinical_history_incomplete    +25   (allergies/meds/PMH not collected)

      Review signals — add to score, do not block alone:
        emergency_flag_active          +50   (score alone crosses threshold)
        crisis_detected_in_session     +40
        identity_unverified            +20
        identity_mismatch_flagged      +15
        extraction_quality_low         +20
        extraction_retried             +10
        ed_mode_baseline               +10

    REVIEW_THRESHOLD = 50.
    """

    # Weight table — used both here and in eval assertions
    WEIGHTS: Dict[str, float] = {
        "chief_complaint_missing":     35.0,
        "patient_name_missing":        30.0,
        "clinical_history_incomplete": 25.0,
        "emergency_flag_active":       50.0,
        "crisis_detected_in_session":  40.0,
        "identity_unverified":         20.0,
        "identity_mismatch_flagged":   15.0,
        "extraction_quality_low":      20.0,
        "extraction_retried":          10.0,
        "ed_mode_baseline":            10.0,
    }

    @classmethod
    def compute(cls, state: "IntakeState") -> PreflightResult:
        score: float = 0.0
        blocking: List[str] = []
        review: List[str] = []

        identity = state.get("identity") or {}
        triage   = state.get("triage") or {}
        mode     = (state.get("mode") or "clinic").lower()
        cc       = (state.get("chief_complaint") or "").strip()
        q        = state.get("extraction_quality_score")
        threshold_q = 0.75 if mode == "ed" else 0.60

        # --- Hard blocks ---

        if not cc:
            score += cls.WEIGHTS["chief_complaint_missing"]
            blocking.append(
                "chief_complaint_missing: Cannot generate a clinician note without a chief complaint."
            )

        if not (identity.get("name") or "").strip():
            score += cls.WEIGHTS["patient_name_missing"]
            blocking.append(
                "patient_name_missing: Patient identity not established — report cannot be attributed."
            )

        if not state.get("clinical_complete"):
            score += cls.WEIGHTS["clinical_history_incomplete"]
            blocking.append(
                "clinical_history_incomplete: Allergy, medication, and PMH collection not completed — "
                "omitting allergies from a report is a patient safety risk."
            )

        # --- Review signals ---

        if triage.get("emergency_flag"):
            score += cls.WEIGHTS["emergency_flag_active"]
            review.append(
                "emergency_flag_active: Emergency red-flag phrase detected during intake — "
                "clinical review mandatory before discharging patient."
            )

        if state.get("crisis_detected"):
            score += cls.WEIGHTS["crisis_detected_in_session"]
            review.append(
                "crisis_detected_in_session: Patient expressed self-harm or suicidal language. "
                "Mental health review required before intake is finalised."
            )

        if (state.get("identity_status") or "unverified") != "verified":
            score += cls.WEIGHTS["identity_unverified"]
            review.append(
                "identity_unverified: Patient identity was not confirmed against the EHR record. "
                "Verify before treatment decisions are made."
            )

        if state.get("needs_identity_review"):
            score += cls.WEIGHTS["identity_mismatch_flagged"]
            review.append(
                "identity_mismatch_flagged: Patient-provided identity differs from EHR on file. "
                "A nurse should reconcile before this visit."
            )

        if q is not None and q < threshold_q:
            score += cls.WEIGHTS["extraction_quality_low"]
            review.append(
                f"extraction_quality_low: OPQRST completeness score {q:.2f} is below the "
                f"{threshold_q:.2f} threshold for {mode.upper()} mode. "
                "Key symptom fields may be missing from the note."
            )

        if int(state.get("extraction_retry_count") or 0) > 0:
            score += cls.WEIGHTS["extraction_retried"]
            review.append(
                "extraction_retried: Quality gate triggered at least one retry round — "
                "symptom data collection was incomplete on first pass."
            )

        if mode == "ed":
            score += cls.WEIGHTS["ed_mode_baseline"]
            review.append(
                "ed_mode_baseline: ED visits carry elevated baseline risk; "
                "all reports are flagged for expedited clinician review."
            )

        # Crisis always mandates human review regardless of numeric score
        crisis_override = bool(state.get("crisis_detected"))
        review_required = (score >= REVIEW_THRESHOLD) or bool(blocking) or crisis_override

        return PreflightResult(
            ok=not bool(blocking),
            blocking_reasons=blocking,
            review_reasons=review,
            safety_score=round(score, 1),
            review_required=review_required,
        )


# ---------------------------------------------------------------------------
# Structured reason trail
# ---------------------------------------------------------------------------

_SEVERITY_MAP: Dict[str, str] = {
    "emergency":       "critical",
    "crisis":          "critical",
    "report_blocked":  "high",
    "review_required": "high",
    "identity_review": "medium",
}


def build_reason_trail(
    kind: str,
    state: "IntakeState",
    override_reasons: Optional[List[str]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardised, clinician-readable escalation payload.

    Returned dict is stored as escalation payload_json and surfaced in
    GET /clinician/case/{thread_id}.  Every escalation record contains:

      kind               type of escalation
      severity           derived from kind
      triggered_at_phase phase when escalation fired
      reasons            list of reason strings (code: human text)
      safety_score       session safety score at time of escalation
      review_required    whether human review is indicated
      context            key clinical snapshot (name, CC, triage risk, …)
      [extra fields]     kind-specific data (red_flags, matched_phrases, …)
    """
    preflight = SafetyChecker.compute(state)

    # Build kind-specific reasons if none were explicitly supplied
    if override_reasons is not None:
        reasons = list(override_reasons)
    elif kind == "emergency":
        flags = (extra_data or {}).get("red_flags") or \
                (state.get("triage") or {}).get("red_flags") or []
        reasons = [f"emergency_phrase_detected: '{f}'" for f in flags] or preflight.all_reasons
    elif kind == "crisis":
        phrases = (extra_data or {}).get("matched_phrases") or []
        reasons = [f"crisis_phrase_detected: '{p}'" for p in phrases] or preflight.all_reasons
    elif kind == "identity_review":
        stored  = state.get("stored_identity") or {}
        current = state.get("identity") or {}
        diffs   = [k for k in ["name", "dob", "phone", "address"]
                   if (stored.get(k) or "") != (current.get(k) or "")]
        reasons = [f"identity_field_mismatch: '{k}' — EHR value differs from patient-provided value"
                   for k in diffs] or preflight.all_reasons
    else:
        reasons = preflight.all_reasons

    identity = state.get("identity") or {}
    op       = state.get("opqrst") or {}

    payload: Dict[str, Any] = {
        "kind":               kind,
        "severity":           _SEVERITY_MAP.get(kind, "medium"),
        "triggered_at_phase": state.get("current_phase") or "unknown",
        "reasons":            reasons,
        "safety_score":       preflight.safety_score,
        "review_required":    preflight.review_required,
        "context": {
            "patient_name":          (identity.get("name") or "unknown")[:80],
            "chief_complaint":       (state.get("chief_complaint") or "")[:300],
            "mode":                  state.get("mode") or "clinic",
            "identity_status":       state.get("identity_status") or "unverified",
            "triage_risk":           (state.get("triage") or {}).get("risk_level") or "unknown",
            "intake_classification": state.get("intake_classification"),
            "opqrst_quality_score":  state.get("extraction_quality_score"),
            "opqrst_onset":          (op.get("onset") or "")[:100],
            "opqrst_severity":       (op.get("severity") or "")[:60],
        },
    }

    if extra_data:
        # Merge extra_data but never overwrite reserved top-level keys
        reserved = {"kind", "severity", "triggered_at_phase", "reasons",
                    "safety_score", "review_required", "context"}
        for k, v in extra_data.items():
            if k not in reserved:
                payload[k] = v

    return payload
