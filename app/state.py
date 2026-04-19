from __future__ import annotations
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    text: str


Phase = Literal[
    "consent",
    "identity",
    "identity_review",
    "subjective",
    "validate",           # agentic: validation gate before phase transitions
    "clinical_history",
    "confirm",
    "report",
    "handoff",
    "done",
]


class IntakeState(TypedDict, total=False):
    # ── Session ───────────────────────────────────────────────────────────
    thread_id: str
    current_phase: Phase
    mode: Literal["clinic", "ed"]
    triage_attempts: int
    consent_given: bool

    # ── Identity ──────────────────────────────────────────────────────────
    identity: Dict[str, str]
    stored_identity: Optional[Dict[str, str]]
    identity_attempts: int
    identity_status: Literal["unverified", "verified"]
    needs_identity_review: bool

    # ── Subjective ────────────────────────────────────────────────────────
    chief_complaint: str
    opqrst: Dict[str, str]
    subjective_complete: bool
    subjective_incomplete_turns: int

    # ── Clinical history ──────────────────────────────────────────────────
    clinical_step: Literal["allergies", "meds", "pmh", "results", "done"]
    allergies: Optional[List[str]]
    medications: Optional[List[Dict[str, str]]]
    pmh: Optional[List[str]]
    recent_results: Optional[List[str]]
    clinical_complete: bool

    # ── Triage ────────────────────────────────────────────────────────────
    triage: Dict[str, Any]
    needs_emergency_review: bool

    # ── Agentic fields ────────────────────────────────────────────────────
    # Feature 1: intake classification
    intake_classification: Optional[str]   # "emergency_visit"|"routine_checkup"|"specialist_referral"|"mental_health"|"pediatric"
    classification_confidence: Optional[str]  # "high"|"medium"|"low"

    # Feature 3: extraction quality retry
    extraction_quality_score: Optional[float]  # 0.0–1.0
    extraction_retry_count: int                 # increments on each quality-gate retry

    # Feature 4: validation gate
    validation_errors: List[str]                # error keys from validate_node; empty = passed
    validation_target_phase: Optional[str]      # phase to advance to if validation passes

    # ── Safety ────────────────────────────────────────────────────────────
    crisis_detected: bool                       # True if crisis language was detected in this session
    human_review_required: bool                 # True if SafetyChecker blocked or flagged this session
    human_review_reasons: List[str]             # reasons from preflight check
    safety_score: Optional[float]               # weighted safety score from SafetyChecker
    extraction_confidence: Optional[str]        # LLM self-assessment: "high" | "medium" | "low"

    # ── Failure tracking ──────────────────────────────────────────────────
    last_failed_phase: Optional[str]            # phase where last LLM fallback/failure occurred
    last_failure_reason: Optional[str]          # brief reason string (parse_error, fallback_used, etc.)

    # ── Messages ──────────────────────────────────────────────────────────
    messages: Annotated[List[Message], operator.add]

    # ── Cost tracking ─────────────────────────────────────────────────────
    session_cost_usd: float                     # accumulated LLM spend for this session; checked against cap

    # ── Memory (Layer 2) ──────────────────────────────────────────────────
    patient_id: Optional[str]
    prior_summary: Optional[Dict[str, Any]]     # loaded once identity is known
