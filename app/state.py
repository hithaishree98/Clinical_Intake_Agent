from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from typing import Annotated
import operator

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
    thread_id: str
    current_phase: Phase
    mode: Literal["clinic", "ed"]
    triage_attempts: int
    consent_given: bool


    identity: Dict[str, str]                 
    stored_identity: Optional[Dict[str, str]]
    identity_attempts: int
    identity_status: Literal["unverified", "verified"]
    needs_identity_review: bool              


    chief_complaint: str
    opqrst: Dict[str, str]                   
    subjective_complete: bool


    clinical_step: Literal["allergies", "meds", "pmh", "results", "done"]
    allergies: List[str]
    medications: List[Dict[str, str]]       
    pmh: List[str]
    recent_results: List[str]
    clinical_complete: bool


    triage: Dict[str, Any]
    needs_emergency_review: bool

    # --- Agentic fields ---
    # Feature 1: intake classification
    intake_classification: Optional[str]    # "emergency_visit"|"routine_checkup"|"specialist_referral"|"mental_health"|"pediatric"
    classification_confidence: Optional[str]  # "high"|"medium"|"low"

    # Feature 3: extraction quality retry
    extraction_quality_score: Optional[float]  # 0.0–1.0
    extraction_retry_count: int                 # increments on each quality-gate retry

    # Feature 4: validation gate
    validation_errors: List[str]                # error keys from validate_node; empty = passed
    validation_target_phase: Optional[str]      # phase to advance to if validation passes

    # --- Safety fields ---
    crisis_detected: bool                       # True if crisis language was detected in this session
    human_review_required: bool                 # True if SafetyChecker blocked or flagged this session
    human_review_reasons: List[str]             # reasons from preflight check
    safety_score: Optional[float]               # weighted safety score from SafetyChecker
    extraction_confidence: Optional[str]        # LLM self-assessment: "high" | "medium" | "low"

    # --- Failure tracking fields ---
    last_failed_phase: Optional[str]            # phase where last LLM fallback/failure occurred
    last_failure_reason: Optional[str]          # brief reason string (parse_error, fallback_used, etc.)

    messages: Annotated[List[Message], operator.add]