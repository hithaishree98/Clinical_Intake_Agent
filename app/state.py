from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from typing import Annotated
import operator

class Message(TypedDict):
    role: Literal["user", "assistant"]
    text: str

Phase = Literal[
    "identity",
    "identity_review",
    "subjective",
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


    messages: Annotated[List[Message], operator.add]
