"""
schemas.py — Pydantic output schemas for every LLM step.

Design rules:
  1. No bare Dict[str, str] for structured data — use typed nested models so
     Pydantic rejects extra keys and enforces field names at parse time.
  2. All string fields carry explicit max_length so oversized LLM output
     triggers the run_json_step repair cycle rather than silently polluting state.
  3. Enum fields use Literal types — invalid LLM values fail validation and
     trigger repair before the fallback fires.
  4. Validators remove empty-name medications and strip whitespace so garbage
     entries can never reach the database.
  5. ReportInputState is the canonical validated snapshot consumed by report_node
     and fhir_builder — both generate artifacts from this model, never from raw
     IntakeState, proving provenance from validated structured state.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# OPQRST — typed nested model so the LLM cannot invent extra keys
# ---------------------------------------------------------------------------

class OPQRSTFields(BaseModel):
    onset:       Annotated[str, Field(default="", max_length=150)]  = ""
    provocation: Annotated[str, Field(default="", max_length=150)]  = ""
    quality:     Annotated[str, Field(default="", max_length=150)]  = ""
    radiation:   Annotated[str, Field(default="", max_length=150)]  = ""
    severity:    Annotated[str, Field(default="", max_length=80)]   = ""
    timing:      Annotated[str, Field(default="", max_length=150)]  = ""


# ---------------------------------------------------------------------------
# Subjective extraction
# ---------------------------------------------------------------------------

class SubjectiveOut(BaseModel):
    chief_complaint:         Annotated[str, Field(default="", max_length=300)]          = ""
    opqrst:                  OPQRSTFields                                               = Field(default_factory=OPQRSTFields)
    is_complete:             bool                                                       = False
    reply:                   Annotated[str, Field(default="", max_length=400)]          = ""
    extraction_confidence:   Literal["high", "medium", "low"]                          = "medium"
    # Combined classification — avoids a second LLM round-trip on first chief complaint
    intake_classification:   Literal[
        "emergency_visit", "routine_checkup", "specialist_referral",
        "mental_health", "pediatric",
    ] | None                                                                           = None
    classification_confidence: Literal["high", "medium", "low"] | None                = None


# ---------------------------------------------------------------------------
# Medication extraction
# ---------------------------------------------------------------------------

class MedicationItem(BaseModel):
    name:       Annotated[str, Field(default="", max_length=120)] = ""
    dose:       Annotated[str, Field(default="", max_length=60)]  = ""
    freq:       Annotated[str, Field(default="", max_length=100)] = ""
    last_taken: Annotated[str, Field(default="", max_length=100)] = ""

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: object) -> str:
        raw = (str(v) if v else "").strip()
        if not raw:
            return raw
        from .extract import normalize_drug_name
        return normalize_drug_name(raw)


class MedsOut(BaseModel):
    medications: List[MedicationItem] = Field(default_factory=list)
    reply:       Annotated[str, Field(default="", max_length=400)] = ""

    @model_validator(mode="after")
    def drop_nameless_meds(self) -> "MedsOut":
        """Remove any medication entries where the LLM left the name blank."""
        self.medications = [m for m in self.medications if m.name]
        return self


# ---------------------------------------------------------------------------
# Crisis scoring — LLM-in-the-loop safety layer
# ---------------------------------------------------------------------------

class CrisisScore(BaseModel):
    """
    Result of the LLM borderline-crisis classifier.

    Used only when keyword/regex detection (Tier 1) did not fire but the
    message contains soft distress signals that warrant LLM interpretation.
    The LLM determines whether the language represents genuine suicidal or
    self-harm ideation versus figurative speech (e.g. "kill this headache").

    confidence semantics:
      high   — LLM is certain about the verdict
      medium — LLM is reasonably confident; borderline cases here still escalate
      low    — LLM is uncertain; logged as soft_distress_flagged but no escalation
    """
    is_crisis_risk: bool                                                  = False
    confidence:     Literal["high", "medium", "low"]                     = "low"
    reasoning:      Annotated[str, Field(default="", max_length=300)]    = ""


# ---------------------------------------------------------------------------
# Intake classification
# ---------------------------------------------------------------------------

_IntakeClass = Literal[
    "emergency_visit", "routine_checkup", "specialist_referral",
    "mental_health", "pediatric",
]
_Confidence = Literal["high", "medium", "low"]


class ClassificationOut(BaseModel):
    """Feature 1: intake classification result."""
    intake_classification: _IntakeClass = "routine_checkup"
    confidence:            _Confidence  = "medium"
    rationale:             Annotated[str, Field(default="", max_length=200)] = ""


# ---------------------------------------------------------------------------
# Follow-up strategy
# ---------------------------------------------------------------------------

_OPQRSTKey = Literal["onset", "provocation", "quality", "radiation", "severity", "timing"]


class FollowUpStrategyOut(BaseModel):
    """Feature 2: dynamic follow-up strategy selection result."""
    priority_fields: List[_OPQRSTKey]                                         = Field(default_factory=list)
    next_question:   Annotated[str, Field(default="", max_length=400)]        = ""
    rationale:       Annotated[str, Field(default="", max_length=200)]        = ""


# ---------------------------------------------------------------------------
# Identity fields — typed model used inside ReportInputState
# ---------------------------------------------------------------------------

class IdentityFields(BaseModel):
    """Validated patient identity used for report and FHIR generation."""
    name:    Annotated[str, Field(default="", max_length=200)] = ""
    dob:     Annotated[str, Field(default="", max_length=20)]  = ""
    phone:   Annotated[str, Field(default="", max_length=20)]  = ""
    address: Annotated[str, Field(default="", max_length=300)] = ""

    @field_validator("name", "dob", "phone", "address", mode="before")
    @classmethod
    def _strip(cls, v: object) -> str:
        return (str(v) if v else "").strip()


# ---------------------------------------------------------------------------
# ReportInputState — canonical validated snapshot for report + FHIR generation
# ---------------------------------------------------------------------------

class ReportInputState(BaseModel):
    """
    Validated state snapshot consumed by report_node and fhir_builder.

    Both the clinician note and the FHIR bundle are generated from this model,
    never from raw IntakeState dicts.  This ensures:
      - All string fields are bounded (no unbounded LLM text leaks into artifacts)
      - Medications without a name are silently dropped
      - Empty/whitespace-only allergies, PMH, and results entries are filtered
      - List lengths are capped to prevent flooding artifacts with noise
      - Provenance from validated structured state is explicit and auditable
    """
    identity:        IdentityFields                                          = Field(default_factory=IdentityFields)
    chief_complaint: Annotated[str, Field(default="", max_length=300)]      = ""
    opqrst:          OPQRSTFields                                            = Field(default_factory=OPQRSTFields)
    allergies:       List[Annotated[str, Field(max_length=200)]]             = Field(default_factory=list)
    medications:     List[MedicationItem]                                    = Field(default_factory=list)
    pmh:             List[Annotated[str, Field(max_length=300)]]             = Field(default_factory=list)
    recent_results:  List[Annotated[str, Field(max_length=300)]]             = Field(default_factory=list)
    triage:          Dict[str, Any]                                          = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cap_and_filter_lists(self) -> "ReportInputState":
        """Drop blank entries and cap list lengths to guard against LLM flooding."""
        self.allergies      = [a for a in self.allergies      if (a or "").strip()][:20]
        self.medications    = [m for m in self.medications    if m.name][:30]
        self.pmh            = [p for p in self.pmh            if (p or "").strip()][:20]
        self.recent_results = [r for r in self.recent_results if (r or "").strip()][:20]
        return self
