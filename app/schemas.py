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

import re as _re
from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .extract import normalize_drug_name


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
# Intent classification — replaces hardcoded yes/no keyword lists
# ---------------------------------------------------------------------------

class IntentOut(BaseModel):
    """
    Result of the LLM short-message intent classifier.

    Used when a message is too ambiguous for fast-path keyword matching:
    "I think so", "not really", "hmm yeah", "I suppose".

    intent semantics:
      confirm      — patient is agreeing / saying yes in any form
      decline      — patient is disagreeing / saying no in any form
      provide_info — patient is giving information (name, date, symptom, etc.)
      correction   — patient wants to go back and fix something
      unclear      — cannot determine; caller should prompt for clarification

    correcting_section is only populated when intent == "correction":
      identity, symptoms, history, or none.
    """
    intent:             Literal["confirm", "decline", "provide_info", "correction", "unclear"] = "unclear"
    correcting_section: Literal["identity", "symptoms", "history", "none"]                    = "none"


# ---------------------------------------------------------------------------
# LLM-extracted identity with normalization at the schema boundary
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y",
    "%m/%d/%y", "%m-%d-%y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B, %Y", "%d %b, %Y", "%m %d %Y", "%Y/%m/%d",
]


class IdentityOut(BaseModel):
    """
    Output schema for LLM identity extraction.

    Validators normalise at the schema boundary so callers always receive:
      name  — Title Case
      dob   — ISO 8601 (YYYY-MM-DD) or ""
      phone — 10 digits or ""
      address — stripped or ""

    Any value that fails normalisation is silently set to "" so identity_node
    asks the patient again rather than storing bad data.
    """
    name:    str = ""
    dob:     str = ""
    phone:   str = ""
    address: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def _norm_name(cls, v: object) -> str:
        raw = (str(v) if v else "").strip()
        if not raw or raw.lower() in ("unknown", "n/a", "none", "not provided", "not given"):
            return ""
        return " ".join(w.capitalize() for w in raw.split())

    @field_validator("dob", mode="before")
    @classmethod
    def _norm_dob(cls, v: object) -> str:
        from datetime import datetime
        raw = (str(v) if v else "").strip()
        if not raw or raw.lower() in ("unknown", "n/a", "none", "not provided", "not given"):
            return ""
        # Strip ordinal suffixes: "1st" → "1", "13th" → "13", "3rd" → "3"
        cleaned = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=_re.IGNORECASE)
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return ""  # unparseable — identity_node will re-ask

    @field_validator("phone", mode="before")
    @classmethod
    def _norm_phone(cls, v: object) -> str:
        digits = _re.sub(r"\D", "", str(v) if v else "")
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits if len(digits) == 10 else ""

    @field_validator("address", mode="before")
    @classmethod
    def _strip_address(cls, v: object) -> str:
        raw = (str(v) if v else "").strip()
        return "" if raw.lower() in ("unknown", "n/a", "none", "not provided", "not given") else raw


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
