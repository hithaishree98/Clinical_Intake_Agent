"""schemas.py — Pydantic output schemas for every LLM step."""
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
    """LLM Tier-2 crisis classifier result. low confidence → log only, no escalation."""
    is_crisis_risk: bool                                                  = False
    confidence:     Literal["high", "medium", "low"]                     = "low"
    reasoning:      Annotated[str, Field(default="", max_length=300)]    = ""


# ---------------------------------------------------------------------------
# Intent classification — replaces hardcoded yes/no keyword lists
# ---------------------------------------------------------------------------

class IntentOut(BaseModel):
    """LLM short-message intent classifier result."""
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
    """LLM identity extraction. Validators normalise to Title Case / ISO 8601 / 10-digit phone."""
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

        # Refuse genuinely ambiguous numeric dates.  "01/02/1990" could be
        # 02 Jan (US) or 01 Feb (everywhere else); silently picking m/d
        # has put wrong DOBs in the EHR.  When both leading numbers are
        # in [1,12] AND the format is purely numeric, return "" so
        # identity_node re-asks ("any format works, like '15 March 1985'").
        # 4-digit-leading (ISO) and any string containing a month name are
        # unambiguous and pass through.
        ambig = _re.match(r"^(\d{1,2})[\/\-](\d{1,2})[\/\-]\d{2,4}$", cleaned)
        if ambig:
            a, b = int(ambig.group(1)), int(ambig.group(2))
            if 1 <= a <= 12 and 1 <= b <= 12 and a != b:
                return ""

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
        if not raw or raw.lower() in ("unknown", "n/a", "none", "not provided", "not given"):
            return ""
        # Require a 5-digit zip code as a proxy for a complete address.
        # Addresses missing a zip (street only, city+state only) are returned
        # as "" so identity_node re-asks for the full address.
        if not _re.search(r"\b\d{5}(-\d{4})?\b", raw):
            return ""
        return raw


# ---------------------------------------------------------------------------
# Generic clinical list extraction (allergies, PMH, recent results)
# ---------------------------------------------------------------------------

class ListExtractOut(BaseModel):
    """Shared schema for allergy / PMH / results list extraction."""
    items: List[Annotated[str, Field(max_length=200)]]            = Field(default_factory=list)
    items_complete: bool                                          = True
    reply: Annotated[str, Field(default="", max_length=400)]      = ""

    @model_validator(mode="after")
    def _strip_blanks_and_cap(self) -> "ListExtractOut":
        """Drop empty entries, dedup by case-insensitive match, cap to 30."""
        seen: set[str] = set()
        cleaned: List[str] = []
        for it in self.items:
            v = (it or "").strip()
            k = v.lower()
            if v and k not in seen:
                seen.add(k)
                cleaned.append(v)
        self.items = cleaned[:30]
        return self


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
    """Validated state snapshot consumed by report_node and fhir_builder."""
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
