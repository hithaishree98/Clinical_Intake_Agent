"""
Builds a FHIR R4 Bundle from completed intake state.

Resources: Patient, Condition, AllergyIntolerance, MedicationStatement, Observation.
No external FHIR library needed — resources are plain dicts serialized as JSON.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, List


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_dob(dob: str) -> str:
    """Convert MM/DD/YYYY or MM-DD-YYYY to YYYY-MM-DD (FHIR R4 date format)."""
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", dob.strip())
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return dob


def _entry(resource: dict) -> dict:
    return {
        "fullUrl": f"urn:uuid:{resource['id']}",
        "resource": resource,
    }


def _patient(patient_id: str, identity: dict) -> dict:
    name = (identity.get("name") or "").strip()
    parts = name.split() if name else []
    family = parts[-1] if len(parts) > 1 else name
    given = parts[:-1] if len(parts) > 1 else []

    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{"use": "official", "family": family, "given": given, "text": name or "Unknown"}],
        "telecom": [],
        "address": [],
    }

    phone = (identity.get("phone") or "").strip()
    if phone:
        patient["telecom"].append({"system": "phone", "value": phone, "use": "mobile"})

    address = (identity.get("address") or "").strip()
    if address:
        patient["address"].append({"use": "home", "text": address})

    dob = (identity.get("dob") or "").strip()
    if dob:
        patient["birthDate"] = _normalize_dob(dob)

    return patient


def _condition(patient_id: str, state: dict) -> dict:
    cc = (state.get("chief_complaint") or "").strip()
    op = state.get("opqrst") or {}

    note_lines = [f"Chief complaint: {cc}"] if cc else []
    for label, key in [
        ("Onset", "onset"),
        ("Provocation", "provocation"),
        ("Quality", "quality"),
        ("Radiation", "radiation"),
        ("Severity", "severity"),
        ("Timing", "timing"),
    ]:
        value = (op.get(key) or "").strip()
        if value:
            note_lines.append(f"{label}: {value}")

    return {
        "resourceType": "Condition",
        "id": _uid(),
        "subject": {"reference": f"urn:uuid:{patient_id}"},
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
                "display": "Active",
            }]
        },
        "code": {"text": cc or "Not provided"},
        "note": [{"text": "\n".join(note_lines)}] if note_lines else [],
    }


def _allergy(patient_id: str, allergy_text: str) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": _uid(),
        "patient": {"reference": f"urn:uuid:{patient_id}"},
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                "code": "active",
                "display": "Active",
            }]
        },
        "code": {"text": allergy_text.strip()},
    }


def _medication_statement(patient_id: str, med: dict) -> dict:
    name = (med.get("name") or "Unknown").strip()
    dose = (med.get("dose") or "").strip()
    freq = (med.get("freq") or "").strip()
    last_taken = (med.get("last_taken") or "").strip()

    note_parts = []
    if dose:
        note_parts.append(f"Dose: {dose}")
    if freq:
        note_parts.append(f"Frequency: {freq}")
    if last_taken:
        note_parts.append(f"Last taken: {last_taken}")

    stmt: dict[str, Any] = {
        "resourceType": "MedicationStatement",
        "id": _uid(),
        "subject": {"reference": f"urn:uuid:{patient_id}"},
        "status": "active",
        "medicationCodeableConcept": {"text": name},
    }
    if note_parts:
        stmt["note"] = [{"text": ", ".join(note_parts)}]

    return stmt


def _triage_observation(patient_id: str, triage: dict) -> dict:
    risk_level = (triage.get("risk_level") or "unknown").strip()
    visit_type = (triage.get("visit_type") or "unknown").strip()
    rationale = (triage.get("rationale") or "").strip()
    red_flags = triage.get("red_flags") or []

    note_text = f"Visit type: {visit_type}"
    if rationale:
        note_text += f". {rationale}"
    if red_flags:
        note_text += f" Red flags: {', '.join(red_flags)}."

    return {
        "resourceType": "Observation",
        "id": _uid(),
        "subject": {"reference": f"urn:uuid:{patient_id}"},
        "status": "preliminary",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "11283-9", "display": "Acuity assessment"}],
            "text": "Triage assessment",
        },
        "valueString": risk_level,
        "note": [{"text": note_text}] if note_text else [],
    }


def validate_fhir_input(state: dict) -> List[str]:
    """
    Validate that state contains the fields needed for a meaningful FHIR bundle.

    Returns a list of warning codes (empty = all clear).  The caller logs
    these before calling build_bundle so every gap is observable.

    Codes:
      fhir_missing_patient_name    — identity.name is blank or absent
      fhir_missing_dob             — identity.dob is blank or absent
      fhir_missing_chief_complaint — chief_complaint is blank or absent
      fhir_allergies_not_collected — allergies list is None (step not reached)
    """
    warnings: List[str] = []
    identity = state.get("identity") or {}
    if not (identity.get("name") or "").strip():
        warnings.append("fhir_missing_patient_name")
    if not (identity.get("dob") or "").strip():
        warnings.append("fhir_missing_dob")
    if not (state.get("chief_complaint") or "").strip():
        warnings.append("fhir_missing_chief_complaint")
    if state.get("allergies") is None:
        warnings.append("fhir_allergies_not_collected")
    return warnings


def build_bundle(state: dict) -> dict:
    """Build a FHIR R4 Bundle from a completed intake state dict."""
    thread_id = state.get("thread_id") or _uid()
    patient_id = f"patient-{thread_id[:8]}"

    entries = [_entry(_patient(patient_id, state.get("identity") or {}))]

    if state.get("chief_complaint"):
        entries.append(_entry(_condition(patient_id, state)))

    for allergy in (state.get("allergies") or []):
        if allergy and allergy.strip():
            entries.append(_entry(_allergy(patient_id, allergy)))

    for med in (state.get("medications") or []):
        if med and (med.get("name") or "").strip():
            entries.append(_entry(_medication_statement(patient_id, med)))

    triage = state.get("triage") or {}
    if triage.get("risk_level"):
        entries.append(_entry(_triage_observation(patient_id, triage)))

    return {
        "resourceType": "Bundle",
        "id": _uid(),
        "type": "document",
        "timestamp": _now(),
        "entry": entries,
    }


# ---------------------------------------------------------------------------
# Post-build structural validation (R4 spec — lightweight, no external lib)
# ---------------------------------------------------------------------------

_REQUIRED_RESOURCE_TYPES = {"Patient", "Condition"}

_RESOURCE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Patient":             ["id", "resourceType", "name"],
    "Condition":           ["id", "resourceType", "subject", "clinicalStatus", "code"],
    "AllergyIntolerance":  ["id", "resourceType", "patient", "clinicalStatus", "code"],
    "MedicationStatement": ["id", "resourceType", "subject", "status", "medicationCodeableConcept"],
    "Observation":         ["id", "resourceType", "subject", "status", "code"],
}


def validate_fhir_bundle(bundle: dict) -> List[str]:
    """
    Structural validation of a built FHIR R4 Bundle.

    Checks:
    1. Bundle has resourceType=Bundle, type, timestamp, and a non-empty entry list.
    2. Every entry has fullUrl and resource.
    3. Required resource types (Patient, Condition) are present.
    4. Each resource contains its required R4 fields.

    Returns a list of error strings (empty = valid).
    Called in report_node after build_bundle() — failures are logged before
    the bundle reaches the EHR boundary so malformed bundles are never silently
    forwarded.
    """
    errors: List[str] = []

    if (bundle.get("resourceType") or "") != "Bundle":
        errors.append("bundle_missing_resourceType_Bundle")
    if not bundle.get("type"):
        errors.append("bundle_missing_type")
    if not bundle.get("timestamp"):
        errors.append("bundle_missing_timestamp")

    entries = bundle.get("entry") or []
    if not entries:
        errors.append("bundle_empty_entry_list")
        return errors  # nothing more to check

    seen_types: set[str] = set()
    for i, entry in enumerate(entries):
        if not entry.get("fullUrl"):
            errors.append(f"entry[{i}]_missing_fullUrl")
        resource = entry.get("resource") or {}
        rtype = resource.get("resourceType") or ""
        if not rtype:
            errors.append(f"entry[{i}]_missing_resourceType")
            continue
        seen_types.add(rtype)
        for field in _RESOURCE_REQUIRED_FIELDS.get(rtype, []):
            if field not in resource:
                errors.append(f"{rtype}_missing_{field}")

    for required in _REQUIRED_RESOURCE_TYPES:
        if required not in seen_types:
            errors.append(f"bundle_missing_required_resource_{required}")

    return errors
