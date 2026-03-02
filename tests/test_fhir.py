"""
Tests for FHIR R4 Bundle generation.

Covers the fhir_builder module in isolation — no LLM or database needed.
"""

import json
import pytest
from app.fhir_builder import build_bundle, _normalize_dob


# ---------------------------------------------------------------------------
# DOB normalization
# ---------------------------------------------------------------------------

class TestNormalizeDob:
    def test_mm_dd_yyyy_slash(self):
        assert _normalize_dob("03/15/1985") == "1985-03-15"

    def test_mm_dd_yyyy_dash(self):
        assert _normalize_dob("03-15-1985") == "1985-03-15"

    def test_single_digit_month_and_day(self):
        assert _normalize_dob("3/5/1990") == "1990-03-05"

    def test_already_iso_format_passthrough(self):
        # If it's already YYYY-MM-DD or unrecognised, return as-is.
        assert _normalize_dob("1985-03-15") == "1985-03-15"

    def test_empty_string_passthrough(self):
        assert _normalize_dob("") == ""


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------

FULL_STATE = {
    "thread_id": "test-thread-abc123",
    "identity": {
        "name":    "Jane Doe",
        "dob":     "04/12/1980",
        "phone":   "4125551234",
        "address": "123 Main St, Pittsburgh PA",
    },
    "chief_complaint": "Chest pain",
    "opqrst": {
        "onset":       "2 hours ago",
        "provocation": "exertion",
        "quality":     "pressure",
        "radiation":   "left arm",
        "severity":    "7/10",
        "timing":      "constant",
    },
    "allergies":   ["Penicillin", "Latex"],
    "medications": [
        {"name": "Metformin", "dose": "500mg", "freq": "twice daily", "last_taken": "this morning"},
        {"name": "Lisinopril", "dose": "10mg", "freq": "once daily", "last_taken": ""},
    ],
    "triage": {
        "risk_level":      "medium",
        "visit_type":      "urgent_care_today",
        "rationale":       "High severity chest pain.",
        "emergency_flag":  False,
        "red_flags":       [],
    },
}


class TestBundleStructure:
    def test_returns_fhir_r4_bundle(self):
        bundle = build_bundle(FULL_STATE)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "document"
        assert "timestamp" in bundle
        assert "entry" in bundle

    def test_bundle_is_json_serializable(self):
        bundle = build_bundle(FULL_STATE)
        serialized = json.dumps(bundle)
        parsed = json.loads(serialized)
        assert parsed["resourceType"] == "Bundle"

    def test_entry_count(self):
        # Patient + Condition + 2 AllergyIntolerances + 2 MedicationStatements + Observation = 7
        bundle = build_bundle(FULL_STATE)
        assert len(bundle["entry"]) == 7

    def test_every_entry_has_full_url_and_resource(self):
        bundle = build_bundle(FULL_STATE)
        for entry in bundle["entry"]:
            assert entry["fullUrl"].startswith("urn:uuid:")
            assert "resource" in entry
            assert "resourceType" in entry["resource"]

    def test_resource_types_present(self):
        bundle = build_bundle(FULL_STATE)
        types = {e["resource"]["resourceType"] for e in bundle["entry"]}
        assert types == {"Patient", "Condition", "AllergyIntolerance", "MedicationStatement", "Observation"}


class TestPatientResource:
    def _get(self, bundle):
        return next(
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Patient"
        )

    def test_patient_name(self):
        p = self._get(build_bundle(FULL_STATE))
        assert p["name"][0]["text"] == "Jane Doe"

    def test_patient_dob_normalized(self):
        p = self._get(build_bundle(FULL_STATE))
        assert p["birthDate"] == "1980-04-12"

    def test_patient_phone(self):
        p = self._get(build_bundle(FULL_STATE))
        assert p["telecom"][0]["value"] == "4125551234"

    def test_patient_address(self):
        p = self._get(build_bundle(FULL_STATE))
        assert "Pittsburgh" in p["address"][0]["text"]

    def test_patient_id_is_stable_across_calls(self):
        # Same thread_id should produce the same patient id.
        b1 = build_bundle(FULL_STATE)
        b2 = build_bundle(FULL_STATE)
        pid1 = next(e["resource"]["id"] for e in b1["entry"] if e["resource"]["resourceType"] == "Patient")
        pid2 = next(e["resource"]["id"] for e in b2["entry"] if e["resource"]["resourceType"] == "Patient")
        assert pid1 == pid2


class TestConditionResource:
    def _get(self, bundle):
        return next(
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Condition"
        )

    def test_chief_complaint_in_code_text(self):
        c = self._get(build_bundle(FULL_STATE))
        assert c["code"]["text"] == "Chest pain"

    def test_opqrst_in_note(self):
        c = self._get(build_bundle(FULL_STATE))
        note = c["note"][0]["text"]
        assert "Onset: 2 hours ago" in note
        assert "Severity: 7/10" in note
        assert "Radiation: left arm" in note

    def test_clinical_status_is_active(self):
        c = self._get(build_bundle(FULL_STATE))
        code = c["clinicalStatus"]["coding"][0]["code"]
        assert code == "active"


class TestAllergyResources:
    def _get_all(self, bundle):
        return [
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "AllergyIntolerance"
        ]

    def test_two_allergies_produced(self):
        allergies = self._get_all(build_bundle(FULL_STATE))
        assert len(allergies) == 2

    def test_allergy_substance_names(self):
        substances = {a["code"]["text"] for a in self._get_all(build_bundle(FULL_STATE))}
        assert substances == {"Penicillin", "Latex"}

    def test_no_allergies_produces_no_resources(self):
        state = {**FULL_STATE, "allergies": []}
        bundle = build_bundle(state)
        allergies = self._get_all(bundle)
        assert len(allergies) == 0

    def test_empty_allergy_strings_skipped(self):
        state = {**FULL_STATE, "allergies": ["", "  ", "Penicillin"]}
        bundle = build_bundle(state)
        allergies = self._get_all(bundle)
        assert len(allergies) == 1


class TestMedicationStatements:
    def _get_all(self, bundle):
        return [
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "MedicationStatement"
        ]

    def test_two_medications_produced(self):
        meds = self._get_all(build_bundle(FULL_STATE))
        assert len(meds) == 2

    def test_med_name_in_codeable_concept(self):
        meds = self._get_all(build_bundle(FULL_STATE))
        names = {m["medicationCodeableConcept"]["text"] for m in meds}
        assert "Metformin" in names

    def test_dose_and_freq_in_note(self):
        meds = self._get_all(build_bundle(FULL_STATE))
        metformin = next(m for m in meds if m["medicationCodeableConcept"]["text"] == "Metformin")
        note = metformin["note"][0]["text"]
        assert "500mg" in note
        assert "twice daily" in note
        assert "this morning" in note

    def test_med_without_name_skipped(self):
        state = {**FULL_STATE, "medications": [{"name": "", "dose": "10mg"}]}
        bundle = build_bundle(state)
        meds = self._get_all(bundle)
        assert len(meds) == 0


class TestTriageObservation:
    def _get(self, bundle):
        return next(
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Observation"
        )

    def test_risk_level_in_value_string(self):
        obs = self._get(build_bundle(FULL_STATE))
        assert obs["valueString"] == "medium"

    def test_status_is_preliminary(self):
        obs = self._get(build_bundle(FULL_STATE))
        assert obs["status"] == "preliminary"

    def test_note_contains_visit_type_and_rationale(self):
        obs = self._get(build_bundle(FULL_STATE))
        note = obs["note"][0]["text"]
        assert "urgent_care_today" in note
        assert "High severity" in note

    def test_no_triage_produces_no_observation(self):
        state = {**FULL_STATE, "triage": {}}
        bundle = build_bundle(state)
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Observation" not in types


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_minimal_state_produces_patient_only(self):
        bundle = build_bundle({"thread_id": "x", "identity": {"name": "John"}})
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert types == ["Patient"]

    def test_empty_state_does_not_crash(self):
        bundle = build_bundle({})
        assert bundle["resourceType"] == "Bundle"
        assert len(bundle["entry"]) >= 1  # at least Patient

    def test_bundle_id_is_unique_per_call(self):
        b1 = build_bundle(FULL_STATE)
        b2 = build_bundle(FULL_STATE)
        assert b1["id"] != b2["id"]
