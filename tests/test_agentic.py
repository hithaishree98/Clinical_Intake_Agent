"""Tests for agentic helpers (app/agentic.py) — all pure functions, no LLM."""
import pytest
from app.agentic import (
    score_extraction_quality,
    build_gap_fill_question,
    adapt_clinical_question,
    build_validation_gap_message,
)


class TestScoreExtractionQuality:
    def test_empty_everything_is_zero(self):
        assert score_extraction_quality("", {}) == 0.0

    def test_chief_complaint_only(self):
        assert score_extraction_quality("headache", {}) == pytest.approx(0.25)

    def test_cc_plus_onset_plus_severity(self):
        opqrst = {"onset": "2 hours ago", "severity": "7/10"}
        assert score_extraction_quality("headache", opqrst) == pytest.approx(0.65)

    def test_all_fields_scores_one(self):
        opqrst = {
            "onset": "2h", "severity": "7/10", "quality": "throbbing",
            "timing": "constant", "provocation": "light", "radiation": "neck",
        }
        assert score_extraction_quality("headache", opqrst) == pytest.approx(1.0)

    def test_empty_field_values_do_not_score(self):
        opqrst = {"onset": "", "severity": "   "}
        assert score_extraction_quality("headache", opqrst) == pytest.approx(0.25)

    def test_whitespace_only_cc_scores_zero(self):
        assert score_extraction_quality("   ", {}) == 0.0

    def test_score_never_exceeds_one(self):
        opqrst = {
            "onset": "x", "severity": "x", "quality": "x",
            "timing": "x", "provocation": "x", "radiation": "x",
        }
        assert score_extraction_quality("cc", opqrst) <= 1.0


class TestBuildGapFillQuestion:
    def test_severity_asked_first_when_missing(self):
        q = build_gap_fill_question("headache", {}, "routine_checkup")
        assert "0 to 10" in q or "severe" in q.lower()

    def test_onset_asked_when_severity_present(self):
        q = build_gap_fill_question("headache", {"severity": "7/10"}, "routine_checkup")
        assert "start" in q.lower() or "when" in q.lower()

    def test_quality_asked_when_severity_and_onset_present(self):
        q = build_gap_fill_question("pain", {"severity": "7", "onset": "1h"}, "routine_checkup")
        assert "describe" in q.lower() or "sharp" in q.lower() or "dull" in q.lower()

    def test_all_fields_present_returns_fallback(self):
        opqrst = {
            "severity": "7", "onset": "2h", "quality": "sharp",
            "timing": "constant", "radiation": "none", "provocation": "none",
        }
        q = build_gap_fill_question("headache", opqrst, "routine_checkup")
        assert "headache" in q

    def test_empty_cc_uses_symptom_placeholder(self):
        q = build_gap_fill_question("", {}, "routine_checkup")
        assert "symptom" in q


class TestAdaptClinicalQuestion:
    def test_emergency_allergies_returns_adapted_question(self):
        q = adapt_clinical_question("allergies", "emergency")
        assert q != ""
        assert "quick" in q.lower()

    def test_emergency_meds_returns_adapted_question(self):
        q = adapt_clinical_question("meds", "emergency")
        assert q != ""
        assert "quick" in q.lower()

    def test_specialist_meds_mentions_other_providers(self):
        q = adapt_clinical_question("meds", "specialist")
        assert "other providers" in q.lower()

    def test_pediatric_allergies_addresses_parent(self):
        q = adapt_clinical_question("allergies", "pediatric")
        assert "patient" in q.lower()

    def test_mental_health_meds_mentions_psychiatric(self):
        q = adapt_clinical_question("meds", "mental_health")
        assert "psychiatric" in q.lower()

    def test_missing_combination_returns_empty_string(self):
        assert adapt_clinical_question("allergies", "routine_checkup") == ""
        assert adapt_clinical_question("unknown_step", "emergency") == ""
        assert adapt_clinical_question("meds", "unknown_class") == ""


class TestBuildValidationGapMessage:
    def test_missing_chief_complaint(self):
        msg = build_validation_gap_message(["chief_complaint"], "", "clinic")
        assert "brought you in" in msg.lower()

    def test_opqrst_incomplete_clinic_mode(self):
        msg = build_validation_gap_message(["opqrst_incomplete"], "headache", "clinic")
        assert "headache" in msg.lower()

    def test_opqrst_incomplete_ed_mode_uses_ed_language(self):
        msg = build_validation_gap_message(["opqrst_incomplete"], "chest pain", "ed")
        assert "emergency" in msg.lower()

    def test_severity_required_error(self):
        msg = build_validation_gap_message(["severity_required"], "back pain", "clinic")
        assert "back pain" in msg.lower()

    def test_allergies_not_collected(self):
        msg = build_validation_gap_message(["allergies_not_collected"], "headache", "clinic")
        assert "allerg" in msg.lower()

    def test_extraction_confidence_low(self):
        msg = build_validation_gap_message(["extraction_confidence_low"], "fever", "clinic")
        assert "fever" in msg.lower()

    def test_clinical_history_incomplete(self):
        msg = build_validation_gap_message(["clinical_history_incomplete"], "", "clinic")
        assert "clinical history" in msg.lower() or "haven" in msg.lower() or "finished" in msg.lower()

    def test_empty_errors_returns_default_message(self):
        assert build_validation_gap_message([], "headache", "clinic") == \
               "I need a bit more information before we continue."

    def test_multiple_errors_joined(self):
        msg = build_validation_gap_message(
            ["chief_complaint", "allergies_not_collected"], "", "clinic"
        )
        assert "brought you in" in msg.lower()
        assert "allerg" in msg.lower()
