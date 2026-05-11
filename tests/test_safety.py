"""Tests for SafetyChecker.compute() and build_reason_trail() (app/safety.py)."""
import pytest
from app.safety import SafetyChecker, build_reason_trail, REVIEW_THRESHOLD


def _state(**kwargs):
    base = {
        "chief_complaint": "chest pain",
        "identity": {"name": "Jane Doe", "dob": "1985-03-15", "phone": "4125551234", "address": ""},
        "clinical_complete": True,
        "identity_status": "verified",
        "mode": "clinic",
        "triage": {},
        "current_phase": "report",
    }
    base.update(kwargs)
    return base


class TestSafetyCheckerHardBlocks:
    def test_missing_chief_complaint_blocks(self):
        r = SafetyChecker.compute(_state(chief_complaint=""))
        assert r.ok is False
        assert any("chief_complaint_missing" in b for b in r.blocking_reasons)

    def test_missing_name_blocks(self):
        r = SafetyChecker.compute(_state(identity={"name": "", "dob": "", "phone": "", "address": ""}))
        assert r.ok is False
        assert any("patient_name_missing" in b for b in r.blocking_reasons)

    def test_incomplete_clinical_history_blocks(self):
        r = SafetyChecker.compute(_state(clinical_complete=False))
        assert r.ok is False
        assert any("clinical_history_incomplete" in b for b in r.blocking_reasons)

    def test_all_three_hard_blocks_fire_together(self):
        r = SafetyChecker.compute(_state(
            chief_complaint="",
            identity={"name": "", "dob": "", "phone": "", "address": ""},
            clinical_complete=False,
        ))
        assert r.ok is False
        assert len(r.blocking_reasons) == 3

    def test_complete_session_passes(self):
        r = SafetyChecker.compute(_state())
        assert r.ok is True
        assert r.blocking_reasons == []


class TestSafetyCheckerScoring:
    def test_emergency_flag_crosses_threshold(self):
        r = SafetyChecker.compute(_state(triage={"emergency_flag": True}))
        assert r.safety_score >= REVIEW_THRESHOLD
        assert r.review_required is True

    def test_crisis_detected_sets_review_required(self):
        # Crisis weight (40) is below the numeric threshold (50), but the
        # crisis override forces review_required=True regardless.
        r = SafetyChecker.compute(_state(crisis_detected=True))
        assert r.safety_score == pytest.approx(SafetyChecker.WEIGHTS["crisis_detected_in_session"])
        assert r.review_required is True

    def test_crisis_forces_review_regardless_of_numeric_score(self):
        # crisis_override bypasses the numeric threshold check
        r = SafetyChecker.compute(_state(crisis_detected=True, identity_status="verified"))
        assert r.review_required is True

    def test_ed_mode_adds_baseline_score(self):
        r_clinic = SafetyChecker.compute(_state(mode="clinic"))
        r_ed     = SafetyChecker.compute(_state(mode="ed"))
        assert r_ed.safety_score > r_clinic.safety_score
        assert any("ed_mode_baseline" in rv for rv in r_ed.review_reasons)

    def test_low_extraction_quality_adds_review_reason(self):
        r = SafetyChecker.compute(_state(extraction_quality_score=0.40))
        assert any("extraction_quality_low" in rv for rv in r.review_reasons)

    def test_extraction_quality_above_threshold_no_flag(self):
        r = SafetyChecker.compute(_state(extraction_quality_score=0.80))
        assert not any("extraction_quality_low" in rv for rv in r.review_reasons)

    def test_identity_unverified_adds_score(self):
        r = SafetyChecker.compute(_state(identity_status="unverified"))
        assert r.safety_score >= SafetyChecker.WEIGHTS["identity_unverified"]
        assert any("identity_unverified" in rv for rv in r.review_reasons)

    def test_extraction_retried_adds_review_reason(self):
        r = SafetyChecker.compute(_state(extraction_retry_count=1))
        assert any("extraction_retried" in rv for rv in r.review_reasons)

    def test_no_retries_no_flag(self):
        r = SafetyChecker.compute(_state(extraction_retry_count=0))
        assert not any("extraction_retried" in rv for rv in r.review_reasons)

    def test_single_hard_block_score_arithmetic(self):
        r = SafetyChecker.compute(_state(chief_complaint=""))
        assert r.safety_score == pytest.approx(SafetyChecker.WEIGHTS["chief_complaint_missing"])


class TestBuildReasonTrail:
    def test_emergency_kind_includes_red_flags(self):
        state = _state(triage={"emergency_flag": True, "red_flags": ["chest pain"]})
        trail = build_reason_trail("emergency", state, extra_data={"red_flags": ["chest pain"]})
        assert trail["kind"] == "emergency"
        assert trail["severity"] == "critical"
        assert any("chest pain" in r for r in trail["reasons"])

    def test_crisis_kind_includes_matched_phrases(self):
        state = _state(crisis_detected=True)
        trail = build_reason_trail("crisis", state, extra_data={"matched_phrases": ["kill myself"]})
        assert trail["kind"] == "crisis"
        assert trail["severity"] == "critical"
        assert any("kill myself" in r for r in trail["reasons"])

    def test_identity_review_computes_field_diffs(self):
        state = _state(
            stored_identity={"name": "Jane Doe",   "dob": "1985-03-15", "phone": "4125551234", "address": ""},
            identity=       {"name": "Jane Smith", "dob": "1985-03-15", "phone": "4125551234", "address": ""},
        )
        trail = build_reason_trail("identity_review", state)
        assert any("name" in r for r in trail["reasons"])

    def test_trail_contains_context_snapshot(self):
        trail = build_reason_trail("review_required", _state())
        ctx = trail["context"]
        assert "patient_name" in ctx
        assert "chief_complaint" in ctx
        assert "triage_risk" in ctx
        assert "safety_score" in trail

    def test_extra_data_merged_into_trail(self):
        trail = build_reason_trail("emergency", _state(), extra_data={"custom_field": "value"})
        assert trail["custom_field"] == "value"

    def test_reserved_keys_not_overwritten_by_extra_data(self):
        trail = build_reason_trail("review_required", _state(), extra_data={"kind": "HACKED"})
        assert trail["kind"] == "review_required"

    def test_override_reasons_used_verbatim(self):
        trail = build_reason_trail("review_required", _state(), override_reasons=["custom: reason"])
        assert trail["reasons"] == ["custom: reason"]
