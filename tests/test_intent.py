"""Tests for intent classification helpers (app/intent.py)."""
import pytest
from unittest.mock import patch
from app.intent import (
    parse_quick_reply,
    is_bare_acknowledgment,
    detect_correction_section,
    QuickReply,
)


class TestParseQuickReply:
    def test_yes_tokens_return_yes(self):
        for tok in ("yes", "y", "yeah", "ok", "confirm", "i agree", "consent"):
            assert parse_quick_reply(tok) == QuickReply.YES, f"failed for {tok!r}"

    def test_no_tokens_return_no(self):
        for tok in ("no", "nope", "decline", "cancel", "stop"):
            assert parse_quick_reply(tok) == QuickReply.NO, f"failed for {tok!r}"

    def test_ack_only_tokens_return_ack(self):
        for tok in ("thanks", "got it", "sounds good"):
            assert parse_quick_reply(tok) == QuickReply.ACK, f"failed for {tok!r}"

    def test_correction_phrase_returns_correction(self):
        assert parse_quick_reply("go back") == QuickReply.CORRECTION
        assert parse_quick_reply("let me change that") == QuickReply.CORRECTION

    def test_yes_prefix_in_substantive_message_returns_none(self):
        # "yes I have chest pain" must NOT be classified as YES
        assert parse_quick_reply("yes I have chest pain") is None

    def test_no_prefix_in_substantive_message_returns_none(self):
        # "no I don't have allergies" must NOT be classified as NO
        assert parse_quick_reply("no I don't have any allergies") is None

    def test_empty_string_returns_none(self):
        assert parse_quick_reply("") is None

    def test_whitespace_returns_none(self):
        assert parse_quick_reply("   ") is None

    def test_unrecognized_text_returns_none(self):
        assert parse_quick_reply("I have a headache") is None
        assert parse_quick_reply("my knee hurts") is None


class TestIsBareAcknowledgment:
    def test_yes_tokens_are_bare_ack(self):
        assert is_bare_acknowledgment("yes") is True
        assert is_bare_acknowledgment("ok") is True
        assert is_bare_acknowledgment("sure") is True

    def test_ack_only_tokens_are_bare_ack(self):
        assert is_bare_acknowledgment("thanks") is True
        assert is_bare_acknowledgment("got it") is True

    def test_no_is_not_bare_ack(self):
        # "no" in clinical history = "no allergies/no meds" — must reach the extractor
        assert is_bare_acknowledgment("no") is False

    def test_substantive_with_ok_prefix_is_not_bare_ack(self):
        # The critical boundary: "ok I have penicillin allergy" carries real content
        assert is_bare_acknowledgment("ok I have penicillin allergy") is False

    def test_substantive_message_is_not_bare_ack(self):
        assert is_bare_acknowledgment("I have chest pain") is False
        assert is_bare_acknowledgment("penicillin and sulfa") is False


class TestDetectCorrectionSection:
    def test_name_detected_as_identity(self):
        assert detect_correction_section("fix my name") == "identity"

    def test_phone_detected_as_identity(self):
        assert detect_correction_section("my phone number is wrong") == "identity"

    def test_dob_detected_as_identity(self):
        assert detect_correction_section("I want to change my date of birth") == "identity"

    def test_symptom_field_detected(self):
        assert detect_correction_section("change my symptoms") == "symptoms"
        assert detect_correction_section("the severity I gave was wrong") == "symptoms"

    def test_allergy_field_detected_as_history(self):
        assert detect_correction_section("I forgot an allergy") == "history"

    def test_medication_field_detected_as_history(self):
        assert detect_correction_section("fix my meds") == "history"

    def test_pmh_detected_as_history(self):
        assert detect_correction_section("my medical history is incomplete") == "history"

    def test_generic_go_back_returns_none(self):
        assert detect_correction_section("go back please") == "none"
        assert detect_correction_section("I want to start over") == "none"


class TestClassifyIntentTier2:
    """LLM path: short ambiguous messages that don't match any token set."""

    def test_ambiguous_message_calls_llm(self):
        from app.intent import classify_intent
        from app.schemas import IntentOut

        mock_result = IntentOut(intent="confirm", correcting_section="none")
        mock_meta   = {"input_tokens": 5, "output_tokens": 3}

        with patch("app.llm.run_json_step", return_value=(mock_result, mock_meta)), \
             patch("app.sqlite_db.record_llm_usage"):
            result = classify_intent("I think so", thread_id="test-thread")

        assert result.intent == "confirm"

    def test_long_message_skips_llm(self):
        from app.intent import classify_intent

        long_msg = "my head hurts and has been hurting all morning since yesterday"
        with patch("app.llm.run_json_step") as mock_llm:
            result = classify_intent(long_msg)
            mock_llm.assert_not_called()

        assert result.intent == "provide_info"

    def test_exact_token_skips_llm(self):
        from app.intent import classify_intent

        with patch("app.llm.run_json_step") as mock_llm:
            result = classify_intent("yes")
            mock_llm.assert_not_called()

        assert result.intent == "confirm"

    def test_empty_message_skips_llm(self):
        from app.intent import classify_intent

        with patch("app.llm.run_json_step") as mock_llm:
            result = classify_intent("")
            mock_llm.assert_not_called()

        assert result.intent == "unclear"
