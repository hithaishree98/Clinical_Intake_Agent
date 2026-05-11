"""Tests for webhook HMAC helpers (app/webhook.py) — pure functions, no HTTP."""
import hashlib
import hmac
import pytest
from app.webhook import verify_webhook_signature, _compute_signature


_SECRET  = "test-webhook-secret"
_PAYLOAD = b'{"event": "intake_complete", "thread_id": "abc123"}'


def _valid_sig(payload=_PAYLOAD, secret=_SECRET):
    return _compute_signature(secret, payload)


class TestVerifyWebhookSignature:
    def test_correct_signature_returns_true(self):
        assert verify_webhook_signature(_PAYLOAD, _valid_sig(), _SECRET) is True

    def test_wrong_secret_returns_false(self):
        sig = _compute_signature("wrong-secret", _PAYLOAD)
        assert verify_webhook_signature(_PAYLOAD, sig, _SECRET) is False

    def test_tampered_payload_returns_false(self):
        assert verify_webhook_signature(_PAYLOAD + b" extra", _valid_sig(), _SECRET) is False

    def test_empty_signature_returns_false(self):
        assert verify_webhook_signature(_PAYLOAD, "", _SECRET) is False

    def test_missing_sha256_prefix_returns_false(self):
        raw_hex = hmac.new(_SECRET.encode(), _PAYLOAD, hashlib.sha256).hexdigest()
        assert verify_webhook_signature(_PAYLOAD, raw_hex, _SECRET) is False

    def test_empty_secret_returns_false(self):
        assert verify_webhook_signature(_PAYLOAD, _valid_sig(), "") is False

    def test_truncated_signature_returns_false(self):
        truncated = _valid_sig()[len("sha256="):]
        assert verify_webhook_signature(_PAYLOAD, truncated, _SECRET) is False

    def test_different_payloads_produce_different_sigs(self):
        sig1 = _compute_signature(_SECRET, b"payload-one")
        sig2 = _compute_signature(_SECRET, b"payload-two")
        assert sig1 != sig2

    def test_signature_format_matches_github_convention(self):
        sig = _compute_signature(_SECRET, _PAYLOAD)
        assert sig.startswith("sha256=")
        assert len(sig) == len("sha256=") + 64  # 32-byte hash = 64 hex chars
