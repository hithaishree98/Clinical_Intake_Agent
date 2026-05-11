"""Tests for CircuitBreaker state transitions (app/llm/circuit_breaker.py)."""
import time
import pytest
from app.llm.circuit_breaker import CircuitBreaker


def _breaker(threshold=3, recovery=60.0):
    return CircuitBreaker(failure_threshold=threshold, recovery_timeout=recovery)


class TestInitialState:
    def test_starts_closed(self):
        b = _breaker()
        assert b.state == CircuitBreaker.CLOSED

    def test_allows_requests_when_closed(self):
        assert _breaker().allow_request() is True

    def test_zero_failures_initially(self):
        assert _breaker().failures == 0


class TestClosedToOpen:
    def test_failures_below_threshold_stay_closed(self):
        b = _breaker(threshold=3)
        b.record_failure()
        b.record_failure()
        assert b.state == CircuitBreaker.CLOSED
        assert b.allow_request() is True

    def test_reaching_threshold_opens_breaker(self):
        b = _breaker(threshold=3)
        for _ in range(3):
            b.record_failure()
        assert b.state == CircuitBreaker.OPEN

    def test_open_breaker_blocks_requests(self):
        b = _breaker(threshold=1)
        b.record_failure()
        assert b.allow_request() is False

    def test_success_before_threshold_resets_failures(self):
        b = _breaker(threshold=5)
        b.record_failure()
        b.record_failure()
        b.record_success()
        assert b.failures == 0
        assert b.state == CircuitBreaker.CLOSED


class TestOpenToHalfOpen:
    def test_transitions_to_half_open_after_recovery_timeout(self):
        b = _breaker(threshold=1, recovery=0.01)
        b.record_failure()
        assert b.state == CircuitBreaker.OPEN
        time.sleep(0.05)
        assert b.state == CircuitBreaker.HALF_OPEN

    def test_half_open_allows_one_request(self):
        b = _breaker(threshold=1, recovery=0.01)
        b.record_failure()
        time.sleep(0.05)
        assert b.allow_request() is True

    def test_open_before_timeout_stays_open(self):
        b = _breaker(threshold=1, recovery=3600.0)
        b.record_failure()
        assert b.state == CircuitBreaker.OPEN


class TestHalfOpenResolution:
    def test_success_in_half_open_closes_breaker(self):
        b = _breaker(threshold=1, recovery=0.01)
        b.record_failure()
        time.sleep(0.05)
        assert b.state == CircuitBreaker.HALF_OPEN
        b.record_success()
        assert b.state == CircuitBreaker.CLOSED
        assert b.failures == 0

    def test_failure_in_half_open_reopens_breaker(self):
        b = _breaker(threshold=1, recovery=0.01)
        b.record_failure()
        time.sleep(0.05)
        assert b.state == CircuitBreaker.HALF_OPEN
        b.record_failure()
        assert b.state == CircuitBreaker.OPEN
