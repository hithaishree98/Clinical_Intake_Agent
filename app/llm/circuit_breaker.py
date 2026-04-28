"""
CircuitBreaker — three-state, provider-agnostic.

Lives in its own module so any provider implementation can use it (or not).
The runner wraps every provider call with the breaker, so individual
providers don't need to know it exists.

States:
  CLOSED    — normal; requests pass through
  OPEN      — backend is degraded; calls reject immediately with a cached error
  HALF_OPEN — recovery probe window; one request allowed, success closes the
              breaker, failure re-opens it and resets the recovery timer

"""
from __future__ import annotations

import threading
import time

from ..logging_utils import log_event


class CircuitBreaker:
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self._state     = self.CLOSED
        self._failures  = 0
        self._threshold = failure_threshold
        self._recovery  = recovery_timeout
        self._opened_at = 0.0
        self._lock      = threading.Lock()

    # ── State transitions ────────────────────────────────────────────────
    # Each method computes the new state under the lock, captures any log
    # intent into a local variable, then emits the structured log event
    # AFTER releasing the lock.  log_event ultimately writes through the
    # logging stack (json.dumps + handler I/O); holding the breaker lock
    # while doing that would briefly serialize all state checks under
    # contention.  Capturing-and-emitting decouples the hot path from log
    # I/O latency.

    @property
    def failures(self) -> int:
        """Consecutive failure count.  Read-only public surface for /health."""
        with self._lock:
            return self._failures

    @property
    def state(self) -> str:
        transitioned_to_half_open = False
        with self._lock:
            if self._state == self.OPEN and time.monotonic() - self._opened_at >= self._recovery:
                self._state = self.HALF_OPEN
                transitioned_to_half_open = True
            current = self._state
        if transitioned_to_half_open:
            log_event("circuit_breaker_half_open")
        return current

    def allow_request(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)

    def record_success(self) -> None:
        closed_from_half_open = False
        prior_failures = 0
        with self._lock:
            if self._state == self.HALF_OPEN:
                closed_from_half_open = True
                prior_failures = self._failures
            self._failures = 0
            self._state    = self.CLOSED
        if closed_from_half_open:
            log_event("circuit_breaker_closed", after_failures=prior_failures)

    def record_failure(self) -> None:
        opened_now = False
        consecutive = 0
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._state != self.OPEN:
                self._state    = self.OPEN
                self._opened_at = time.monotonic()
                opened_now = True
                consecutive = self._failures
        if opened_now:
            log_event(
                "circuit_breaker_opened",
                level="warning",
                consecutive_failures=consecutive,
                recovery_in_sec=self._recovery,
            )
