"""
Provider registry — singleton access for the configured LLM backend.

This is the single point of indirection between business code and a
concrete provider.  Callers do:

    from app.llm import get_provider
    result = get_provider().generate_text(...)

To swap backends in production, call set_provider(YourProvider()) from a
bootstrap hook before build_graph() runs.  In tests, monkeypatch this
module's _provider directly or call set_provider(MockProvider()).
"""
from __future__ import annotations

import threading
from typing import Optional

from ..settings import get_settings
from .base import LLMProvider
from .circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Provider singleton
# ---------------------------------------------------------------------------

_provider: Optional[LLMProvider] = None
_provider_lock = threading.Lock()


def get_provider() -> LLMProvider:
    """
    Return the configured provider, initialising the default (Gemini) on
    first call.  Thread-safe via double-checked locking.
    """
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                # Default to Gemini.  Imported lazily so swapping providers
                # before the first call doesn't pull in google-genai.
                from .gemini import GeminiProvider
                _provider = GeminiProvider()
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """
    Override the provider.  Use at app startup (before the graph is built)
    or in tests to inject a mock.  Calling this also resets the circuit
    breaker so prior failures from the old provider don't trip the new one.
    """
    global _provider
    with _provider_lock:
        _provider = provider
    _reset_breaker()


# ---------------------------------------------------------------------------
# Circuit breaker singleton
# ---------------------------------------------------------------------------

_breaker: Optional[CircuitBreaker] = None
_breaker_lock = threading.Lock()


def _get_breaker() -> CircuitBreaker:
    """
    Return the process-wide circuit breaker, lazily initialised from settings.
    """
    global _breaker
    if _breaker is None:
        with _breaker_lock:
            if _breaker is None:
                _breaker = CircuitBreaker(
                    failure_threshold=get_settings().circuit_breaker_failure_threshold,
                    recovery_timeout=get_settings().circuit_breaker_recovery_seconds,
                )
    return _breaker


def _reset_breaker() -> None:
    """Force a re-init of the breaker. Used by set_provider() and tests."""
    global _breaker
    with _breaker_lock:
        _breaker = None


def is_llm_available() -> bool:
    """
    Probe used by callers (e.g. /chat) to short-circuit when the backend is
    known-down.  Returns False only when the breaker is fully OPEN — HALF_OPEN
    still passes (the probe request flows through and either closes or
    re-opens the breaker).
    """
    return _get_breaker().allow_request()
