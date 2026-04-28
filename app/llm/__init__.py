"""
app.llm — provider-agnostic LLM access layer.

Public API (use these for new code):
  • LLMProvider            — ABC every backend implements (base.py)
  • LLMResult              — value type for completions (base.py)
  • run_json_step          — primary structured-completion entry point (runner.py)
  • validate_llm_response  — diagnosis-language guardrail (guardrails.py)
  • is_llm_available       — circuit-breaker probe (registry.py)
  • get_provider           — current provider singleton (registry.py)
  • set_provider           — override the provider, e.g. for tests (registry.py)
  • CircuitBreaker         — exposed for direct introspection (circuit_breaker.py)

Backward-compatible re-exports for existing call sites:
  • get_gemini             — alias for get_provider() (was the old factory)
  • _breaker, _get_breaker — used by tests and /health
  • extract_json, make_repair_prompt, MAX_RESPONSE_CHARS — internals that
    older code imported directly

Swapping the LLM backend:
  1. Subclass LLMProvider (see app/llm/base.py).
  2. At app startup, before build_graph runs:
        from app.llm import set_provider
        set_provider(MyProvider())
  3. Nothing else changes.  Every call site goes through this module.
"""
from __future__ import annotations

# ── Public, provider-agnostic API ─────────────────────────────────────────
from .base import (
    LLMProvider,
    LLMResult,
    MAX_RESPONSE_CHARS,
    is_transient_error,
)
from .circuit_breaker import CircuitBreaker
from .registry import (
    get_provider,
    set_provider,
    is_llm_available,
    _get_breaker,
)
from .runner import (
    run_json_step,
    extract_json,
    make_repair_prompt,
)
from .guardrails import validate_llm_response


# ── Backward-compatible aliases ───────────────────────────────────────────
# `get_gemini()` was the old factory.  It now returns whatever provider is
# configured (Gemini by default).  Kept so existing code and tests that
# patch `app.llm.get_gemini` continue to work.
def get_gemini() -> LLMProvider:
    return get_provider()


# `_breaker` was the module-level singleton.  Some call sites still read it
# directly (e.g. /health, conftest.py).  Module-level __getattr__ delegates
# to the registry so the read returns the live instance, not a None snapshot.
def __getattr__(name: str):
    if name == "_breaker":
        return _get_breaker()
    raise AttributeError(f"module 'app.llm' has no attribute {name!r}")


__all__ = [
    # Public API
    "LLMProvider",
    "LLMResult",
    "MAX_RESPONSE_CHARS",
    "is_transient_error",
    "CircuitBreaker",
    "get_provider",
    "set_provider",
    "is_llm_available",
    "run_json_step",
    "extract_json",
    "make_repair_prompt",
    "validate_llm_response",
    # Back-compat
    "get_gemini",
    "_get_breaker",
]
