"""
LLMProvider — provider-agnostic interface every backend must implement.

This module defines the contract that the rest of the application uses to
talk to a large language model.  All other modules (nodes, runner, intent,
safety) call into this interface, never into a specific provider's SDK.

Adding a new LLM (e.g. Anthropic, OpenAI):
  1. Create a new file under app/llm/ implementing LLMProvider.
  2. In your bootstrap (main.py or a settings hook) call set_provider(YourProvider()).
  3. Nothing else changes — run_json_step, the circuit breaker, intent
     classification, cost accounting, retries, repair loop, and structured
     logging all work against this interface.

Why this lives in its own module:
  - The interface is small and stable; provider implementations are large and
    SDK-specific.  Keeping them apart prevents accidental SDK leakage into
    business code.
  - Test doubles can implement LLMProvider directly (no SDK mocking needed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Hard cap on raw response chars
# ---------------------------------------------------------------------------
# Applied centrally after the provider returns, so a single runaway response
# can never fill the DB, message history, or context window.  Providers
# should NOT enforce their own cap — let the runner handle it uniformly.
MAX_RESPONSE_CHARS: int = 8_000


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResult:
    """
    Value type returned by every LLMProvider.generate_text call.

    cached_input_tokens is the count of input tokens that were billed at the
    cache-hit rate (e.g. Gemini implicit caching, Anthropic ephemeral cache).
    Providers that don't support caching report 0.  cost computation in
    runner.py uses input_tokens for full-price tokens and the cached count for
    discounted ones, so cost numbers stay accurate across backends.
    """
    ok: bool
    text: str
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


# ---------------------------------------------------------------------------
# Transient error classification
# ---------------------------------------------------------------------------

def is_transient_error(e: Exception) -> bool:
    """
    Decide whether a provider exception is worth retrying.

    Centralised so every provider gets the same retry policy:
      - Timeouts, rate limits, 5xx, quota errors → retry with backoff
      - Auth, permission, malformed-argument errors → never retry (permanent)
      - Unknown errors → conservative: don't retry, surface to caller
    """
    name = type(e).__name__.lower()
    msg = str(e).lower()

    if any(k in name for k in ["timeout", "deadline", "unavailable", "resourceexhausted"]):
        return True
    if any(k in msg for k in ["timeout", "timed out", "rate limit", "429",
                              "unavailable", "503", "temporarily", "quota exceeded"]):
        return True
    if any(k in msg for k in ["api key", "permission", "unauthorized",
                              "forbidden", "invalid argument", "not found",
                              "invalid api", "api_key"]):
        return False
    return False


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Abstract backend for any LLM (Gemini, Anthropic, OpenAI, local, mock).

    Responsibility split:
      generate_text           — synchronous structured/free-text completion
      generate_text_stream    — optional streaming variant (default impl falls
                                back to non-streaming and yields one chunk)
      validate                — startup health check; raise on auth/connectivity
      model_name              — identifier for logs and cost queries
      input_cost_per_million  — current input pricing (per 1M tokens)
      output_cost_per_million — current output pricing
      cached_input_cost_per_million — pricing for cache-hit input tokens

    Providers should be pure: no business logic, no DB writes, no logging
    other than provider-internal trace events.  The runner does all of that.
    """

    # ── Completion ────────────────────────────────────────────────────────

    @abstractmethod
    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_mime_type: Optional[str] = "application/json",
        cache_key: Optional[str] = None,
    ) -> LLMResult:
        """
        Synchronous completion.

        cache_key is an opaque hint for prompt-caching backends.  When set,
        callers expect the system prompt to be eligible for the backend's
        prefix-cache (Gemini explicit CachedContent, Anthropic ephemeral
        cache, OpenAI prompt caching, …).  Providers may ignore it; the
        runner passes it whenever a system prompt is reused across turns.
        """
        ...

    def generate_text_stream(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_mime_type: Optional[str] = "application/json",
        cache_key: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming completion.

        Default implementation runs the non-streaming path and yields the
        full response as a single chunk.  Override in providers that natively
        support SSE so callers get token-level latency.
        """
        result = self.generate_text(
            system=system, prompt=prompt, temperature=temperature,
            max_tokens=max_tokens, response_mime_type=response_mime_type,
            cache_key=cache_key,
        )
        if result.ok and result.text:
            yield result.text

    @abstractmethod
    def validate(self) -> None:
        """
        Startup probe.  Must raise on auth / configuration / connectivity
        failures so the application fails fast rather than discovering the
        problem on the first patient turn.
        """
        ...

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier (e.g. 'gemini-2.5-flash-lite')."""
        ...

    @property
    @abstractmethod
    def input_cost_per_million(self) -> float:
        """USD cost per 1M input tokens at full (non-cached) price."""
        ...

    @property
    @abstractmethod
    def output_cost_per_million(self) -> float:
        """USD cost per 1M output tokens."""
        ...

    @property
    def cached_input_cost_per_million(self) -> float:
        """
        USD cost per 1M cached-hit input tokens.  Default: 25% of full price,
        which matches typical provider discounts.  Override if your provider
        uses a different ratio (Anthropic 90% off, OpenAI 50% off, etc.).
        """
        return self.input_cost_per_million * 0.25
