"""
runner.py — The provider-agnostic completion runner.

This is what business code calls when it wants a structured LLM response.
It does NOT depend on any specific provider; it routes through the registry.

Responsibilities (in call order):
  1. Circuit-breaker check          — short-circuit when backend is degraded
  2. Token-budget check             — log if the prompt is unusually large
  3. Retry with exponential full-jitter backoff
  4. Truncate oversize responses    — bounded by base.MAX_RESPONSE_CHARS
  5. JSON extract                   — strip markdown fences, find first JSON
  6. Pydantic schema validation
  7. One repair attempt on validation failure (cheap; usually succeeds)
  8. Hardcoded fallback             — guarantees the session continues
  9. Cost accounting                — full + cached token rates per provider
 10. Structured meta dict           — for logging and analytics

Why this design:
  - The runner is the only place that knows about retries, repair, fallback,
    and cost.  Provider implementations are pure SDK adapters.
  - Replacing the provider only requires implementing LLMProvider — the
    runner needs no changes.
  - A failed validation is more common (and cheaper to fix) than a network
    failure, so the repair loop is in the runner, not in the provider.
"""
from __future__ import annotations

import json
import random
import time
from typing import Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from ..logging_utils import log_event
from ..settings import get_settings
from .base import LLMProvider, LLMResult, MAX_RESPONSE_CHARS, is_transient_error
from .registry import get_provider, _get_breaker


# ---------------------------------------------------------------------------
# Token-budget guard (R5)
# ---------------------------------------------------------------------------
# Rough estimator — 1 token ≈ 4 chars for English-ish text.  Used only for
# logging warnings, not enforcement, since true token counts are
# tokenizer-specific.  Real enforcement happens via provider max_tokens
# and provider-side billing.

_TOKEN_BUDGET_WARN_INPUT = 8_000   # log a warning above this estimated input size

def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ---------------------------------------------------------------------------
# JSON extraction and repair
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> str:
    """
    Pull the first valid JSON object/array out of model output.

    Tolerates:
      - leading/trailing prose
      - ``` json fences
      - leading bullets, "Output:", etc.

    Returns "" if no valid JSON is found.
    """
    raw = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    if not raw:
        return ""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(raw[i:])
            return raw[i: i + end].strip()
        except Exception:
            continue
    return ""


def make_repair_prompt(original_prompt: str, schema: Type[BaseModel], error: str, previous_output: str) -> str:
    """
    Build a tightly prescriptive repair prompt naming the exact validation
    error, the required schema keys, and forbidding extra text.

    More aggressive than the original prompt because it has to overcome
    whatever drift caused the first response to fail.
    """
    keys = list(schema.model_fields.keys())
    return (
        f"{original_prompt}\n\n"
        "═══ REPAIR REQUIRED ═══\n"
        "Your previous response FAILED JSON validation. Produce a corrected version.\n"
        "RULES (non-negotiable):\n"
        "  • Return ONLY a JSON object. No markdown, no ```json, no preamble.\n"
        f"  • Required top-level keys (exactly these, no extras): {keys}\n"
        "  • All string values must be strings — never null.\n"
        f"Validation error: {error}\n"
        f"Bad output (first 800 chars): {previous_output[:800]}"
    )


def _clamp_fallback_strings(data: dict, max_len: int = 600) -> dict:
    return {k: (v.strip()[:max_len] if isinstance(v, str) else v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Retry wrapper around the provider call
# ---------------------------------------------------------------------------

def _retry_provider_call(
    provider: LLMProvider,
    *,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    response_mime_type: Optional[str],
    cache_key: Optional[str],
    op: str,
    max_retries: Optional[int] = None,
) -> LLMResult:
    """
    Wrap a single provider.generate_text call with:
      - circuit-breaker gate
      - exponential full-jitter retry on transient errors only
    """
    if not _get_breaker().allow_request():
        log_event("circuit_breaker_rejected", level="warning", op=op)
        return LLMResult(False, "", "circuit_breaker_open")

    settings = get_settings()
    max_retries = max_retries or settings.max_retries
    base = settings.base_retry_delay
    cap  = settings.max_retry_delay

    last_result: LLMResult = LLMResult(False, "", "no_attempt")

    for attempt in range(max_retries):
        try:
            result = provider.generate_text(
                system=system, prompt=prompt,
                temperature=temperature, max_tokens=max_tokens,
                response_mime_type=response_mime_type,
                cache_key=cache_key,
            )

            if result.ok:
                # Truncate oversize responses centrally — providers shouldn't
                # need to know the cap.
                if len(result.text) > MAX_RESPONSE_CHARS:
                    log_event("llm_response_truncated", level="warning",
                              op=op, original_chars=len(result.text))
                    result = LLMResult(
                        ok=result.ok,
                        text=result.text[:MAX_RESPONSE_CHARS],
                        error=result.error,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cached_input_tokens=result.cached_input_tokens,
                    )
                _get_breaker().record_success()
                return result

            # Provider returned ok=False.  Treat error message as an exception
            # for transient/permanent classification.
            err = Exception(result.error or "provider_returned_not_ok")
            permanent    = not is_transient_error(err)
            last_attempt = attempt == max_retries - 1
            last_result  = result

            if permanent or last_attempt:
                log_event("llm_error", level="error", op=op,
                          model=provider.model_name,
                          attempt=attempt + 1,
                          error=result.error,
                          permanent=permanent)
                _get_breaker().record_failure()
                return result

            ceiling = min(cap, base * (2 ** attempt))
            sleep_s = random.uniform(0, ceiling)
            log_event("llm_retry", op=op, attempt=attempt + 1,
                      sleep_ms=int(sleep_s * 1000), error=result.error[:100])
            time.sleep(sleep_s)

        except Exception as e:
            permanent    = not is_transient_error(e)
            last_attempt = attempt == max_retries - 1

            if permanent or last_attempt:
                log_event("llm_error", level="error", op=op,
                          model=provider.model_name,
                          attempt=attempt + 1,
                          error_type=type(e).__name__,
                          error=str(e)[:400],
                          permanent=permanent)
                _get_breaker().record_failure()
                return LLMResult(False, "", f"{type(e).__name__}: {str(e)[:200]}")

            ceiling = min(cap, base * (2 ** attempt))
            sleep_s = random.uniform(0, ceiling)
            log_event("llm_retry", op=op, attempt=attempt + 1,
                      sleep_ms=int(sleep_s * 1000), error=str(e)[:100])
            time.sleep(sleep_s)

    return last_result if last_result.error else LLMResult(False, "", "retry_exhausted")


# ---------------------------------------------------------------------------
# run_json_step — the canonical structured-LLM entry point
# ---------------------------------------------------------------------------

def run_json_step(
    *,
    system: str,
    prompt: str,
    schema: Type[BaseModel],
    fallback: dict,
    temperature: float = 0.2,
    max_tokens: int = 900,
    cache_key: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
) -> Tuple[BaseModel, dict]:
    """
    Three-level degradation:
      Level 1: primary call → JSON extract → schema validation
      Level 2: repair prompt (only when LLM responded but content was bad)
      Level 3: hardcoded fallback dict (session always continues)

    Returns (parsed_or_fallback_model, meta_dict).

    meta_dict keys (used for logging and analytics):
      llm_ok, llm_error, latency_ms, parse_ok, parse_error,
      repair_used, fallback_used, raw_preview, cleaned_preview,
      input_tokens, output_tokens, cached_input_tokens,
      cost_usd, model

    cache_key flows to the provider for any prefix-cache it may support.
    Defaults to the schema class name when not supplied — repeated calls
    against the same schema share cache entries naturally.
    """
    provider = provider or get_provider()
    cache_key = cache_key or schema.__name__

    # Token-budget guard (warn-only, no enforcement)
    est_input = _estimate_tokens(system) + _estimate_tokens(prompt)
    if est_input > _TOKEN_BUDGET_WARN_INPUT:
        log_event("llm_input_oversize", level="warning",
                  schema=schema.__name__,
                  est_input_tokens=est_input,
                  warn_threshold=_TOKEN_BUDGET_WARN_INPUT)

    t0 = time.time()

    res = _retry_provider_call(
        provider,
        system=system, prompt=prompt,
        temperature=temperature, max_tokens=max_tokens,
        response_mime_type="application/json",
        cache_key=cache_key,
        op=f"json_step:{schema.__name__}",
    )

    cleaned = extract_json(res.text)
    parse_ok, parse_error = False, ""
    obj: BaseModel

    if res.ok and cleaned:
        try:
            obj = schema.model_validate_json(cleaned)
            parse_ok = True
        except ValidationError as ve:
            parse_error = f"schema: {ve.errors()[0].get('msg', '')}"
        except Exception as e:
            parse_error = f"json: {str(e)}"
    else:
        parse_error = res.error or "empty_response"

    repair_used = False
    res2: Optional[LLMResult] = None
    if (not parse_ok) and res.ok:
        repair_used = True
        res2 = _retry_provider_call(
            provider,
            system=system,
            prompt=make_repair_prompt(prompt, schema, parse_error, res.text),
            temperature=temperature, max_tokens=max_tokens,
            response_mime_type="application/json",
            cache_key=cache_key,
            op=f"json_step_repair:{schema.__name__}",
        )
        cleaned2 = extract_json(res2.text)
        if res2.ok and cleaned2:
            try:
                obj = schema.model_validate_json(cleaned2)
                parse_ok, parse_error, cleaned = True, "", cleaned2
            except ValidationError as ve:
                parse_error = f"schema_after_repair: {ve.errors()[0].get('msg', '')}"
            except Exception as e:
                parse_error = f"json_after_repair: {str(e)}"

    if not parse_ok:
        log_event("llm_fallback_used", level="warning",
                  parse_error=parse_error,
                  repair_attempted=repair_used,
                  system_preview=system[:300],
                  prompt_preview=prompt[:200])
        obj = schema.model_validate(_clamp_fallback_strings(fallback))

    # Aggregate tokens across primary + optional repair call
    total_input  = res.input_tokens          + (res2.input_tokens          if res2 else 0)
    total_output = res.output_tokens         + (res2.output_tokens         if res2 else 0)
    total_cached = res.cached_input_tokens   + (res2.cached_input_tokens   if res2 else 0)

    # Cost: cached input tokens billed at the provider's discount rate
    full_input_tokens = max(0, total_input - total_cached)
    cost_usd = round(
        (full_input_tokens / 1_000_000) * provider.input_cost_per_million
        + (total_cached    / 1_000_000) * provider.cached_input_cost_per_million
        + (total_output    / 1_000_000) * provider.output_cost_per_million,
        8,
    )

    latency_ms = int((time.time() - t0) * 1000)
    meta = {
        "llm_ok":              res.ok,
        "llm_error":           res.error,
        "latency_ms":          latency_ms,
        "parse_ok":            parse_ok,
        "parse_error":         parse_error,
        "repair_used":         repair_used,
        "fallback_used":       not parse_ok,
        "raw_preview":         (res.text or "")[:160],
        "cleaned_preview":     (cleaned or "")[:160],
        "input_tokens":        total_input,
        "output_tokens":       total_output,
        "cached_input_tokens": total_cached,
        "cost_usd":            cost_usd,
        "model":               provider.model_name,
    }
    return obj, meta
