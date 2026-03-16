
from __future__ import annotations

import time
import random
import json
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

from .settings import settings
from .logging_utils import log_event


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------

def is_transient_error(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg  = str(e).lower()

    if any(k in name for k in ["timeout", "deadline", "unavailable", "resourceexhausted"]):
        return True
    if any(k in msg for k in ["timeout", "timed out", "rate limit", "429",
                               "unavailable", "503", "temporarily", "quota exceeded"]):
        return True
    if any(k in msg for k in ["api key", "permission", "unauthorized",
                               "forbidden", "invalid argument", "not found",
                               "invalid api", "api_key"]):
        return False   # permanent — never retry
    return False       # unknown — conservative: don't retry


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Three-state breaker: CLOSED → OPEN → HALF_OPEN → CLOSED

    CLOSED   : normal — all requests pass through
    OPEN     : API is failing — all requests fail immediately with a cached error
    HALF_OPEN: recovery window — one probe request allowed; success closes it,
               failure re-opens it and resets the recovery timer
    """
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

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._opened_at >= self._recovery:
                    self._state = self.HALF_OPEN
                    log_event("circuit_breaker_half_open")
            return self._state

    def allow_request(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                log_event("circuit_breaker_closed", after_failures=self._failures)
            self._failures = 0
            self._state    = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._state != self.OPEN:
                self._state    = self.OPEN
                self._opened_at = time.time()
                log_event(
                    "circuit_breaker_opened",
                    level="warning",
                    consecutive_failures=self._failures,
                    recovery_in_sec=self._recovery,
                )


_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

MAX_RESPONSE_CHARS = 8_000   # hard cap — prevents runaway tokens filling DB


# ---------------------------------------------------------------------------
# LLMResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResult:
    ok:    bool
    text:  str
    error: str = ""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient:

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def validate(self) -> None:
        result = self.generate_text(
            system="You are a health check.",
            prompt="Reply with the single word: ok",
            temperature=0.0,
            max_tokens=5,
            response_mime_type="text/plain",
        )
        if not result.ok:
            raise RuntimeError(
                f"Gemini API validation failed at startup: {result.error}\n"
                f"Model: {settings.gemini_flash_model}\n"
                "Check your GEMINI_API_KEY and GEMINI_FLASH_MODEL in .env"
            )

    def _retry(self, func: Callable[[], str], op: str, max_retries: Optional[int] = None) -> LLMResult:
        """
        Exponential backoff with FULL JITTER.
        Sleep duration = uniform(0, min(cap, base * 2^attempt))
        Full jitter is superior to multiplicative jitter for thundering herd prevention.
        See: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
        """
        if not _breaker.allow_request():
            log_event("circuit_breaker_rejected", level="warning", op=op)
            return LLMResult(False, "", "circuit_breaker_open")

        max_retries = max_retries or settings.max_retries
        base  = settings.base_retry_delay
        cap   = settings.max_retry_delay

        for attempt in range(max_retries):
            try:
                text = func() or ""
                if len(text) > MAX_RESPONSE_CHARS:
                    log_event("llm_response_truncated", level="warning", op=op,
                              original_chars=len(text))
                    text = text[:MAX_RESPONSE_CHARS]
                _breaker.record_success()
                return LLMResult(True, text)

            except Exception as e:
                permanent    = not is_transient_error(e)
                last_attempt = attempt == max_retries - 1

                if permanent or last_attempt:
                    log_event("llm_error", level="error", op=op,
                              model=settings.gemini_flash_model,
                              attempt=attempt + 1,
                              error_type=type(e).__name__,
                              error=str(e)[:400],
                              permanent=permanent)
                    _breaker.record_failure()
                    return LLMResult(False, "", f"{type(e).__name__}: {str(e)[:200]}")

                ceiling = min(cap, base * (2 ** attempt))
                sleep_s = random.uniform(0, ceiling)   # full jitter
                log_event("llm_retry", op=op, attempt=attempt + 1,
                          sleep_ms=int(sleep_s * 1000), error=str(e)[:100])
                time.sleep(sleep_s)

        return LLMResult(False, "", "retry_exhausted")

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_mime_type: str | None = "application/json",
    ) -> LLMResult:
        timeout = settings.llm_timeout_seconds

        def _call() -> str:
            # google-genai SDK ignores HTTP timeouts, so we enforce our own.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self.client.models.generate_content,
                    model=settings.gemini_flash_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type=response_mime_type,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                resp = future.result(timeout=timeout)
            return resp.text or ""

        return self._retry(_call, op="generate_text")


_gemini: Optional[GeminiClient] = None

def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


# ---------------------------------------------------------------------------
# JSON extraction and repair
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> str:
    """Extract the first valid JSON object/array from model output."""
    raw = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    if not raw:
        return ""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(raw[i:])
            return raw[i : i + end].strip()
        except Exception:
            continue
    return ""


def make_repair_prompt(original_prompt: str, schema: Type[BaseModel], error: str, previous_output: str) -> str:
    """
    Targeted repair prompt — more prescriptive than the original.
    Names the exact error, lists required keys, forbids extra text.
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
# run_json_step — three-level degradation
# ---------------------------------------------------------------------------

def run_json_step(
    *,
    system: str,
    prompt: str,
    schema: Type[BaseModel],
    fallback: dict,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[BaseModel, dict]:
    """
    Level 1: primary call → JSON extract → schema validation
    Level 2: repair prompt (only when LLM responded but content was bad)
    Level 3: hardcoded fallback dict (session always continues)
    """
    t0 = time.time()

    res     = get_gemini().generate_text(system=system, prompt=prompt,
                                         temperature=temperature, max_tokens=max_tokens,
                                         response_mime_type="application/json")
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
    if (not parse_ok) and res.ok:
        repair_used = True
        res2     = get_gemini().generate_text(system=system,
                                               prompt=make_repair_prompt(prompt, schema, parse_error, res.text),
                                               temperature=temperature, max_tokens=max_tokens,
                                               response_mime_type="application/json")
        cleaned2 = extract_json(res2.text)
        if res2.ok and cleaned2:
            try:
                obj = schema.model_validate_json(cleaned2)
                parse_ok, parse_error, cleaned, res = True, "", cleaned2, res2
            except ValidationError as ve:
                parse_error = f"schema_after_repair: {ve.errors()[0].get('msg', '')}"
            except Exception as e:
                parse_error = f"json_after_repair: {str(e)}"

    if not parse_ok:
        log_event("llm_fallback_used", level="warning",
                  parse_error=parse_error, repair_attempted=repair_used)
        obj = schema.model_validate(_clamp_fallback_strings(fallback))

    meta = {
        "llm_ok": res.ok, "llm_error": res.error,
        "latency_ms": int((time.time() - t0) * 1000),
        "parse_ok": parse_ok, "parse_error": parse_error,
        "repair_used": repair_used, "fallback_used": not parse_ok,
        "raw_preview": (res.text or "")[:160],
        "cleaned_preview": (cleaned or "")[:160],
    }
    return obj, meta


# ---------------------------------------------------------------------------
# Diagnosis language filter — catches LLM output that drifts into diagnosing
# ---------------------------------------------------------------------------

import re as _re

_DIAGNOSIS_PATTERNS = [
    r"\byou\s+(have|likely\s+have|probably\s+have|may\s+have|might\s+have)\b",
    r"\bdiagnos(is|ed|ing|e)\b",
    r"\bI\s+think\s+you\b",
    r"\bconsistent\s+with\b",
    r"\bsounds?\s+like\s+(you\s+have|a\s+case\s+of)\b",
    r"\bthis\s+is\s+(likely|probably|possibly)\s+(a|an)\s+\w+\s+(condition|disease|disorder|infection)\b",
]
_DIAGNOSIS_RE = _re.compile("|".join(_DIAGNOSIS_PATTERNS), _re.IGNORECASE)

_SAFE_REPLACEMENT = (
    "I've noted your symptoms. The clinician will review everything "
    "when they see you. Is there anything else you'd like to add?"
)


def validate_llm_response(text: str) -> tuple[str, bool]:
    """Returns (safe_text, was_modified). Replaces diagnosis language with a safe fallback."""
    if _DIAGNOSIS_RE.search(text or ""):
        log_event("guardrail_diagnosis_blocked", level="warning", preview=(text or "")[:200])
        return _SAFE_REPLACEMENT, True
    return text, False