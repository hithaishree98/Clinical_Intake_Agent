from __future__ import annotations

import time
import random
import json
from dataclasses import dataclass
from typing import Callable, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

from .settings import settings
from .logging_utils import log_event


def is_transient_error(e: Exception) -> bool:
    # Retry only for issues like timeout/429/temp outage.
    name = type(e).__name__.lower()
    msg = str(e).lower()

    
    if any(k in name for k in ["timeout", "deadline", "unavailable", "resourceexhausted"]):
        return True
    if any(k in msg for k in ["timeout", "timed out", "rate limit", "429", "unavailable", "503", "temporarily"]):
        return True

    # auth/config issues
    if any(k in msg for k in ["api key", "permission", "unauthorized", "forbidden", "invalid argument", "not found"]):
        return False

    # don't retry unknown exceptions 
    return False


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    error: str = ""


class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def _retry(self, func: Callable[[], str], op: str, max_retries: Optional[int] = None) -> LLMResult:
        max_retries = max_retries or settings.max_retries
        delay = settings.base_retry_delay

        for attempt in range(max_retries):
            try:
                return LLMResult(True, func() or "")
            except Exception as e:
                # Fail fast on non-transient errors or last attempt
                if (not is_transient_error(e)) or (attempt == max_retries - 1):
                    log_event(
                        "llm_error",
                        level="error",
                        op=op,
                        model=settings.gemini_flash_model,
                        error_type=type(e).__name__,
                        error=str(e)[:400],
                    )
                    return LLMResult(False, "", f"{type(e).__name__}: {str(e)[:200]}")

                # Exponential backoff with jitter for transient failures
                time.sleep(min(settings.max_retry_delay, delay) * random.uniform(0.8, 1.2))
                delay *= 2

        return LLMResult(False, "", "unknown_retry_failure")

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_mime_type: str | None = "application/json",
    ) -> LLMResult:
        def _call() -> str:
            resp = self.client.models.generate_content(
                model=settings.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type=response_mime_type,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""

        return self._retry(_call, op="generate_text")


_gemini: Optional[GeminiClient] = None


def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


def make_repair_prompt(original_prompt: str, schema: Type[BaseModel], error: str, previous_output: str) -> str:
    # Strict repair prompt when JSON/schema validation fails.
    keys = list(schema.model_fields.keys())
    return (
        f"{original_prompt}\n\n"
        "Return ONLY valid JSON. No markdown. No extra keys.\n"
        f"Required keys: {keys}\n"
        f"Validation error: {error}\n"
        f"Previous output: {previous_output[:800]}"
    )


def extract_json(raw: str) -> str:
    """Extract the first valid JSON object/array from model text."""
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


def _clamp_fallback_strings(data: dict, max_len: int = 600) -> dict:

    #Keep fallback safe/compact (prevents overly long strings leaking into UI/state).

    out = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = v.strip()[:max_len]
        else:
            out[k] = v
    return out


def run_json_step(
    *,
    system: str,
    prompt: str,
    schema: Type[BaseModel],
    fallback: dict,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[BaseModel, dict]:
    """Try JSON once, one repair try, else fallback."""
    t0 = time.time()

    # 1) First attempt
    res = get_gemini().generate_text(
        system=system,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_mime_type="application/json",
    )

    cleaned = extract_json(res.text)
    parse_ok = False
    parse_error = ""
    obj: BaseModel

    if res.ok and cleaned:
        try:
            obj = schema.model_validate_json(cleaned)
            parse_ok = True
        except ValidationError as ve:
            parse_error = f"schema_validation_error: {ve.errors()[0].get('msg','')}"
        except Exception as e:
            parse_error = f"json_parse_error: {str(e)}"
    else:
        parse_error = res.error or "no_json_found"

    # 2) One content-repair retry (ONLY if model responded but content didn't validate)
    repair_used = False
    if (not parse_ok) and res.ok:
        repair_used = True
        res2 = get_gemini().generate_text(
            system=system,
            prompt=make_repair_prompt(prompt, schema, parse_error, res.text),
            temperature=temperature,
            max_tokens=max_tokens,
            response_mime_type="application/json",
        )
        cleaned2 = extract_json(res2.text)

        if res2.ok and cleaned2:
            try:
                obj = schema.model_validate_json(cleaned2)
                parse_ok = True
                parse_error = ""
                cleaned = cleaned2
                res = res2
            except ValidationError as ve:
                parse_error = f"schema_validation_error: {ve.errors()[0].get('msg','')}"
            except Exception as e:
                parse_error = f"json_parse_error: {str(e)}"

    # 3) Deterministic fallback
    if not parse_ok:
        obj = schema.model_validate(_clamp_fallback_strings(fallback))

    meta = {
        "llm_ok": res.ok,
        "llm_error": res.error,
        "latency_ms": int((time.time() - t0) * 1000),
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "repair_used": repair_used,
        "fallback_used": (not parse_ok),
        "raw_preview": (res.text or "")[:160],
        "cleaned_preview": (cleaned or "")[:160],
    }
    return obj, meta