from __future__ import annotations
import time
import random
from dataclasses import dataclass
from typing import Callable, Optional, Type, Tuple
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

from .settings import settings
from .logging_utils import log_event

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
                if attempt == max_retries - 1:
                    log_event("llm_error", level="error", op=op, model=settings.gemini_flash_model,
                              error_type=type(e).__name__, error=str(e)[:400])
                    return LLMResult(False, "", f"{type(e).__name__}: {str(e)[:200]}")
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
        def _call():
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

def _safe_json_extract(raw: str) -> str:
    raw = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    if not raw:
        return ""
    o1, o2 = raw.find("{"), raw.rfind("}")
    a1, a2 = raw.find("["), raw.rfind("]")
    cands = []
    if o1 != -1 and o2 != -1 and o2 > o1:
        cands.append(raw[o1:o2 + 1])
    if a1 != -1 and a2 != -1 and a2 > a1:
        cands.append(raw[a1:a2 + 1])
    return max(cands, key=len).strip() if cands else ""

def run_json_step(
    *,
    system: str,
    prompt: str,
    schema: Type[BaseModel],
    fallback: dict,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[BaseModel, dict]:
    t0 = time.time()
    res = get_gemini().generate_text(
        system=system,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_mime_type="application/json",
    )

    cleaned = _safe_json_extract(res.text)
    parse_ok = False
    parse_error = ""

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

    if not parse_ok:
        obj = schema.model_validate(fallback)

    meta = {
        "llm_ok": res.ok,
        "llm_error": res.error,
        "latency_ms": int((time.time() - t0) * 1000),
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "fallback_used": (not parse_ok),
        "raw_preview": (res.text or "")[:160],
        "cleaned_preview": (cleaned or "")[:160],
    }
    return obj, meta
