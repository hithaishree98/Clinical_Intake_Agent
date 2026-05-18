"""
GeminiProvider — Google Gemini implementation of LLMProvider.

Isolated from the rest of the codebase so the SDK import is contained here.
If you swap to a different backend, this is the only file that needs to
import google-genai (or that the user needs to delete).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    _GENAI_AVAILABLE = False

from ..logging_utils import log_event
from ..settings import get_settings
from .base import LLMProvider, LLMResult

# Client-side TTL for the cache registry.  Gemini's server-side TTL is set to
# _CACHE_SERVER_TTL_SECONDS; we refresh our entry 5 minutes early so a call
# never races against expiry.
_CACHE_CLIENT_TTL = 3300   # 55 minutes
_CACHE_SERVER_TTL = 3600   # 60 minutes (Gemini minimum)


@dataclass
class _CacheEntry:
    name: str          # "cachedContents/<id>" returned by Gemini
    prompt_hash: str   # SHA-256[:16] of the system prompt — detects prompt changes
    expires_at: float  # time.monotonic() deadline for client-side invalidation


class GeminiProvider(LLMProvider):
    """
    Gemini Flash backend with per-schema-type system-prompt caching.
    Falls back to inline system_instruction if cache creation fails.
    """

    def __init__(self) -> None:
        if not _GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai is not installed. Install it or set a different "
                "provider via app.llm.set_provider()."
            )
        if not get_settings().gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        self._client = genai.Client(api_key=get_settings().gemini_api_key)
        # Per-provider cache registry.  Keyed by cache_key (schema name by
        # default).  Accessed from multiple request threads — guarded by lock.
        self._cache_registry: dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return get_settings().gemini_flash_model

    @property
    def input_cost_per_million(self) -> float:
        return get_settings().intake.gemini_input_cost_per_million

    @property
    def output_cost_per_million(self) -> float:
        return get_settings().intake.gemini_output_cost_per_million

    # ── Validation ────────────────────────────────────────────────────────

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
                f"Model: {self.model_name}\n"
                "Check your GEMINI_API_KEY and GEMINI_FLASH_MODEL in .env"
            )

    # ── Explicit system-prompt cache ───────────────────────────────────────

    def _get_cached_content_name(self, system: str, cache_key: str) -> Optional[str]:
        """
        Return a live Gemini CachedContent name for this system prompt,
        creating or refreshing the cache entry if needed.

        Returns None if caching is unavailable or creation fails — callers
        fall back to sending system_instruction inline.

        Called inside the ThreadPoolExecutor so it is covered by the same
        overall timeout as generate_content.
        """
        # Gemini requires ≥2048 tokens (~8 000 chars) to create a cache.
        # Short prompts (validate calls, stubs) must go inline.
        if len(system) < 8000:
            return None

        prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
        now = time.monotonic()

        with self._cache_lock:
            entry = self._cache_registry.get(cache_key)
            if entry and entry.prompt_hash == prompt_hash and entry.expires_at > now:
                return entry.name

        # Cache miss or stale — create a new CachedContent on Gemini.
        # Done outside the lock: this is a network call and we don't want to
        # block other threads reading the registry while it completes.
        try:
            cache = self._client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=system,
                    ttl=f"{_CACHE_SERVER_TTL}s",
                ),
            )
            with self._cache_lock:
                self._cache_registry[cache_key] = _CacheEntry(
                    name=cache.name,
                    prompt_hash=prompt_hash,
                    expires_at=now + _CACHE_CLIENT_TTL,
                )
            log_event("gemini_cache_created", cache_key=cache_key,
                      cache_name=cache.name)
            return cache.name
        except Exception as exc:
            log_event("gemini_cache_create_failed", level="warning",
                      cache_key=cache_key, error=str(exc)[:200])
            return None

    # ── Completion ────────────────────────────────────────────────────────

    def _call_generate(
        self,
        system: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response_mime_type: Optional[str],
        cache_key: str,
    ):
        """Inner call executed inside the ThreadPoolExecutor timeout boundary."""
        cache_name = self._get_cached_content_name(system, cache_key)

        if cache_name:
            # System prompt already cached on Gemini — do not send it again.
            config = types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type=response_mime_type,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type=response_mime_type,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        return self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

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
        # Validate calls (short system prompts) skip caching — they're one-shot
        # and below Gemini's minimum cacheable token count.
        resolved_key = cache_key or "default"
        timeout = get_settings().llm_timeout_seconds

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._call_generate,
                    system, prompt, temperature, max_tokens,
                    response_mime_type, resolved_key,
                )
                resp = future.result(timeout=timeout)
        except Exception as e:
            return LLMResult(
                ok=False, text="",
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )

        # Token accounting — usage_metadata may include cached_content_token_count
        # on cache hits.  Tolerate its absence.
        meta = getattr(resp, "usage_metadata", None)
        input_tokens  = int(getattr(meta, "prompt_token_count",          0) or 0) if meta else 0
        output_tokens = int(getattr(meta, "candidates_token_count",      0) or 0) if meta else 0
        cached_tokens = int(getattr(meta, "cached_content_token_count",  0) or 0) if meta else 0

        return LLMResult(
            ok=True,
            text=resp.text or "",
            error="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_tokens,
        )
