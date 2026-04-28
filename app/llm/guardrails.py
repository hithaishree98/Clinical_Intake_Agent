"""
guardrails.py — Output filters that scan LLM responses for unsafe content.

These run AFTER a provider call returns and BEFORE the response reaches the
patient.  They are intentionally provider-agnostic: any backend's output
must pass the same filter.

Currently implements:
  - validate_llm_response — blocks diagnosis language ("you have appendicitis",
    "consistent with X").  A clinical intake bot must never diagnose.

Adding more filters here keeps them centralised and easy to test in isolation.
"""
from __future__ import annotations

import re

from ..logging_utils import log_event


# ---------------------------------------------------------------------------
# Diagnosis-language detection
# ---------------------------------------------------------------------------

_DIAGNOSIS_PATTERNS = [
    r"\byou\s+(have|likely\s+have|probably\s+have|may\s+have|might\s+have)\b",
    r"\bdiagnos(is|ed|ing|e)\b",
    r"\bI\s+think\s+you\b",
    r"\bconsistent\s+with\b",
    r"\bsounds?\s+like\s+(you\s+have|a\s+case\s+of)\b",
    r"\bthis\s+is\s+(likely|probably|possibly)\s+(a|an)\s+\w+\s+(condition|disease|disorder|infection)\b",
]
_DIAGNOSIS_RE = re.compile("|".join(_DIAGNOSIS_PATTERNS), re.IGNORECASE)

_SAFE_REPLACEMENT = (
    "I've noted your symptoms. The clinician will review everything "
    "when they see you. Is there anything else you'd like to add?"
)


def validate_llm_response(text: str) -> tuple[str, bool]:
    """
    Check an LLM-generated reply for diagnosis language and replace it if found.

    Returns:
        (safe_text, was_modified)

    Modified replies are logged with a 200-char preview so prompt regressions
    that drift the LLM toward diagnosis can be detected post-deploy without
    forcing a code change.
    """
    if _DIAGNOSIS_RE.search(text or ""):
        log_event("guardrail_diagnosis_blocked", level="warning", preview=(text or "")[:200])
        return _SAFE_REPLACEMENT, True
    return text, False
