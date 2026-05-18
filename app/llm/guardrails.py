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
    # Negative lookbehinds exclude "Do you have / Did you have" question forms
    r"(?<!do )(?<!Do )(?<!did )(?<!Did )(?<!does )(?<!Does )\byou\s+(have|likely\s+have|probably\s+have|may\s+have|might\s+have)\b",
    r"\bdiagnos(is|ed|ing|e)\b",
    r"\bI\s+think\s+you\b",
    r"\bconsistent\s+with\b",
    r"\bsounds?\s+like\s+(you\s+have|a\s+case\s+of)\b",
    r"\bthis\s+is\s+(likely|probably|possibly)\s+(a|an)\s+\w+\s+(condition|disease|disorder|infection)\b",
    # Treatment recommendations
    r"\byou\s+should\s+(take|start|begin|try)\b",
    r"\byou\s+(will|would)\s+(likely\s+|probably\s+)?(need|require)\b",
    r"\b(recommend|suggest|prescribe)\s+(starting|giving|a\s+course)\b",
    # Prognosis language
    r"\b(this|it)\s+(?:\w+\s+)?(will|may|could)\s+(?:\w+\s+)?(get\s+worse|progress|worsen)\b",
    r"\byou['']?ll\s+be\s+fine\b",
    # Minimising / alarming
    r"\bnothing\s+to\s+worry\b",
    r"\b(this|it)\s+(sounds?|seems?)\s+(very\s+)?serious\b",
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
