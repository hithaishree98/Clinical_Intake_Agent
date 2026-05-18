"""intent.py — Single source of truth for patient-message intent classification."""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from .extract import _norm
from .schemas import IntentOut


# ---------------------------------------------------------------------------
# Quick-reply enum — the canonical intent classification
# ---------------------------------------------------------------------------

class QuickReply(str, Enum):
    YES          = "yes"
    NO           = "no"
    ACK          = "ack"          # acknowledgment without commitment ("ok", "got it")
    CORRECTION   = "correction"   # patient wants to fix something prior
    PROVIDE_INFO = "provide_info" # patient is volunteering content
    UNCLEAR      = "unclear"      # cannot determine; caller should re-prompt


# Note: button labels live next to the response shape in api/patient.py
# (_quick_replies_for_state) because they're phase-specific ("Yes, I consent",
# "Keep on file", "Confirm").  This module deliberately exposes only the
# enum + token sets — a generic label table here would risk drift between
# the canonical UI strings and what the server actually accepts.


# ---------------------------------------------------------------------------
# Token sets — exact-match deterministic classifier
# ---------------------------------------------------------------------------

YES_TOKENS: frozenset[str] = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "mhm",
    "correct", "right", "confirm", "confirmed", "looks good",
    "that's right", "thats right", "sounds right", "go ahead", "proceed",
    "i agree", "agree", "consent",
})
NO_TOKENS: frozenset[str] = frozenset({
    "no", "n", "nope", "nah", "no thanks", "nah thanks",
    "decline", "i decline", "cancel", "stop",
})
ACK_TOKENS: frozenset[str] = frozenset({
    "ok", "okay", "k", "sure", "alright", "fine", "done",
    "got it", "sounds good", "thanks", "thank you",
})

# Note: "ok" / "okay" / "sure" appear in both YES_TOKENS and ACK_TOKENS by
# design — in confirm-style gates they read as YES, in subjective extraction
# they read as ACK.  parse_quick_reply prefers YES.

# Correction triggers — explicit go-back / fix language, not generic decline.
_CORRECTION_RE = re.compile(
    r"\b(go\s+back|start\s+over|change\s+my|fix\s+my|correct\s+my|"
    r"update\s+my|edit\s+my|modify\s+my|"
    r"(want|like|need)\s+to\s+(edit|change|fix|correct|update|modify)|"
    r"i\s+made\s+a\s+mistake|that('?s|\s+is)\s+(wrong|incorrect|not\s+right)|"
    r"actually\s+my|wait[,\s]+my|i\s+said\s+(the\s+)?wrong|"
    r"(wrong|incorrect)\s+answer|let\s+me\s+(change|correct|fix|edit)|"
    r"can\s+(i|we)\s+(change|correct|fix|edit|go\s+back))\b",
    re.IGNORECASE,
)

# Section regexes — used to figure out WHAT the patient wants to correct.
# Allow common plurals (symptoms, pains, headaches) — a previous version was
# singular-only and matched "fix my symptom" but not "fix my symptoms".
_IDENTITY_FIELDS_RE = re.compile(
    r"\b(names?|dob|date\s+of\s+birth|birthday|phones?|numbers?|address(es)?|contacts?)\b",
    re.IGNORECASE,
)
_SYMPTOM_FIELDS_RE = re.compile(
    r"\b(symptoms?|pains?|complaints?|onsets?|qualit(y|ies)|severit(y|ies)|"
    r"timings?|radiations?|provocations?|headaches?|hurts?|aches?)\b",
    re.IGNORECASE,
)
_HISTORY_FIELDS_RE = re.compile(
    r"\b(allerg(y|ies|ic)|med(ication|icine|s)?|histor(y|ies)|pmh|"
    r"surgeri(es)?|surgery|tests?|labs?|imaging|results?)\b",
    re.IGNORECASE,
)

# Granular history sub-field regexes — used by _try_correction to route to the
# specific clinical step the patient wants to change rather than always resetting
# to allergies.  Check in this order: more specific before more general.
_ALLERGY_FIELDS_RE  = re.compile(r"\b(allerg(y|ies|ic))\b", re.IGNORECASE)
_MEDS_FIELDS_RE     = re.compile(r"\b(med(ication|icine|s)?|prescription|pills?)\b", re.IGNORECASE)
_PMH_FIELDS_RE      = re.compile(r"\b(histor(y|ies)|pmh|condition(s)?|surgeri(es)?|surgery)\b", re.IGNORECASE)
_RESULTS_FIELDS_RE  = re.compile(r"\b(tests?|labs?|lab\s+work|imaging|results?|scans?|x-?rays?)\b", re.IGNORECASE)



# ---------------------------------------------------------------------------
# Deterministic parser — used for button clicks and obvious cases
# ---------------------------------------------------------------------------

def parse_quick_reply(text: str) -> Optional[QuickReply]:
    """
    Exact-token match against YES / NO / ACK. Returns None on no match.
    No prefix matching — "yes I have chest pain" must reach classify_intent().
    """
    t = _norm(text)
    if not t:
        return None
    if t in YES_TOKENS:
        return QuickReply.YES
    if t in NO_TOKENS:
        return QuickReply.NO
    if t in ACK_TOKENS:
        return QuickReply.ACK
    # Correction takes precedence even when it's the only thing in the message.
    if _CORRECTION_RE.search(text or ""):
        return QuickReply.CORRECTION
    return None


def is_bare_acknowledgment(text: str) -> bool:
    """
    True when the message is a short acknowledgment with no actual information.
    Returns False for NO_TOKENS ("no" means "no allergies" and must reach the extractor).
    Exact-match only — prefix matching ("ok I have penicillin") caused data loss.
    """
    return parse_quick_reply(text) in (QuickReply.YES, QuickReply.ACK)


def detect_correction_section(text: str) -> str:
    """
    For a CORRECTION intent, return which section the patient wants to fix.

    Returns one of: "identity", "symptoms", "history", "none".
    """
    if _IDENTITY_FIELDS_RE.search(text or ""):
        return "identity"
    if _SYMPTOM_FIELDS_RE.search(text or ""):
        return "symptoms"
    if _HISTORY_FIELDS_RE.search(text or ""):
        return "history"
    return "none"


# ---------------------------------------------------------------------------
# LLM-backed classifier — for short ambiguous messages
# ---------------------------------------------------------------------------

# Messages longer than this are almost never bare yes/no.  Fast-path them as
# provide_info without an LLM call.
_INTENT_MAX_WORDS_FOR_LLM = 8


def classify_intent(user: str, thread_id: str = "") -> IntentOut:
    """
    Two-tier: parse_quick_reply for exact tokens, LLM for short ambiguous messages.
    thread_id is for token-usage accounting only.
    """
    # Lazy imports avoid a circular import: nodes → intent → run_json_step → schemas.
    from . import sqlite_db as db
    from .llm import run_json_step
    from .prompts import intent_classify_system

    t = (user or "").strip()
    if not t:
        return IntentOut(intent="unclear", correcting_section="none")

    # Long messages are PROVIDE_INFO without LLM disambiguation
    if len(t.split()) > _INTENT_MAX_WORDS_FOR_LLM:
        return IntentOut(intent="provide_info", correcting_section="none")

    # Exact token match → done, no LLM
    quick = parse_quick_reply(t)
    if quick == QuickReply.YES:
        return IntentOut(intent="confirm", correcting_section="none")
    if quick == QuickReply.NO:
        return IntentOut(intent="decline", correcting_section="none")
    if quick == QuickReply.CORRECTION:
        return IntentOut(intent="correction",
                         correcting_section=detect_correction_section(t))

    # Tier 2: LLM for ambiguous short messages
    obj, meta = run_json_step(
        system=intent_classify_system(),
        prompt=f"PATIENT_MESSAGE={t}",
        schema=IntentOut,
        fallback={"intent": "unclear", "correcting_section": "none"},
        temperature=0.0,
        max_tokens=40,
        cache_key="intent_classify",
    )

    inp    = meta.get("input_tokens") or 0
    out    = meta.get("output_tokens") or 0
    cached = meta.get("cached_input_tokens") or 0
    if inp or out:
        db.record_llm_usage(
            thread_id=thread_id, node="intent_classify",
            input_tokens=inp, output_tokens=out,
            cached_input_tokens=cached,
        )
    return obj


