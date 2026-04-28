"""
intent.py — Single source of truth for patient-message intent classification.

Before this module existed, intent logic was scattered across:
  - extract.py: is_yes / is_no / is_ack / is_consent_accepted / is_consent_declined
  - nodes.py:  _HARD_YES / _HARD_NO / _classify_intent / _CORRECTION_RE / _IDENTITY_FIELDS_RE / ...

That fragmentation produced two real bugs:
  1. is_no("no problem")  matched via prefix and discarded substantive content
  2. The same patient message could be classified differently in different
     phases because each phase used a different helper.

This module consolidates everything into one classifier returning a single
QuickReply enum.  Callers either:
  - Use parse_quick_reply(text) for a deterministic, free, exact-token match
  - Use classify_intent(text, state) when LLM disambiguation is warranted

UI integration:
  Phases that ask binary questions (consent, identity_review, confirm) emit
  quick_replies in their response so the frontend renders buttons.  Button
  clicks send the canonical reply string ("Yes", "No", etc.) which always
  matches parse_quick_reply at zero LLM cost.  Free-text remains supported
  for accessibility.
"""
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
    Exact-token match against YES / NO / ACK.

    Returns None if the message doesn't match exactly.  Callers should then
    fall back to classify_intent() (LLM) for short ambiguous messages, or
    treat the message as PROVIDE_INFO for substantive input.

    Important: prefix matching ("yes I have chest pain") is NOT done here.
    A previous version used startswith("yes ") / startswith("no ") and
    misclassified substantive messages as binary replies, discarding the
    actual content.  Use classify_intent() to handle ambiguous text.
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
    True when the message is a single short acknowledgment with no actual
    information.  Used by clinical-history nodes to decide whether to
    re-show the prompt vs. send the message to the LLM extractor.

    Returns True for:
      - exact YES_TOKENS  ("yes", "ok", "sure", "go ahead")
      - exact ACK_TOKENS  ("thanks", "got it", "sounds good")
    Both signal "the patient acknowledged but didn't answer yet".

    Returns False for:
      - NO_TOKENS ("no") — in clinical history this means "no allergies /
        no PMH / no recent tests" and must reach the LLM extractor which
        produces an empty list, not a re-prompt.
      - Anything substantive ("ok I have penicillin", "yes I'm allergic
        to peanuts") — these carry real content and must go to the LLM
        extractor.  A previous "is_ack" with startswith matching
        misclassified those as bare acks and re-prompted, discarding the
        patient's answer.

    Trade-off (multi-word acks):
      Pure exact match means "ok thanks" / "ok sure" / "yeah ok thanks"
      do NOT short-circuit and instead get sent to the LLM extractor,
      costing one extra LLM call for a non-answer.  This is intentional:
      the alternative (prefix matching) caused real data loss on inputs
      like "ok I have penicillin allergy".  The LLM call is small
      (max_tokens=300) and the prompt's Example D handles the case
      gracefully (returns items_complete=False, asks the question
      again), so the UX is unchanged at modest extra cost.
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
    Two-tier classifier:
      Tier 1 (free): parse_quick_reply for exact tokens; fast-path long
                     messages to PROVIDE_INFO.
      Tier 2 (LLM):  short ambiguous messages ("I think so", "not really",
                     "hmm yeah") go through an LLM with the IntentOut schema.

    Returns an IntentOut model so callers can dispatch on .intent and
    .correcting_section.

    thread_id is used only for token-usage accounting (so per-session cost
    queries stay accurate).  Pass "" if you don't have one — accounting
    just isn't recorded for that call.
    """
    # Imports inside the function to avoid a circular import at module load:
    # nodes.py imports intent.py, and intent's LLM path imports run_json_step
    # which transitively touches schemas/prompts that some node modules also
    # use during graph construction.
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

    # Token-usage accounting matches the convention in nodes._track_llm_failure:
    # record any non-zero usage, even when thread_id is empty.  Empty-thread
    # rows are filtered out at the analytics layer (cost queries are filtered
    # by thread_id).  Doing it here would make this call site silently differ
    # from every other LLM call site.
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


# Note: parse_quick_reply / classify_intent are the only intent helpers in
# this codebase.  Earlier versions kept loose-prefix is_yes / is_no / is_ack
# in extract.py for backwards compatibility, but those have been removed —
# every call site goes through this module.
