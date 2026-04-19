"""
nodes.py — LangGraph conversation nodes.

Each node corresponds to one phase of the intake state machine.
All nodes return a dict of state fields to update (LangGraph merges them).
"""
from __future__ import annotations

import json
import re
from datetime import datetime as _dt
from typing import Dict, List

from .prompts import subjective_extract_system, meds_extract_system, report_system, identity_extract_system, intent_classify_system, PROMPT_VERSIONS
from .state import IntakeState
from .schemas import SubjectiveOut, MedsOut, ReportInputState, IdentityOut, IntentOut
from .llm import run_json_step, get_gemini, validate_llm_response
from .agentic import (
    adapt_clinical_question,
    score_extraction_quality,
    build_gap_fill_question,
    build_validation_gap_message,
)
from .safety import SafetyChecker, build_reason_trail
from .logging_utils import log_event
from . import sqlite_db as db
from . import fhir_builder
from . import webhook
from .integrations.fhir_client import push_bundle as _fhir_push_bundle
from .extract import (
    is_yes,
    is_no,
    is_ack,
    detect_emergency_red_flags,
    extract_allergies_simple,
    extract_list_simple,
    detect_crisis,
    has_soft_distress,
    llm_crisis_score,
    CRISIS_RESOURCE,
    CONSENT_MESSAGE,
    validate_dob,
)
from .settings import get_settings as settings
from .memory import format_for_prompt, merge_summary

RESPONSE_RULES = (
    "Be warm, empathetic, and human — patients may be anxious or in pain. "
    "Acknowledge what the patient shares before asking the next question. "
    "Ask ONE question only if needed. "
    "Never invent facts. "
    "No diagnosis. "
    "No medical advice."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def last_user(state: IntakeState) -> str:
    # Scan within the window only — the most recent user message is always
    # inside the last `messages_window_size` turns, so there is no need to
    # walk the full (potentially unbounded) history list.
    msgs = _window_messages(state)
    for m in reversed(msgs):
        if m["role"] == "user":
            return m["text"]
    return ""


def _window_messages(state: IntakeState) -> list:
    """
    Return the last N messages from state, where N = settings.intake.messages_window_size.

    Rationale: the messages list grows with every turn via LangGraph's operator.add
    reducer.  On a 50-turn intake that is 50 dicts passed verbatim into every LLM
    prompt.  Capping to the recent window keeps token usage predictable (O(1)
    instead of O(n)) without losing any data — the full history is always persisted
    in the messages DB table and the checkpointer.

    Only the windowed slice is sent to the LLM; older turns are never discarded
    from state itself.
    """
    n = settings().intake.messages_window_size
    msgs = state.get("messages") or []
    return msgs[-n:] if len(msgs) > n else msgs


def _summary_identity(x: Dict[str, str]) -> str:
    return (
        f"Name: {x.get('name') or '—'}, "
        f"DOB: {x.get('dob') or '—'}, "
        f"Phone: {x.get('phone') or '—'}, "
        f"Address: {x.get('address') or '—'}"
    )



# ---------------------------------------------------------------------------
# Intent classification — replaces hardcoded yes/no keyword lists
# ---------------------------------------------------------------------------

# Fast-path thresholds — messages longer than this are almost never bare yes/no,
# so we skip the LLM classifier and treat them as provide_info directly.
_INTENT_MAX_WORDS_FOR_LLM = 8

# Hard yes/no that are unambiguous even without LLM — exact matches only.
# Anything beyond this short list goes to the LLM.
_HARD_YES = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "mhm",
             "correct", "right", "confirm", "confirmed", "looks good", "that's right",
             "thats right", "sounds right", "go ahead", "proceed"}
_HARD_NO  = {"no", "n", "nope", "nah", "no thanks", "nah thanks"}


def _classify_intent(user: str, state: IntakeState) -> IntentOut:
    """
    Classify patient message intent using a two-tier approach.

    Tier 1 (free): exact match on a small set of unambiguous tokens.
      - Long messages (> 8 words) → provide_info without any LLM call.
      - Exact match in _HARD_YES → confirm.
      - Exact match in _HARD_NO  → decline.
      - Obvious correction regex  → correction + section detection.

    Tier 2 (LLM): short ambiguous messages only.
      - "I think so", "not really", "hmm yeah", "I suppose".
      - run_json_step with IntentOut schema + intent_classify_system.
      - Fast: max_tokens=40, temperature=0.0.
      - Falls back to "unclear" on LLM error — callers handle "unclear"
        by asking the patient to rephrase.

    This replaces is_yes() / is_no() / is_ack() everywhere they are used
    to classify intent (not to extract facts).
    """
    t = (user or "").strip()
    tl = t.lower()

    # Long messages are never bare yes/no
    if len(t.split()) > _INTENT_MAX_WORDS_FOR_LLM:
        return IntentOut(intent="provide_info")

    if tl in _HARD_YES:
        return IntentOut(intent="confirm")
    if tl in _HARD_NO:
        return IntentOut(intent="decline")

    # Obvious correction patterns — regex is fine for explicit language like
    # "go back", "change my name", "I said the wrong DOB".
    if _CORRECTION_RE.search(t):
        section = "none"
        if _IDENTITY_FIELDS_RE.search(t):
            section = "identity"
        elif _SYMPTOM_FIELDS_RE.search(t):
            section = "symptoms"
        elif _HISTORY_FIELDS_RE.search(t):
            section = "history"
        return IntentOut(intent="correction", correcting_section=section)

    # Tier 2: LLM for genuinely ambiguous short messages
    thread_id = (state or {}).get("thread_id", "")
    obj, meta = run_json_step(
        system=intent_classify_system(),
        prompt=f"PATIENT_MESSAGE={t}",
        schema=IntentOut,
        fallback={"intent": "unclear", "correcting_section": "none"},
        temperature=0.0,
        max_tokens=40,
    )
    inp = meta.get("input_tokens") or 0
    out = meta.get("output_tokens") or 0
    if inp or out:
        db.record_llm_usage(thread_id=thread_id, node="intent_classify",
                            input_tokens=inp, output_tokens=out)
    return obj


# ---------------------------------------------------------------------------
# Guard node — runs on every message before any business node
# ---------------------------------------------------------------------------

def guard_node(state: IntakeState):
    """
    Centralised safety and intent pre-processor.

    Runs automatically BEFORE every interactive node via LangGraph routing.
    Responsibilities:
      1. Crisis detection (Tier 1 keyword + Tier 2 LLM) with full side effects
         (DB escalation, Slack webhook). Crisis → routes to handoff regardless
         of current phase. Identity is never required for a safety response.
      2. (Future) global abuse/prompt-injection detection.

    Returns a state patch. If no crisis is detected, returns {} so the router
    can proceed to the normal business node. The node itself does not ask the
    patient any question — it either intercepts (crisis) or is transparent.

    Why this is a node and not a function called inside each business node:
      - A dedicated node means adding a new business node tomorrow never
        accidentally skips safety.
      - Side effects (DB writes, webhooks) happen in exactly one place.
      - Testable in isolation without running the full state machine.
    """
    user = last_user(state).strip()
    if not user:
        return {}   # nothing to check — first turn or empty message

    thread_id    = state.get("thread_id", "")
    patient_name = (state.get("identity") or {}).get("name") or "unknown patient"

    # ── Tier 1: keyword / regex ──────────────────────────────────────────
    crisis_flags = detect_crisis(user)
    if crisis_flags:
        log_event("crisis_detected_tier1", level="warning",
                  thread_id=thread_id, matched_phrases=crisis_flags)
        db.create_escalation(
            thread_id=thread_id, kind="crisis",
            payload=build_reason_trail(
                "crisis", state,
                extra_data={"matched_phrases": crisis_flags, "detection_tier": "keyword"},
            ),
        )
        webhook.dispatch_crisis_alert(
            thread_id=thread_id,
            patient_name=patient_name,
            matched_phrases=crisis_flags,
            partial_identity=state.get("identity") or {},
            message_preview=user,
        )
        return {
            "crisis_detected":      True,
            "human_review_required": True,
            "current_phase":        "handoff",
            "messages": [{"role": "assistant", "text": CRISIS_RESOURCE}],
        }

    # ── Tier 2: LLM classifier for soft distress signals ─────────────────
    if has_soft_distress(user):
        score = llm_crisis_score(user)
        if score.is_crisis_risk and score.confidence in ("high", "medium"):
            log_event("crisis_detected_tier2", level="warning",
                      thread_id=thread_id,
                      confidence=score.confidence, reasoning=score.reasoning)
            db.create_escalation(
                thread_id=thread_id, kind="crisis",
                payload=build_reason_trail(
                    "crisis", state,
                    extra_data={"matched_phrases": [f"llm:{score.reasoning}"],
                                "detection_tier": "llm",
                                "llm_confidence": score.confidence},
                ),
            )
            webhook.dispatch_crisis_alert(
                thread_id=thread_id, patient_name=patient_name,
                matched_phrases=[f"llm_detected ({score.confidence}): {score.reasoning}"],
                partial_identity=state.get("identity") or {},
                message_preview=user,
            )
            return {
                "crisis_detected":       True,
                "human_review_required": True,
                "current_phase":         "handoff",
                "messages": [{"role": "assistant", "text": CRISIS_RESOURCE}],
            }
        if score.is_crisis_risk and score.confidence == "low":
            log_event("soft_distress_flagged", level="info",
                      thread_id=thread_id, reasoning=score.reasoning)

    return {}   # no crisis — proceed to business node


# ---------------------------------------------------------------------------
# Global correction intent — shared by every interactive node
# ---------------------------------------------------------------------------

_CORRECTION_RE = re.compile(
    r"\b(go\s+back|start\s+over|change\s+my|fix\s+my|correct\s+my|"
    r"update\s+my|i\s+made\s+a\s+mistake|that('?s|\s+is)\s+(wrong|incorrect|not\s+right)|"
    r"actually\s+my|wait[,\s]+my|i\s+said\s+(the\s+)?wrong|let\s+me\s+(change|correct|fix)|"
    r"can\s+i\s+(change|correct|fix|go\s+back))\b",
    re.IGNORECASE,
)

_IDENTITY_FIELDS_RE = re.compile(
    r"\b(name|dob|date\s+of\s+birth|birthday|phone|number|address|contact)\b",
    re.IGNORECASE,
)
_SYMPTOM_FIELDS_RE = re.compile(
    r"\b(symptom|pain|complaint|onset|quality|severity|timing|radiation|provocation|headache|hurt|ache)\b",
    re.IGNORECASE,
)
_HISTORY_FIELDS_RE = re.compile(
    r"\b(allerg|med(ication|icine|s)?|history|pmh|surgeri|surgery|test|lab|imaging|result)\b",
    re.IGNORECASE,
)


def _try_correction(user: str, state: IntakeState) -> dict | None:
    """
    Detect patient intent to correct previously entered information from any node.

    Returns a state-patch dict with the corrected phase + a helpful message,
    or None if the message is not a correction intent.

    Design: runs AFTER crisis check, BEFORE normal extraction, in every
    interactive node. Only fires when the message contains an explicit correction
    signal (go back, change my X, I said the wrong X) so it never accidentally
    intercepts normal intake answers.
    """
    if not _CORRECTION_RE.search(user):
        return None

    # Determine what they want to change
    if _IDENTITY_FIELDS_RE.search(user):
        return {
            "current_phase": "identity",
            "identity_attempts": 0,
            "messages": [{"role": "assistant", "text":
                "Of course — what would you like to update? "
                "You can share your name, date of birth, phone, or address in any format."}],
        }
    if _SYMPTOM_FIELDS_RE.search(user):
        return {
            "current_phase": "subjective",
            "messages": [{"role": "assistant", "text":
                "No problem — what would you like to change about your symptoms?"}],
        }
    if _HISTORY_FIELDS_RE.search(user):
        return {
            "current_phase": "clinical_history",
            "clinical_step": "allergies",
            "medications": None,
            "pmh": None,
            "recent_results": None,
            "messages": [{"role": "assistant", "text":
                "Sure — let's go through your health history again. "
                "Do you have any allergies?"}],
        }
    # Generic go-back without specifying what — show a menu
    phase = state.get("current_phase") or "identity"
    _phase_labels = {
        "subjective": "symptoms", "clinical_history": "health history",
        "confirm": "review", "identity": "contact details",
    }
    options = [
        "your contact details (name, DOB, phone, address)",
        "your symptoms",
        "your health history (allergies, medications, past conditions)",
    ]
    return {
        "current_phase": phase,   # stay in current phase until they specify
        "messages": [{"role": "assistant", "text":
            "Of course — what would you like to go back and change?\n" +
            "\n".join(f"  • {o}" for o in options)}],
    }


_HOLLOW_PREFIX_RE = re.compile(
    r"^(?:yes[,!\.]*\s*|yeah[,!\.]*\s*|sure[,!\.]*\s*|absolutely[,!\.]*\s*|"
    r"of\s+course[,!\.]*\s*|okay[,!\.]*\s*|ok[,!\.]*\s*|"
    r"i\s+see[,!\.]*\s*|i\s+understand[,!\.]*\s*|understood[,!\.]*\s*|"
    r"noted[,!\.]*\s*|great[,!\.]*\s*|got\s+it[,!\.]*\s*)+",
    re.IGNORECASE,
)


def _clean_reply(text: str) -> str:
    """
    Strip hollow leading affirmatives from LLM reply.

    Prevents "Yes, yes, I understand. When did it start?" → keeps only the actual question.
    Empathy framed *within* the question (e.g. "I'm sorry to hear that — when did it start?")
    is preserved because it doesn't match the hollow-prefix pattern.
    """
    cleaned = _HOLLOW_PREFIX_RE.sub("", (text or "")).strip()
    # Capitalise after stripping prefix
    return cleaned[0].upper() + cleaned[1:] if cleaned else text


def _safe_reply(text: str) -> str:
    """Run LLM response through the diagnosis-language guardrail and hollow-affirmative strip."""
    safe, _ = validate_llm_response(text)
    return _clean_reply(safe)


def _track_llm_failure(thread_id: str, node: str, meta: dict) -> None:
    """
    Persist LLM failure + token usage for one run_json_step call.

    Token usage is always recorded regardless of success/failure so billing
    can aggregate per-session costs accurately.
    Does not raise — tracking must never break the happy path.
    """
    # Always record token usage (even on clean runs).
    inp = meta.get("input_tokens") or 0
    out = meta.get("output_tokens") or 0
    if inp or out:
        db.record_llm_usage(thread_id=thread_id, node=node,
                            input_tokens=inp, output_tokens=out)

    if not (meta.get("fallback_used") or meta.get("repair_used")):
        return
    failure_type = "fallback_used" if meta.get("fallback_used") else "repair_used"
    try:
        db.record_llm_failure(
            thread_id=thread_id,
            node=node,
            failure_type=failure_type,
            raw_snippet=(meta.get("raw_preview") or "")[:300],
            error_detail=meta.get("parse_error") or meta.get("llm_error") or "",
        )
    except Exception as exc:
        log_event("llm_failure_log_error", level="warning",
                  thread_id=thread_id, node=node, error=str(exc)[:200])


def _cost_patch(state: IntakeState, meta: dict) -> dict:
    """
    Return a state patch accumulating this call's cost into session_cost_usd.

    Also logs a warning when the session exceeds the configured cap so ops can
    see which sessions are outliers without waiting for a billing surprise.
    Does not block — the cap check that enforces the limit lives in /chat.
    """
    call_cost = float(meta.get("cost_usd") or 0.0)
    if not call_cost:
        return {}
    current   = float(state.get("session_cost_usd") or 0.0)
    new_total = round(current + call_cost, 8)
    cap       = settings().intake.max_cost_usd_per_session
    if new_total > cap:
        log_event(
            "session_cost_cap_exceeded",
            level="warning",
            thread_id=state.get("thread_id", ""),
            session_cost_usd=new_total,
            cap_usd=cap,
        )
    return {"session_cost_usd": new_total}


def _failure_state_patch(meta: dict, phase: str) -> dict:
    """
    Return a state-patch dict with last_failed_phase / last_failure_reason
    when meta indicates an LLM degradation.  Returns {} when the LLM was clean.
    """
    if meta.get("fallback_used"):
        return {
            "last_failed_phase": phase,
            "last_failure_reason": f"fallback_used: {(meta.get('parse_error') or 'unknown')[:120]}",
        }
    if meta.get("repair_used"):
        return {
            "last_failed_phase": phase,
            "last_failure_reason": f"repair_used: {(meta.get('parse_error') or 'unknown')[:120]}",
        }
    return {}


def _build_validated_report_state(state: IntakeState) -> ReportInputState:
    """
    Extract and validate all fields consumed by report_node and fhir_builder.

    Constructs a ReportInputState from the raw IntakeState, applying Pydantic
    bounds and coercion.  Both the clinician note and the FHIR bundle are built
    from the returned model — never from raw state — proving that every artifact
    is generated from validated structured state, not raw chat content.

    Never raises: validation failures fall back to field defaults (empty strings,
    empty lists).  Any error is logged as a warning.
    """
    try:
        return ReportInputState(
            identity=state.get("identity") or {},
            chief_complaint=state.get("chief_complaint") or "",
            opqrst=state.get("opqrst") or {},
            allergies=state.get("allergies") or [],
            medications=state.get("medications") or [],
            pmh=state.get("pmh") or [],
            recent_results=state.get("recent_results") or [],
            triage=state.get("triage") or {},
        )
    except Exception as exc:
        log_event("report_state_validation_error", level="warning",
                  thread_id=state.get("thread_id", ""), error=str(exc)[:300])
        return ReportInputState()


# ---------------------------------------------------------------------------
# Consent node (only active when settings.require_consent = True)
# ---------------------------------------------------------------------------

def consent_node(state: IntakeState):
    user = last_user(state).strip()

    if not user:
        return {
            "messages": [{"role": "assistant", "text": CONSENT_MESSAGE}],
            "current_phase": "consent",
        }

    # Use _classify_intent for consent — catches "sure go ahead", "I consent",
    # "yeah that's fine", "nah", "I don't want to" without keyword lists.
    intent = _classify_intent(user, state)

    if intent.intent == "confirm":
        log_event("patient_consented", thread_id=state.get("thread_id"))
        return {
            "consent_given": True,
            "messages": [{"role": "assistant", "text":
                "Thank you. To get started, could you share your full name, date of birth, "
                "phone number, and home address? Any format works."}],
            "current_phase": "identity",
        }

    if intent.intent == "decline":
        thread_id = state.get("thread_id", "")
        log_event("patient_declined_consent", thread_id=thread_id)
        if thread_id:
            db.set_session_status(thread_id, "done")
        return {
            "consent_given": False,
            "messages": [{"role": "assistant", "text":
                "Understood. We're sorry we couldn't help today. "
                "Please speak with the front desk when you arrive. "
                "Have a safe visit."}],
            "current_phase": "done",
        }

    return {
        "messages": [{"role": "assistant", "text":
            "Please reply 'yes' to consent and continue, or 'no' to decline. "
            "You can also speak with the front desk directly."}],
        "current_phase": "consent",
    }


# ---------------------------------------------------------------------------
# Identity node
# ---------------------------------------------------------------------------

def identity_node(state: IntakeState):
    user      = last_user(state).strip()
    identity  = dict(state.get("identity") or {"name": "", "dob": "", "phone": "", "address": ""})
    attempts  = int(state.get("identity_attempts") or 0)
    thread_id = state.get("thread_id", "")

    if attempts == 0 and not user:
        return {
            "messages": [{"role": "assistant", "text":
                "To get started, could you share your full name, date of birth, "
                "phone number, and home address? "
                "Feel free to share them all at once or one at a time — any format works."}],
            "current_phase": "identity",
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": False,
        }

    # Go-back intent — "change my address", "I said the wrong name", etc.
    if user and attempts > 0:
        correction = _try_correction(user, state)
        if correction:
            return correction

    # LLM extraction — handles misspellings, ordinal dates, natural phrasing,
    # and multi-field messages that regex cannot parse.
    obj, meta = run_json_step(
        system=identity_extract_system(),
        prompt=f"PATIENT_MESSAGE={user}",
        schema=IdentityOut,
        fallback={"name": "", "dob": "", "phone": "", "address": ""},
        temperature=0.1,
        max_tokens=120,
    )
    id_version, id_exp_id = db.resolve_prompt_variant(
        thread_id, "identity", PROMPT_VERSIONS.get("identity", "")
    )
    log_event("llm_step", thread_id=thread_id, node="identity",
              prompt_version=id_version, experiment_id=id_exp_id, **meta)
    _track_llm_failure(thread_id, "identity", meta)

    extracted = obj.model_dump()  # already normalised: Title Case, ISO 8601, 10-digit phone

    # Merge: only fill fields not yet collected in this session
    for field in ["name", "dob", "phone", "address"]:
        val = (extracted.get(field) or "").strip()
        if val and not (identity.get(field) or "").strip():
            identity[field] = val

    # Sanity-check DOB (future date, impossible age) — validate_dob accepts ISO 8601
    dob_raw = (identity.get("dob") or "").strip()
    if dob_raw:
        dob_clean, dob_err = validate_dob(dob_raw)
        if dob_err:
            identity["dob"] = ""
            return {
                "identity": identity,
                "identity_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text":
                    f"I didn't quite catch your date of birth — {dob_err.lower()} "
                    "Any format works, like '15 March 1985' or '1985-03-15'."}],
                "current_phase": "identity",
            }
        # validate_dob returns MM/DD/YYYY — keep storage as ISO 8601 for consistency
        try:
            identity["dob"] = _dt.strptime(dob_clean, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass  # already ISO 8601 if strptime failed (shouldn't happen)

    attempts += 1
    missing = [k for k in ["name", "dob", "phone", "address"] if not (identity.get(k) or "").strip()]

    if missing:
        if attempts >= settings().intake.max_identity_attempts:
            log_event("identity_max_attempts", level="warning", thread_id=thread_id)
            return {
                "identity": identity,
                "identity_attempts": attempts,
                "identity_status": "unverified",
                "messages": [{"role": "assistant", "text":
                    "I'm having trouble capturing your details. "
                    "Please speak with the front desk when you arrive — "
                    "they'll complete your intake directly."}],
                "current_phase": "done",
            }
        q = {
            "name":    "What's your full name?",
            "dob":     "What's your date of birth? Any format works — for example '15 March 1985' or '1985-03-15'.",
            "phone":   "What's the best phone number to reach you?",
            "address": "What's your home address?",
        }[missing[0]]
        return {
            "identity": identity,
            "identity_attempts": attempts,
            "identity_status": "unverified",
            "messages": [{"role": "assistant", "text": q}],
            "current_phase": "identity",
        }

    stored = db.get_stored_identity_by_name(identity["name"])
    if stored:
        name    = (stored.get("name")    or "").strip()
        phone   = (stored.get("phone")   or "").strip()
        address = (stored.get("address") or "").strip()
        details = ""
        if phone:
            details += f" Your phone on file is {phone}."
        if address:
            details += f" Address: {address}."
        return {
            "identity": identity,
            "stored_identity": stored,
            "messages": [{"role": "assistant", "text":
                f"Welcome back{', ' + name if name else ''}!{details} "
                "Does everything still look right, or has anything changed? "
                "(reply 'yes' to keep, 'no' to update)"
            }],
            "current_phase": "identity_review",
        }

        # Derive a stable patient_id now that name + dob are both captured.
    # All future retrieval (prior visits, patient summary) keys off this.
    pid = db.derive_patient_id(identity.get("name", ""), identity.get("dob", ""))
    prior_summary = None
    if pid:
        db.set_session_patient_id(thread_id, pid)
        prior_summary = db.get_patient_summary(pid)
        if prior_summary:
            log_event("returning_patient_loaded", thread_id=thread_id,
                      patient_id=pid, visit_count=prior_summary["visit_count"])

    return {
        "identity": identity,
        "stored_identity": None,
        "identity_status": "unverified",
        "needs_identity_review": True,
        "messages": [{"role": "assistant", "text":
            f"Got it. I have: {_summary_identity(identity)}. Is this correct? (yes / no)"}],
        "current_phase": "identity_review",
        **_cost_patch(state, meta),
    }


# ---------------------------------------------------------------------------
# Identity review node
# ---------------------------------------------------------------------------

def identity_review_node(state: IntakeState):
    user      = last_user(state).strip()
    identity  = dict(state.get("identity") or {})
    stored    = state.get("stored_identity")
    thread_id = state.get("thread_id", "")

    if stored:
        # For stored-identity decisions the patient must choose "keep" or "update".
        # Use _classify_intent: "confirm" → keep, "decline" → update, correction → identity.
        intent = _classify_intent(user, state)
        if intent.intent == "confirm":
            return {
                "identity": stored,
                "identity_status": "verified",
                "needs_identity_review": False,
                "messages": [{"role": "assistant", "text":
                    "Thanks — I'll keep what's on file. What brings you in today?"}],
                "current_phase": "subjective",
            }
        if intent.intent == "decline":
            db.create_escalation(
                thread_id=thread_id,
                kind="identity_review",
                payload=build_reason_trail(
                    "identity_review",
                    {**state, "identity": identity, "stored_identity": stored,
                     "current_phase": "identity_review"},
                    extra_data={"stored_identity": stored, "new_identity": identity},
                ),
            )
            name = (identity or {}).get("name") or "unknown patient"
            log_event("identity_mismatch_escalated", thread_id=thread_id,
                      patient=name[:40])
            return {
                "identity": identity,
                "identity_status": "unverified",
                "needs_identity_review": True,
                "messages": [{"role": "assistant", "text":
                    "Understood — I'll use what you provided. A nurse may follow up to verify. "
                    "What brings you in today?"}],
                "current_phase": "subjective",
            }
        return {
            "messages": [{"role": "assistant", "text":
                "Would you like to keep the information on file, or use what you provided? "
                "(reply 'keep' or 'update')"}],
            "current_phase": "identity_review",
        }

    # No stored record — patient is confirming or correcting what they just provided.
    intent = _classify_intent(user, state)
    if intent.intent == "confirm":
        return {
            "identity_status": "verified",
            "needs_identity_review": False,
            "messages": [{"role": "assistant", "text":
                "Thanks. What's the main reason for your visit today?"}],
            "current_phase": "subjective",
        }
    if intent.intent in ("decline", "correction"):
        return {
            "identity": {"name": "", "dob": "", "phone": "", "address": ""},
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": True,
            "messages": [{"role": "assistant", "text":
                "No problem — let's fix that. What's your full name?"}],
            "current_phase": "identity",
        }
    # unclear — ask them to confirm explicitly
    return {
        "messages": [{"role": "assistant", "text":
            "Just to confirm — does the information I have look right? "
            "Reply 'yes' to continue or 'no' to re-enter your details."}],
        "current_phase": "identity_review",
    }


# ---------------------------------------------------------------------------
# Needs-ED-followup heuristic
# ---------------------------------------------------------------------------

def needs_ed_followup(cc: str, _op: Dict[str, str]) -> bool:
    t = (cc or "").lower()
    breathing = any(k in t for k in ["shortness of breath", "sob", "difficulty breathing"])
    neuro     = any(k in t for k in ["weakness", "numbness", "slurred speech", "face droop", "confusion"])
    faint     = any(k in t for k in ["faint", "passed out", "syncope"])
    if breathing or neuro or faint:
        return False
    chest = any(k in t for k in ["chest pain", "chest tight", "chest pressure", "pressure in chest"])
    return chest


def _severity_score(op: Dict[str, str]) -> int:
    s = (op.get("severity") or "").lower()
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        try:
            n = int(m.group(1))
            if 0 <= n <= 10:
                return n
        except (ValueError, TypeError):
            pass
    if "severe" in s or "worst" in s: return 9
    if "moderate" in s: return 5
    if "mild" in s: return 2
    return -1


def compute_basic_triage(mode: str, cc: str, op: Dict[str, str]) -> Dict:
    base = {
        "emergency_flag": False,
        "risk_level": "low",
        "visit_type": "routine",
        "confidence": "medium",
        "rationale": "No emergency red flags detected.",
        "red_flags": [],
    }
    if mode != "ed":
        return base

    t   = (cc or "").lower()
    sev = _severity_score(op)
    concerning = any(k in t for k in [
        "chest", "short of breath", "difficulty breathing", "faint", "passed out",
        "severe", "worst headache", "blood", "bleeding", "vision", "confusion",
    ])

    if sev >= 7 or concerning:
        return {**base,
                "risk_level": "medium",
                "visit_type": "urgent_care_today",
                "rationale": "High severity or concerning symptoms — evaluation today recommended."}
    if 0 <= sev <= 3:
        return {**base,
                "risk_level": "low",
                "visit_type": "clinic_24_72h",
                "rationale": "Lower severity, no emergency keywords. Clinic follow-up within 24-72h."}
    return {**base,
            "risk_level": "low",
            "visit_type": "clinic_24_72h",
            "confidence": "low",
            "rationale": "Severity unclear. Clinic follow-up within 24-72h unless symptoms worsen."}


# ---------------------------------------------------------------------------
# Subjective node
# ---------------------------------------------------------------------------

def subjective_node(state: IntakeState):
    user = last_user(state).strip()
    cc   = (state.get("chief_complaint") or "").strip()
    op   = dict(state.get("opqrst") or
                {"onset": "", "provocation": "", "quality": "", "radiation": "", "severity": "", "timing": ""})
    thread_id = state.get("thread_id", "")

    if not cc and (not user or is_ack(user) or is_yes(user) or is_no(user)):
        return {
            "messages": [{"role": "assistant", "text": "What's the main reason for your visit today? (in your own words)"}],
            "current_phase": "subjective",
        }


    # Go-back intent — patient wants to correct identity or history from this phase
    if user:
        correction = _try_correction(user, state)
        if correction and correction.get("current_phase") != "subjective":
            return correction

    # Emergency red flag detection
    flags = detect_emergency_red_flags(cc, op, user)
    if flags:
        triage = {
            "emergency_flag": True,
            "risk_level": "high",
            "visit_type": "emergency",
            "red_flags": flags,
            "confidence": "high",
            "rationale": "Red-flag phrase detected in patient input.",
        }
        db.create_escalation(
            thread_id=thread_id,
            kind="emergency",
            payload=build_reason_trail(
                "emergency",
                {**state, "triage": triage, "current_phase": "subjective"},
                extra_data={"red_flags": flags, "triage": triage},
            ),
        )
        db.set_session_status(thread_id, "escalated")
        log_event("emergency_escalation", level="warning",
                  thread_id=thread_id, red_flags=flags)

        patient_name = (state.get("identity") or {}).get("name") or "unknown patient"
        webhook.dispatch_emergency_alert(
            thread_id=thread_id,
            patient_name=patient_name,
            red_flags=flags,
            session_short=thread_id[:8],
        )
        return {
            "triage": triage,
            "needs_emergency_review": True,
            "messages": [{"role": "assistant", "text":
                "Based on what you shared, this could be urgent. "
                "Please call 911 or go to the nearest emergency room now. "
                "A clinician has been notified."}],
            "current_phase": "handoff",
        }

    # LLM extraction
    prior_ctx = format_for_prompt(state.get("prior_summary") or {})
    prompt_parts = []
    if prior_ctx:
        prompt_parts.append(prior_ctx)
    prompt_parts.append(f"CURRENT_STATE={json.dumps({'chief_complaint': cc, 'opqrst': op})}")
    prompt_parts.append(f"NEW_USER_MESSAGE={user}")
    prompt = "\n\n".join(prompt_parts)

    obj, meta = run_json_step(
        system=subjective_extract_system(RESPONSE_RULES),
        prompt=prompt,
        schema=SubjectiveOut,
        fallback={
            "chief_complaint": cc[:300],
            "opqrst": {k: (v or "")[:150] for k, v in op.items()},
            "is_complete": False,
            "reply": "When did it start, and how severe is it from 0-10?",
            "extraction_confidence": "low",
        },
        temperature=0.2,
    )
    resolved_version, exp_id = db.resolve_prompt_variant(
        thread_id, "subjective", PROMPT_VERSIONS.get("subjective", "")
    )
    log_event("llm_step", thread_id=thread_id, node="subjective",
              prompt_version=resolved_version,
              experiment_id=exp_id, **meta)
    _track_llm_failure(thread_id, "subjective", meta)

    out    = obj.model_dump()
    new_cc = (out.get("chief_complaint") or "").strip()
    new_op = out.get("opqrst") or {}
    llm_extraction_confidence = out.get("extraction_confidence") or "medium"

    # Feature 1: classification is now embedded in SubjectiveOut — no second LLM call.
    classification = state.get("intake_classification")
    classification_confidence = state.get("classification_confidence")
    if new_cc:
        cc = new_cc
        # Use classification returned by the combined extraction call when available.
        llm_cls = out.get("intake_classification")
        llm_cls_conf = out.get("classification_confidence")
        if llm_cls and not classification:
            classification = llm_cls
            classification_confidence = llm_cls_conf
            log_event("intake_classified", thread_id=thread_id,
                      classification=classification, confidence=classification_confidence,
                      source="combined_llm_call")

    for k in op.keys():
        v = (new_op.get(k) or "").strip()
        if v and not (op.get(k) or "").strip():
            op[k] = v

    complete = bool(out.get("is_complete"))
    raw_reply = _safe_reply((out.get("reply") or "").strip())
    # Guard 1: a reply that contains no "?" is not a real question — use deterministic gap-fill instead.
    # This catches cases where the LLM emits an acknowledgment ("I see." / "I understand.") with
    # no actual question, which would leave the patient with no action to take.
    if raw_reply and "?" not in raw_reply:
        log_event("llm_reply_no_question", level="warning",
                  thread_id=thread_id, preview=raw_reply[:120])
        raw_reply = build_gap_fill_question(cc, op, classification or "routine_checkup")
    # Guard 2: if the LLM itself reports low confidence, don't trust its reply — it likely
    # misunderstood the patient. A deterministic gap-fill is more reliable here.
    if llm_extraction_confidence == "low" and not complete:
        log_event("extraction_confidence_low_gap_fill", thread_id=thread_id,
                  llm_reply_preview=(raw_reply or "")[:80])
        raw_reply = build_gap_fill_question(cc, op, classification or "routine_checkup")
    reply = raw_reply

    if complete:
        mode     = state.get("mode") or "clinic"
        attempts = int(state.get("triage_attempts") or 0)

        if mode == "ed" and attempts < 1 and needs_ed_followup(cc, op):
            return {
                "chief_complaint": cc,
                "opqrst": op,
                "intake_classification": classification,
                "classification_confidence": classification_confidence,
                "subjective_complete": False,
                "triage_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text":
                    "Quick safety check: are you having shortness of breath, "
                    "fainting, sweating, or pain spreading to your arm or jaw?"}],
                "current_phase": "subjective",
            }

        # Feature 3: quality gate — retry with targeted gap-fill before advancing
        _cfg = settings().intake
        QUALITY_THRESHOLD = _cfg.ed_quality_threshold if mode == "ed" else _cfg.clinic_quality_threshold
        MAX_RETRY = _cfg.max_quality_retries
        retry_count = int(state.get("extraction_retry_count") or 0)
        quality_score = score_extraction_quality(cc, op)

        if quality_score < QUALITY_THRESHOLD and retry_count < MAX_RETRY:
            gap_q = build_gap_fill_question(cc, op, classification or "routine_checkup")
            log_event("extraction_quality_retry", thread_id=thread_id,
                      score=quality_score, retry=retry_count + 1)
            return {
                "chief_complaint": cc,
                "opqrst": op,
                "intake_classification": classification,
                "classification_confidence": classification_confidence,
                "extraction_quality_score": quality_score,
                "extraction_retry_count": retry_count + 1,
                "extraction_confidence": llm_extraction_confidence,
                "subjective_complete": False,
                "messages": [{"role": "assistant", "text": gap_q}],
                "current_phase": "subjective",
            }

        triage = compute_basic_triage(mode, cc, op)
        # Feature 4: route through validate_node before advancing to clinical_history
        return {
            "chief_complaint": cc,
            "opqrst": op,
            "intake_classification": classification,
            "classification_confidence": classification_confidence,
            "extraction_quality_score": score_extraction_quality(cc, op),
            "extraction_retry_count": retry_count,
            "extraction_confidence": llm_extraction_confidence,
            "triage": triage,
            "needs_emergency_review": False,
            "subjective_complete": True,
            "subjective_incomplete_turns": 0,   # reset on successful completion
            "clinical_step": "allergies",
            "validation_target_phase": "clinical_history",
            "validation_errors": [],
            "current_phase": "validate",
        }

    # On 3rd+ incomplete turn, generate a deterministic progress message so the
    # patient knows exactly what was understood vs what is still missing.
    # This prevents the LLM from asking the same generic question repeatedly.
    incomplete_turns = int(state.get("subjective_incomplete_turns") or 0)
    if incomplete_turns >= 2 and not complete:
        have = []
        if cc:                          have.append(f"complaint ({cc})")
        if (op.get("onset") or "").strip():   have.append(f"onset ({op['onset']})")
        if (op.get("severity") or "").strip(): have.append(f"severity ({op['severity']})")
        if (op.get("quality") or "").strip():  have.append(f"how it feels ({op['quality']})")
        still_need = []
        if not (op.get("severity") or "").strip():
            still_need.append("how severe it is (e.g. '7 out of 10', or say mild / moderate / severe)")
        if not (op.get("onset") or "").strip() and not (op.get("timing") or "").strip():
            still_need.append("when it started (e.g. 'this morning', '2 days ago')")
        if not cc:
            still_need.append("what brought you in today")
        preamble = ("I have " + ", ".join(have) + ". ") if have else ""
        if still_need:
            best_reply = preamble + "I still need: " + still_need[0] + "?"
        else:
            best_reply = reply or build_gap_fill_question(cc, op, classification or "routine_checkup")
        log_event("subjective_stuck_guidance", thread_id=thread_id,
                  incomplete_turns=incomplete_turns, still_need=still_need)
    else:
        best_reply = reply or build_gap_fill_question(cc, op, classification or "routine_checkup")

    return {
        "chief_complaint": cc,
        "opqrst": op,
        "intake_classification": classification,
        "classification_confidence": classification_confidence,
        "extraction_confidence": llm_extraction_confidence,
        "subjective_incomplete_turns": incomplete_turns + 1,
        "subjective_complete": False,
        "messages": [{"role": "assistant", "text": best_reply}],
        "current_phase": "subjective",
        **_failure_state_patch(meta, "subjective"),
        **_cost_patch(state, meta),
    }


# ---------------------------------------------------------------------------
# Confirm summary helper
# ---------------------------------------------------------------------------

def _confirm_summary(state: IntakeState) -> str:
    identity  = state.get("identity") or {}
    cc        = state.get("chief_complaint") or "your concern"
    op        = state.get("opqrst") or {}
    allergies = state.get("allergies")
    meds      = state.get("medications")
    pmh       = state.get("pmh")
    results   = state.get("recent_results")

    name = (identity.get("name") or "").strip()
    dob  = (identity.get("dob")  or "").strip()
    phone = (identity.get("phone") or "").strip()

    # --- identity line ---
    id_parts = []
    if name:  id_parts.append(name)
    if dob:   id_parts.append(f"DOB {dob}")
    if phone: id_parts.append(f"reachable at {phone}")
    id_line = ", ".join(id_parts) if id_parts else "your details on file"

    # --- symptom sentence ---
    onset    = (op.get("onset")    or "").strip()
    quality  = (op.get("quality")  or "").strip()
    severity = (op.get("severity") or "").strip()
    timing   = (op.get("timing")   or "").strip()
    radiation = (op.get("radiation") or "").strip()

    sym_parts = [f"you came in for {cc}"]
    if onset:    sym_parts.append(f"started {onset}")
    if quality:  sym_parts.append(f"described as {quality}")
    if severity:
        # Only append /10 when severity is numeric (e.g. "6/10", "7") — not for
        # descriptive values like "moderate" which are valid per the prompt schema.
        sym_parts.append(
            f"severity {severity}/10"
            if re.search(r"\d", severity) and "/10" not in severity
            else f"severity {severity}"
        )
    if timing:   sym_parts.append(f"occurring {timing}")
    if radiation and radiation.lower() not in ("none", "n/a", "no"):
        sym_parts.append(f"radiating to {radiation}")
    sym_sentence = ", ".join(sym_parts) + "."

    # --- history lines ---
    def _fmt_list(xs) -> str:
        return "none reported" if not xs else ", ".join(xs)

    def _fmt_meds(ms) -> str:
        if not ms:
            return "none reported"
        parts = []
        for m in ms:
            s = (m.get("name") or "Unknown").strip()
            if m.get("dose"): s += f" {m['dose']}"
            if m.get("freq"): s += f" ({m['freq']})"
            parts.append(s)
        return "; ".join(parts)

    allergy_line  = _fmt_list(allergies)
    med_line      = _fmt_meds(meds)
    pmh_line      = _fmt_list(pmh)
    results_line  = _fmt_list(results)

    return (
        f"Here's a quick summary of what I've captured — please let me know if anything looks off.\n\n"
        f"Patient: {id_line}. Reason for visit: {sym_sentence}\n\n"
        f"Allergies: {allergy_line}. "
        f"Current medications: {med_line}. "
        f"Past medical history: {pmh_line}. "
        f"Recent tests: {results_line}."
    )


# ---------------------------------------------------------------------------
# Clinical history node
# ---------------------------------------------------------------------------

def _prescan_volunteered_clinical(text: str) -> dict:
    """
    Lookahead: if the patient volunteers negative info about upcoming steps
    in the current message, pre-fill those fields so we can skip the questions.

    Returns a partial state patch — empty dict means nothing was volunteered.
    Uses simple substring matching (intentionally no LLM call — this is a
    lightweight optimistic scan, not a classification step).
    """
    t = (text or "").lower()
    patch: dict = {}

    _NO_MEDS = [
        "no med", "no medication", "not on any med", "not taking any",
        "no prescription", "no pills", "no current med", "not taking anything",
        "no meds", "don't take any", "dont take any",
    ]
    _NO_PMH = [
        "no past", "no prior condition", "no previous condition", "no history",
        "no medical history", "no surgeries", "no surgery", "no conditions",
        "no chronic", "otherwise healthy", "healthy otherwise", "no significant",
        "nothing significant", "no previous",
    ]
    _NO_RESULTS = [
        "no recent test", "no test", "no lab", "no labs", "no imaging",
        "no bloodwork", "no scan", "no x-ray", "no xray", "no mri",
        "no recent", "haven't had any test", "haven't had any lab",
        "no recent imaging", "haven't had", "no results",
    ]

    if any(s in t for s in _NO_MEDS):
        patch["medications"] = []
    if any(s in t for s in _NO_PMH):
        patch["pmh"] = []
    if any(s in t for s in _NO_RESULTS):
        patch["recent_results"] = []
    return patch


def clinical_history_node(state: IntakeState):
    user = last_user(state).strip()
    step = state.get("clinical_step") or "allergies"
    thread_id = state.get("thread_id", "")
    # Feature 2: adapt questions based on intake classification
    cls = state.get("intake_classification") or "routine_checkup"

    # Go-back intent — patient wants to correct identity or symptoms from clinical phase
    if user:
        correction = _try_correction(user, state)
        if correction and correction.get("current_phase") not in ("clinical_history", None):
            return correction

    # Lookahead: if the patient volunteers negative info about upcoming steps,
    # pre-fill those fields now so we can skip the questions later.
    volunteered = _prescan_volunteered_clinical(user) if user else {}

    prior = state.get("prior_summary") or {}
    # Skip allergy collection entirely if we have a recent summary and
    # the patient has already confirmed no changes to history.
    # For the simple version: pre-populate from memory and move on.
    if step == "allergies" and prior and prior.get("visit_count", 0) >= 1:
        known_allergies = prior.get("allergies") or []
        if known_allergies:
            return {
                "messages": [{"role": "assistant", "text":
                    f"I see we have {', '.join(known_allergies)} on file for your allergies. "
                    "Anything new to add, or is this still accurate? (reply 'same' or list new ones)"}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
            
    if step == "allergies":
        allergy_q = (
            "Just a few quick questions about your health background — "
            "this helps your care team prepare. "
            "Do you have any allergies we should know about, like to medications, latex, or foods? "
            "If you don't have any, that's completely fine."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": allergy_q}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        # "yes" means the patient wants to list allergies but didn't name them yet
        if is_yes(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "Of course — what are you allergic to? "
                    "Please list them (for example, 'penicillin, latex, peanuts')."}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        allergies = extract_allergies_simple(user)

        # Apply any volunteered skips from the same message
        state_patch: dict = {"allergies": allergies, **volunteered}

        # Determine next step — skip steps already answered by lookahead
        if "medications" not in state_patch:
            next_step = "meds"
            next_q = adapt_clinical_question("meds", cls) or (
                "Are you currently taking any medications — prescription, over-the-counter, "
                "vitamins, or supplements? If you're not on anything at the moment, just let me know."
            )
        elif "pmh" not in state_patch:
            next_step = "pmh"
            next_q = adapt_clinical_question("pmh", cls) or (
                "Almost there. Have you had any significant health conditions in the past, "
                "or any surgeries? If nothing comes to mind, that's perfectly fine."
            )
        elif "recent_results" not in state_patch:
            next_step = "results"
            next_q = adapt_clinical_question("results", cls) or (
                "Last one — have you had any recent tests done, like blood work, X-rays, or scans? "
                "If not, we're all set."
            )
        else:
            # All fields volunteered — skip straight to confirm
            results = state_patch.get("recent_results", [])
            summary = _confirm_summary({**state, **state_patch})
            return {
                **state_patch,
                "clinical_complete": True,
                "clinical_step": "done",
                "validation_target_phase": "confirm",
                "validation_errors": [],
                "current_phase": "validate",
                "messages": [{"role": "assistant", "text":
                    summary + "\n\nDoes everything look right? Reply 'confirm' and I'll prepare "
                    "the note for your care team, or let me know what needs changing."}],
            }

        return {
            **state_patch,
            "clinical_step": next_step,
            "messages": [{"role": "assistant", "text": next_q}],
            "current_phase": "clinical_history",
        }

    if step == "meds":
        meds_q = adapt_clinical_question("meds", cls) or (
            "Are you currently taking any medications — prescription, over-the-counter, "
            "vitamins, or supplements? If you're not on anything at the moment, just let me know."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": meds_q}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }

        from .extract import _is_none_response
        if _is_none_response(user):
            state_patch = {"medications": [], **volunteered}
        else:
            # Pass any already-collected partial meds so the LLM can merge new details
            existing_meds = state.get("medications") or []
            prior_ctx = format_for_prompt(state.get("prior_summary") or {})
            prompt_parts = []
            if prior_ctx:
                prompt_parts.append(prior_ctx)
            prompt_parts.append(f"CURRENT_MEDICATIONS={existing_meds}")
            prompt_parts.append(f"NEW_USER_MESSAGE={user}")
            meds_prompt = "\n\n".join(prompt_parts)
            obj, meta = run_json_step(
                system=meds_extract_system(RESPONSE_RULES),
                prompt=meds_prompt,
                schema=MedsOut,
                fallback={"medications": [], "reply": "Could you tell me the names of the medications you take?"},
                temperature=0.2,
            )
            med_version, med_exp_id = db.resolve_prompt_variant(
                thread_id, "medications", PROMPT_VERSIONS.get("medications", "")
            )
            log_event("llm_step", thread_id=thread_id, node="medications",
                      prompt_version=med_version, experiment_id=med_exp_id, **meta)
            _track_llm_failure(thread_id, "medications", meta)

            out    = obj.model_dump()
            parsed = out.get("medications") or []
            reply  = _safe_reply((out.get("reply") or "").strip())

            if not parsed:
                return {
                    "messages": [{"role": "assistant", "text":
                        reply or "Could you tell me the names of the medications you take?"}],
                    "current_phase": "clinical_history",
                    "clinical_step": "meds",
                    **_failure_state_patch(meta, "clinical_history"),
                }

            # Only ask a follow-up if a medication has name but BOTH dose AND freq are absent
            name_only = [
                m for m in parsed
                if (m.get("name") or "").strip()
                and not (m.get("dose") or "").strip()
                and not (m.get("freq") or "").strip()
            ]
            if name_only:
                names = ", ".join(m["name"] for m in name_only)
                # First follow-up: brief prompt
                already_asked_once = bool(
                    existing_meds and any(
                        not (m.get("dose") or "").strip() and not (m.get("freq") or "").strip()
                        for m in existing_meds
                        if (m.get("name") or "").strip()
                    )
                )
                if already_asked_once:
                    # Second attempt — patient gave vague info ("twice a day" without a dose).
                    # Be explicit: show a concrete example so they know exactly what format works.
                    follow_up = (
                        f"I want to make sure I capture this correctly for your care team. "
                        f"For {names}, could you tell me:\n"
                        f"  • The dose (e.g. 500mg, 10mg)\n"
                        f"  • How often (e.g. once a day, twice daily, every morning)\n"
                        f"For example: '{names.split(',')[0].strip()} 500mg, twice a day'. "
                        f"If you don't know the exact dose, that's okay — just say what you can."
                    )
                else:
                    follow_up = (
                        f"Thanks for sharing that. Just a little more detail if you have it — "
                        f"what dose and how often do you take {names}?"
                    )
                return {
                    "medications": parsed,
                    "clinical_step": "meds",
                    "messages": [{"role": "assistant", "text": follow_up}],
                    "current_phase": "clinical_history",
                    **_failure_state_patch(meta, "clinical_history"),
                }

            state_patch = {"medications": parsed, **volunteered, **_failure_state_patch(meta, "clinical_history"), **_cost_patch(state, meta)}

        # Determine next step
        if "pmh" not in state_patch:
            next_step = "pmh"
            next_q = adapt_clinical_question("pmh", cls) or (
                "Almost there. Have you had any significant health conditions in the past, "
                "or any surgeries? If nothing comes to mind, that's perfectly fine."
            )
        elif "recent_results" not in state_patch:
            next_step = "results"
            next_q = adapt_clinical_question("results", cls) or (
                "Last one — have you had any recent tests done, like blood work, X-rays, or scans? "
                "If not, we're all set."
            )
        else:
            summary = _confirm_summary({**state, **state_patch})
            return {
                **state_patch,
                "clinical_complete": True,
                "clinical_step": "done",
                "validation_target_phase": "confirm",
                "validation_errors": [],
                "current_phase": "validate",
                "messages": [{"role": "assistant", "text":
                    summary + "\n\nDoes everything look right? Reply 'confirm' and I'll prepare "
                    "the note for your care team, or let me know what needs changing."}],
            }

        return {
            **state_patch,
            "clinical_step": next_step,
            "messages": [{"role": "assistant", "text": next_q}],
            "current_phase": "clinical_history",
        }

    if step == "pmh":
        pmh_q = adapt_clinical_question("pmh", cls) or (
            "Almost there. Have you had any significant health conditions in the past, "
            "or any surgeries? If nothing comes to mind, that's perfectly fine."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": pmh_q}],
                "current_phase": "clinical_history",
                "clinical_step": "pmh",
            }
        pmh = extract_list_simple(user)
        state_patch = {"pmh": pmh, **volunteered}

        if "recent_results" not in state_patch:
            next_step = "results"
            next_q = adapt_clinical_question("results", cls) or (
                "Last one — have you had any recent tests done, like blood work, X-rays, or scans? "
                "If not, we're all set."
            )
            return {
                **state_patch,
                "clinical_step": next_step,
                "messages": [{"role": "assistant", "text": next_q}],
                "current_phase": "clinical_history",
            }

        # Results were volunteered — skip to confirm
        summary = _confirm_summary({**state, **state_patch})
        return {
            **state_patch,
            "clinical_complete": True,
            "clinical_step": "done",
            "validation_target_phase": "confirm",
            "validation_errors": [],
            "current_phase": "validate",
            "messages": [{"role": "assistant", "text":
                summary + "\n\nDoes everything look right? Reply 'confirm' and I'll prepare "
                "the note for your care team, or let me know what needs changing."}],
        }

    if step == "results":
        results_q = adapt_clinical_question("results", cls) or (
            "Last one — have you had any recent tests done, like blood work, X-rays, or scans? "
            "If not, we're all set."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": results_q}],
                "current_phase": "clinical_history",
                "clinical_step": "results",
            }
        results = extract_list_simple(user)
        summary = _confirm_summary({**state, "recent_results": results})
        # Feature 4: route through validate_node before confirm
        return {
            "recent_results": results,
            "clinical_complete": True,
            "clinical_step": "done",
            "validation_target_phase": "confirm",
            "validation_errors": [],
            "current_phase": "validate",
            "messages": [{"role": "assistant", "text":
                summary + "\n\nDoes everything look right? Reply 'confirm' and I'll prepare "
                "the note for your care team, or let me know what needs changing."}],
        }

    return {"current_phase": "report"}


# ---------------------------------------------------------------------------
# Validate node  (Feature 4 — agentic validation gate)
# ---------------------------------------------------------------------------

# Values that an LLM might write into OPQRST fields that are not clinically
# useful. Treated as empty for validation purposes so ambiguous extraction
# cannot silently advance the state machine.
_OPQRST_PLACEHOLDERS = frozenset({
    "none", "n/a", "na", "unknown", "not provided", "not sure",
    "unsure", "unclear", "unspecified", "not applicable", "not known",
    "not given",
})


def _is_substantive(val: str) -> bool:
    """Return True only when val is non-empty and not a known placeholder."""
    v = (val or "").strip().lower()
    return bool(v) and v not in _OPQRST_PLACEHOLDERS


def validate_node(state: IntakeState):
    """
    Non-interactive routing node that enforces completeness before a phase transition.

    Runs synchronously within the same graph.invoke() call — NOT in interrupt_after.
    On success → routes to validation_target_phase.
    On failure → routes back to the source phase with a targeted gap-fill message.

    Guard details for target == "clinical_history":
      - chief_complaint must be non-empty
      - at least 2 of {onset, severity, quality} must be substantive
        (non-empty AND not a placeholder like "unknown" / "n/a")
      - ED mode additionally requires severity to be substantive
    """
    target = state.get("validation_target_phase") or "clinical_history"
    mode   = state.get("mode") or "clinic"
    errors: list[str] = []

    if target == "clinical_history":
        cc = (state.get("chief_complaint") or "").strip()
        op = state.get("opqrst") or {}
        if not cc:
            errors.append("chief_complaint")
        key_filled = sum(
            1 for f in ["onset", "severity", "quality"]
            if _is_substantive(op.get(f) or "")
        )
        if key_filled < 2:
            errors.append("opqrst_incomplete")
        # ED always requires a substantive severity, even if two other key fields pass
        if mode == "ed" and not _is_substantive(op.get("severity") or ""):
            if "opqrst_incomplete" not in errors:
                errors.append("severity_required")

        # If the LLM flagged its own extraction as low-confidence and the session
        # has already been through at least one retry, ask the patient to clarify.
        # This is NOT a hard block — it routes back to subjective for one more turn.
        conf = state.get("extraction_confidence") or "medium"
        retry_count = int(state.get("extraction_retry_count") or 0)
        if conf == "low" and retry_count >= 1 and not errors:
            # Only trigger if no other errors already caught — avoids sending
            # the patient two different error messages in the same turn.
            errors.append("extraction_confidence_low")

    elif target == "confirm":
        if state.get("allergies") is None:
            errors.append("allergies_not_collected")
        if not state.get("clinical_complete"):
            errors.append("clinical_history_incomplete")

    if not errors:
        log_event("validation_passed", thread_id=state.get("thread_id", ""),
                  target_phase=target)
        return {
            "validation_errors": [],
            "current_phase": target,
        }

    # Validation failed — route back to source phase with gap message
    back_phase = "subjective" if target == "clinical_history" else "clinical_history"
    cc = (state.get("chief_complaint") or "").strip()
    gap_msg = build_validation_gap_message(errors, cc, mode)
    log_event("validation_failed", level="warning",
              thread_id=state.get("thread_id", ""),
              target_phase=target, errors=errors)
    return {
        "validation_errors": errors,
        "current_phase": back_phase,
        "subjective_complete": False if back_phase == "subjective" else state.get("subjective_complete"),
        "messages": [{"role": "assistant", "text": gap_msg}],
    }


# ---------------------------------------------------------------------------
# Confirm node
# ---------------------------------------------------------------------------

def confirm_node(state: IntakeState):
    user = last_user(state).strip()

    intent = _classify_intent(user, state)

    if intent.intent == "confirm":
        return {
            "current_phase": "report",
            "messages": [{"role": "assistant", "text": "Got it — generating the clinician note now."}],
        }

    # Use the shared correction router — handles "change my X", "go back", etc.
    # Also covers explicit correction intents from _classify_intent.
    correction = _try_correction(user, state)
    if correction:
        return correction

    return {
        "current_phase": "confirm",
        "messages": [{"role": "assistant", "text":
            "Reply 'confirm' to proceed, or tell me what to change — "
            "your contact details, symptoms, or health history."}],
    }


# ---------------------------------------------------------------------------
# Report node
# ---------------------------------------------------------------------------

def _fmt_report_text(cc: str, op: dict, allergies: list, meds: list,
                     pmh: list, results: list, identity: dict) -> str:
    """Build a safe structured report without LLM — used as primary fallback."""
    allergies_line = "NKDA" if not allergies else ", ".join(allergies)
    return (
        f"SUBJECTIVE INTAKE\n"
        f"Chief Complaint (CC): {cc}\n\n"
        f"History of Present Illness (HPI):\n"
        f"  Onset:       {op.get('onset') or 'Unknown/Not provided'}\n"
        f"  Provocation: {op.get('provocation') or 'Unknown/Not provided'}\n"
        f"  Quality:     {op.get('quality') or 'Unknown/Not provided'}\n"
        f"  Radiation:   {op.get('radiation') or 'Unknown/Not provided'}\n"
        f"  Severity:    {op.get('severity') or 'Unknown/Not provided'}\n"
        f"  Timing:      {op.get('timing') or 'Unknown/Not provided'}\n\n"
        f"CLINICAL HISTORY & SAFETY\n"
        f"*** ALLERGIES (IMPORTANT): {allergies_line} ***\n"
        f"Current Medications:\n{_fmt_meds_fallback(meds)}\n"
        f"Past Medical History: {', '.join(pmh) if pmh else 'None reported'}\n"
        f"Recent Lab/Imaging: {', '.join(results) if results else 'None reported'}\n\n"
        f"PATIENT IDENTITY\n"
        f"  Name:    {identity.get('name') or 'Unknown'}\n"
        f"  DOB:     {identity.get('dob') or 'Unknown'}\n"
        f"  Phone:   {identity.get('phone') or 'Unknown'}\n"
        f"  Address: {identity.get('address') or 'Unknown'}\n"
    )


def _fmt_meds_fallback(meds: list) -> str:
    if not meds:
        return "- None/Unknown"
    lines = []
    for m in meds:
        name = (m.get("name") or "").strip() or "Unknown"
        line = f"- {name}"
        if m.get("dose"):       line += f", {m['dose']}"
        if m.get("freq"):       line += f", {m['freq']}"
        if m.get("last_taken"): line += f" (last taken: {m['last_taken']})"
        lines.append(line)
    return "\n".join(lines)


_REPORT_REQUIRED_SECTIONS = [
    "SUBJECTIVE INTAKE",
    "CLINICAL HISTORY",
    "PATIENT IDENTITY",
]


def _validate_report_content(text: str) -> List[str]:
    """
    Inspect a generated report for structural and safety issues.

    Returns a list of warning codes (empty = all clear).  Called before
    db.save_report() so problems are logged before the artifact is persisted.

    Codes:
      report_too_short           — less than 150 characters (likely empty/truncated)
      report_too_long            — exceeds 6000 characters
      missing_section_<name>     — required section header absent
      diagnosis_language         — LLM drifted into diagnosis language
    """
    warnings: List[str] = []
    stripped = (text or "").strip()

    if len(stripped) < 150:
        warnings.append("report_too_short")
    if len(stripped) > 6000:
        warnings.append("report_too_long")

    for section in _REPORT_REQUIRED_SECTIONS:
        if section not in stripped:
            warnings.append(f"missing_section_{section.lower().replace(' ', '_')}")

    _, modified = validate_llm_response(stripped)
    if modified:
        warnings.append("diagnosis_language")

    return warnings


def report_node(state: IntakeState):
    thread_id = state.get("thread_id", "")

    # --- Safety preflight: block report if required fields are missing ---
    preflight = SafetyChecker.compute(state)
    if not preflight.ok:
        log_event("report_blocked_preflight", level="warning",
                  thread_id=thread_id,
                  blocking_reasons=preflight.blocking_reasons,
                  safety_score=preflight.safety_score)
        db.create_escalation(
            thread_id=thread_id,
            kind="report_blocked",
            payload=build_reason_trail(
                "report_blocked", state,
                override_reasons=preflight.blocking_reasons,
            ),
        )
        db.set_session_status(thread_id, "escalated")
        return {
            "human_review_required": True,
            "human_review_reasons":  preflight.blocking_reasons,
            "safety_score":          preflight.safety_score,
            "messages": [{"role": "assistant", "text":
                "Some required information is missing from your intake. "
                "A clinician has been notified and will follow up with you directly."}],
            "current_phase": "handoff",
        }

    # Build and validate the state snapshot — both the clinician note and the
    # FHIR bundle are generated from this validated model, never from raw state.
    validated = _build_validated_report_state(state)
    identity  = validated.identity.model_dump()
    cc        = validated.chief_complaint or "Unknown/Not provided"
    op        = validated.opqrst.model_dump()
    allergies = validated.allergies
    meds      = [m.model_dump() for m in validated.medications]
    pmh       = validated.pmh
    results   = validated.recent_results

    if settings().intake.use_llm_report_narrative:
        payload = {
            "identity": identity, "chief_complaint": cc,
            "opqrst": op, "allergies": allergies,
            "medications": meds, "pmh": pmh,
            "recent_results": results, "triage": validated.triage,
        }
        try:
            res = get_gemini().generate_text(
                system=report_system(),
                prompt=json.dumps(payload, indent=2),
                temperature=0.2,
                max_tokens=1300,
                response_mime_type="text/plain",
            )
            if res.input_tokens or res.output_tokens:
                db.record_llm_usage(thread_id=thread_id, node="report_node",
                                    input_tokens=res.input_tokens, output_tokens=res.output_tokens)
            if res.ok and res.text.strip():
                report_text = res.text.strip()
            else:
                log_event("report_llm_failed", level="warning",
                          thread_id=thread_id, error=res.error)
                report_text = _fmt_report_text(cc, op, allergies, meds, pmh, results, identity)
        except Exception as _report_exc:
            log_event("report_llm_error", level="warning",
                      thread_id=thread_id, error=str(_report_exc)[:200])
            report_text = _fmt_report_text(cc, op, allergies, meds, pmh, results, identity)
    else:
        log_event("report_template_used", thread_id=thread_id)
        report_text = _fmt_report_text(cc, op, allergies, meds, pmh, results, identity)

    content_warnings = _validate_report_content(report_text)
    if content_warnings:
        log_event("report_content_warnings", level="warning",
                  thread_id=thread_id, warnings=content_warnings)
        if "diagnosis_language" in content_warnings:
            log_event("report_diagnosis_language_discarded", level="warning",
                      thread_id=thread_id)
            report_text = _fmt_report_text(cc, op, allergies, meds, pmh, results, identity)

    # FHIR bundle — validate input first, then build from validated state.
    # Failure never blocks the clinician note.
    fhir_json: str | None = None
    fhir_warnings = fhir_builder.validate_fhir_input(validated.model_dump())
    if fhir_warnings:
        log_event("fhir_input_warnings", level="warning",
                  thread_id=thread_id, warnings=fhir_warnings)
    try:
        bundle    = fhir_builder.build_bundle(validated.model_dump())
        # Structural validation — catches malformed bundles before EHR delivery.
        struct_errors = fhir_builder.validate_fhir_bundle(bundle)
        if struct_errors:
            log_event("fhir_bundle_validation_errors", level="warning",
                      thread_id=thread_id, errors=struct_errors)
        fhir_json = json.dumps(bundle, indent=2)
        log_event("fhir_bundle_built", thread_id=thread_id,
                  resource_count=len(bundle.get("entry", [])),
                  input_warnings=fhir_warnings or None,
                  struct_errors=struct_errors or None)
    except Exception as e:
        log_event("fhir_bundle_error", level="warning",
                  thread_id=thread_id, error=str(e)[:200])

    triage = validated.triage
    db.save_report(
        thread_id,
        triage.get("risk_level") or "low",
        triage.get("visit_type") or "routine",
        report_text,
        fhir_json,
        pending_review=preflight.review_required,
    )

    # Layer-2 memory upsert: merge this completed visit into the patient summary
    # so the next visit starts with known context rather than a blank slate.
    patient_id = state.get("patient_id")
    if patient_id:
        try:
            prior = db.get_patient_summary(patient_id)
            visit_payload = validated.model_dump()
            visit_payload["crisis_detected"] = bool(state.get("crisis_detected"))
            merged = merge_summary(prior, visit_payload)
            db.upsert_patient_summary(patient_id, merged)
            log_event("patient_summary_upserted", thread_id=thread_id,
                      patient_id=patient_id, visit_count=merged["visit_count"])
        except Exception as e:
            # Never let memory persistence fail the report finalization.
            log_event("patient_summary_upsert_error", level="warning",
                      thread_id=thread_id, error=str(e)[:200])

    db.set_session_status(thread_id, "done")


    log_event(
        "phi_audit_session_complete",
        thread_id=thread_id,
        identity_status=state.get("identity_status") or "unverified",
        mode=state.get("mode") or "clinic",
    )

    # Push to direct FHIR server (best-effort, non-blocking).
    if fhir_json and settings().fhir_server_url:
        import threading
        threading.Thread(
            target=_fhir_push_bundle,
            kwargs={"fhir_bundle_json": fhir_json, "thread_id": thread_id},
            daemon=True,
        ).start()

    patient_name = (identity or {}).get("name") or "unknown patient"
    webhook.dispatch_intake_complete(
        thread_id=thread_id,
        patient_name=patient_name,
        risk_level=triage.get("risk_level") or "low",
        fhir_json=fhir_json,
    )

    return {
        "human_review_required": preflight.review_required,
        "human_review_reasons":  preflight.review_reasons,
        "safety_score":          preflight.safety_score,
        "messages": [{"role": "assistant", "text":
            f"Your intake is complete. Here is the clinician note prepared for your visit:\n\n"
            f"---\n{report_text}\n---\n\n"
            "A copy has been sent to your care team. If anything looks incorrect, "
            "please let the front desk know when you arrive."}],
        "current_phase": "done",
    }


# ---------------------------------------------------------------------------
# Handoff node
# ---------------------------------------------------------------------------

def handoff_node(state: IntakeState):
    return {
        "messages": [{"role": "assistant", "text":
            "To prioritize your safety, please call 911 or go to the nearest emergency room now. "
            "A clinician has been notified."}],
        "current_phase": "handoff",
    }