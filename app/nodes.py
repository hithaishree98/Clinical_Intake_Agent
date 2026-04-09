"""
nodes.py — LangGraph conversation nodes.

Each node corresponds to one phase of the intake state machine.
All nodes return a dict of state fields to update (LangGraph merges them).
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .prompts import subjective_extract_system, meds_extract_system, report_system, PROMPT_VERSIONS
from .state import IntakeState
from .schemas import SubjectiveOut, MedsOut, ReportInputState
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
    extract_identity_deterministic,
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
    is_consent_accepted,
    is_consent_declined,
    validate_phone,
    validate_dob,
)
from .settings import get_settings as settings

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



def _check_crisis(user: str, state: IntakeState) -> str | None:
    """
    Two-tier crisis detection.

    Tier 1 — keyword/regex (detect_crisis): explicit self-harm phrases.
              Zero latency.  High precision.  False-positive risk: low.

    Tier 2 — LLM classifier (llm_crisis_score): borderline cases that Tier 1
              misses (passive ideation, burden language, hopelessness) and
              figurative phrases that Tier 1 would wrongly flag ("kill this
              headache").  Only invoked when has_soft_distress() gate fires.
              Confidence high/medium → escalate; low → soft-log only.

    Returns CRISIS_RESOURCE (the 988 message) if either tier fires.
    Callers must set crisis_detected=True when this returns a non-None value.
    """
    thread_id    = state.get("thread_id", "")
    patient_name = (state.get("identity") or {}).get("name") or "unknown patient"

    # ── Tier 1: keyword / regex ────────────────────────────────────────
    crisis_flags = detect_crisis(user)
    if crisis_flags:
        log_event("crisis_detected_tier1", level="warning",
                  thread_id=thread_id, matched_phrases=crisis_flags)
        db.create_escalation(
            thread_id=thread_id,
            kind="crisis",
            payload=build_reason_trail(
                "crisis", state, extra_data={"matched_phrases": crisis_flags,
                                             "detection_tier": "keyword"}
            ),
        )
        webhook.dispatch_crisis_alert(
            thread_id=thread_id,
            patient_name=patient_name,
            matched_phrases=crisis_flags,
        )
        return CRISIS_RESOURCE

    # ── Tier 2: LLM classifier for borderline distress signals ─────────
    if not has_soft_distress(user):
        return None   # no signals at all — skip LLM call

    score = llm_crisis_score(user)

    if score.is_crisis_risk and score.confidence in ("high", "medium"):
        log_event("crisis_detected_tier2", level="warning",
                  thread_id=thread_id,
                  confidence=score.confidence,
                  reasoning=score.reasoning)
        db.create_escalation(
            thread_id=thread_id,
            kind="crisis",
            payload=build_reason_trail(
                "crisis", state,
                extra_data={
                    "matched_phrases":  [f"llm:{score.reasoning}"],
                    "detection_tier":   "llm",
                    "llm_confidence":   score.confidence,
                    "llm_reasoning":    score.reasoning,
                },
            ),
        )
        webhook.dispatch_crisis_alert(
            thread_id=thread_id,
            patient_name=patient_name,
            matched_phrases=[f"llm_detected ({score.confidence}): {score.reasoning}"],
        )
        return CRISIS_RESOURCE

    if score.is_crisis_risk and score.confidence == "low":
        # Soft flag: not confident enough to escalate, but worth logging for
        # clinical audit.  The intake continues; a supervisor can review.
        log_event("soft_distress_flagged", level="info",
                  thread_id=thread_id,
                  reasoning=score.reasoning)

    return None


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

    if is_consent_accepted(user):
        log_event("patient_consented", thread_id=state.get("thread_id"))
        return {
            "consent_given": True,
            "messages": [{"role": "assistant", "text": "Thank you. What's your full name?"}],
            "current_phase": "identity",
        }

    if is_consent_declined(user):
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
    user     = last_user(state).strip()
    identity = dict(state.get("identity") or {"name": "", "dob": "", "phone": "", "address": ""})
    attempts = int(state.get("identity_attempts") or 0)

    if attempts == 0 and not user:
        return {
            "messages": [{"role": "assistant", "text": "What's your full name?"}],
            "current_phase": "identity",
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": False,
        }

    det = extract_identity_deterministic(user)

    if det.get("name") and not (identity.get("name") or "").strip():
        identity["name"] = det["name"].strip()

    # validate_dob rejects future dates and impossible ages — original had no date check
    if det.get("dob") and not (identity.get("dob") or "").strip():
        dob_clean, dob_err = validate_dob(det["dob"])
        if dob_err:
            return {
                "identity": identity,
                "identity_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text": f"{dob_err} What is your date of birth? (MM/DD/YYYY)"}],
                "current_phase": "identity",
            }
        identity["dob"] = dob_clean

    # validate_phone ensures 10 digits — original just stripped non-digits from anything
    if det.get("phone") and not (identity.get("phone") or "").strip():
        phone_clean, phone_err = validate_phone(det["phone"])
        if phone_err:
            return {
                "identity": identity,
                "identity_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text": f"{phone_err} What's the best phone number to reach you?"}],
                "current_phase": "identity",
            }
        identity["phone"] = phone_clean

    if det.get("address") and not (identity.get("address") or "").strip():
        identity["address"] = det["address"].strip()
    
    attempts += 1
    missing = [k for k in ["name", "dob", "phone", "address"] if not (identity.get(k) or "").strip()]

    if missing:
        if attempts >= 6:
            log_event("identity_max_attempts", level="warning",
                      thread_id=state.get("thread_id", ""))
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
            "dob":     "What's your date of birth? (MM/DD/YYYY)",
            "phone":   "What's the best phone number to reach you?",
            "address": "What's your current home address?",
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
        return {
            "identity": identity,
            "stored_identity": stored,
            "messages": [{"role": "assistant", "text":
                "I found information on file:\n"
                f"- {_summary_identity(stored)}\n\n"
                "You provided:\n"
                f"- {_summary_identity(identity)}\n\n"
                "Should I keep the stored info, or update it with what you provided? (keep / update)"
            }],
            "current_phase": "identity_review",
        }

    return {
        "identity": identity,
        "stored_identity": None,
        "identity_status": "unverified",
        "needs_identity_review": True,
        "messages": [{"role": "assistant", "text":
            f"Got it. I have: {_summary_identity(identity)}. Is this correct? (yes / no)"}],
        "current_phase": "identity_review",
    }


# ---------------------------------------------------------------------------
# Identity review node
# ---------------------------------------------------------------------------

def identity_review_node(state: IntakeState):
    user   = last_user(state).strip().lower()
    identity = dict(state.get("identity") or {})
    stored   = state.get("stored_identity")
    thread_id = state.get("thread_id", "")

    if stored:
        if user.startswith("keep"):
            return {
                "identity": stored,
                "identity_status": "verified",
                "needs_identity_review": False,
                "messages": [{"role": "assistant", "text": "Thanks — I'll keep what's on file. What brings you in today?"}],
                "current_phase": "subjective",
            }
        if user.startswith("update"):
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
            "messages": [{"role": "assistant", "text": "Please reply 'keep' or 'update'."}],
            "current_phase": "identity_review",
        }

    if is_yes(user):
        return {
            "identity_status": "verified",
            "needs_identity_review": False,
            "messages": [{"role": "assistant", "text":
                "Thanks. What's the main reason for your visit today? (in your own words)"}],
            "current_phase": "subjective",
        }
    if is_no(user):
        return {
            "identity": {"name": "", "dob": "", "phone": "", "address": ""},
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": True,
            "messages": [{"role": "assistant", "text": "No problem — let's start over. What's your full name?"}],
            "current_phase": "identity",
        }
    return {
        "messages": [{"role": "assistant", "text": "Just to confirm — is the information I have correct? (yes / no)"}],
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


    # Crisis detection (before emergency to give appropriate response)
    crisis_reply = _check_crisis(user, state)
    if crisis_reply:
        return {
            "crisis_detected": True,   # persisted so SafetyChecker can flag for human review
            "human_review_required": True,
            "messages": [{"role": "assistant", "text": crisis_reply}],
            "current_phase": "subjective",   # don't terminate — patient may still want to complete
        }

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
    prompt = (
        f"CURRENT_STATE={json.dumps({'chief_complaint': cc, 'opqrst': op})}\n"
        f"NEW_USER_MESSAGE={user}"
    )
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
    }


# ---------------------------------------------------------------------------
# Confirm summary helper
# ---------------------------------------------------------------------------

def _confirm_summary(state: IntakeState) -> str:
    identity = state.get("identity") or {}
    cc       = state.get("chief_complaint") or "—"
    op       = state.get("opqrst") or {}
    allergies = state.get("allergies") or []
    meds      = state.get("medications") or []
    pmh       = state.get("pmh") or []
    results   = state.get("recent_results") or []
    triage    = state.get("triage") or {}

    def fmt_list(xs): return "None" if not xs else ", ".join(xs)
    def fmt_meds(ms):
        if not ms: return "None"
        parts = []
        for m in ms:
            s = (m.get("name") or "Unknown").strip()
            if m.get("dose"): s += f" {m['dose']}"
            if m.get("freq"): s += f" ({m['freq']})"
            if m.get("last_taken"): s += f", last: {m['last_taken']}"
            parts.append(s)
        return "; ".join(parts)

    lines = [
        "Here's what I captured:", "",
        "Identity",
        f"- Name: {identity.get('name') or '—'}",
        f"- DOB: {identity.get('dob') or '—'}",
        f"- Phone: {identity.get('phone') or '—'}",
        f"- Address: {identity.get('address') or '—'}",
        "",
        "Symptoms",
        f"- Chief complaint: {cc}",
        f"- Onset: {op.get('onset') or '—'}",
        f"- Provocation: {op.get('provocation') or '—'}",
        f"- Quality: {op.get('quality') or '—'}",
        f"- Radiation: {op.get('radiation') or '—'}",
        f"- Severity: {op.get('severity') or '—'}",
        f"- Timing: {op.get('timing') or '—'}",
        "",
        "History",
        f"- Allergies: {fmt_list(allergies)}",
        f"- Medications: {fmt_meds(meds)}",
        f"- PMH: {fmt_list(pmh)}",
        f"- Recent results: {fmt_list(results)}",
    ]
    if triage:
        lines += ["", "Triage",
                  f"- Risk: {triage.get('risk_level') or '—'}",
                  f"- Visit type: {triage.get('visit_type') or '—'}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clinical history node
# ---------------------------------------------------------------------------

def clinical_history_node(state: IntakeState):
    user = last_user(state).strip()
    step = state.get("clinical_step") or "allergies"
    thread_id = state.get("thread_id", "")
    # Feature 2: adapt questions based on intake classification
    cls = state.get("intake_classification") or "routine_checkup"

    # Crisis detection applies in every phase — a patient may disclose suicidal
    # ideation while answering "any past medical conditions?" rather than the
    # chief-complaint question.
    if user:
        crisis_msg = _check_crisis(user, state)
        if crisis_msg:
            return {
                "crisis_detected": True,
                "messages": [{"role": "assistant", "text": crisis_msg}],
                "current_phase": "handoff",
            }

    if step == "allergies":
        allergy_q = (
            "I'd like to ask a few quick questions about your health history. "
            "First — do you have any known allergies, especially to medications or latex? "
            "If none, just say 'none'."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": allergy_q}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        # "yes" means the patient wants to list allergies but didn't yet — ask them to name them
        if is_yes(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "What are you allergic to? Please list them (e.g. 'penicillin, latex')."}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        allergies = extract_allergies_simple(user)
        meds_q = adapt_clinical_question("meds", cls) or (
            "Got it, thank you. What medications are you currently taking? "
            "Please include the name, dose, how often you take it, and when you last took it. "
            "If you're not taking anything, just say 'none'."
        )
        return {
            "allergies": allergies,
            "clinical_step": "meds",
            "messages": [{"role": "assistant", "text": meds_q}],
            "current_phase": "clinical_history",
        }

    if step == "meds":
        meds_q = adapt_clinical_question("meds", cls) or (
            "What medications are you currently taking? "
            "Please include the name, dose, how often you take it, and when you last took it. "
            "If you're not taking anything, just say 'none'."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": meds_q}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }
        pmh_q = adapt_clinical_question("pmh", cls) or (
            "Thank you. Do you have any past medical conditions or surgeries I should know about? "
            "If none, just say 'none'."
        )
        from .extract import _is_none_response
        if _is_none_response(user) or user.strip().lower() in {"no meds", "not taking anything"}:
            return {
                "medications": [],
                "clinical_step": "pmh",
                "messages": [{"role": "assistant", "text": pmh_q}],
                "current_phase": "clinical_history",
            }

        # Pass any already-collected partial meds so the LLM can merge new details
        existing_meds = state.get("medications") or []
        obj, meta = run_json_step(
            system=meds_extract_system(RESPONSE_RULES),
            prompt=(
                f"CURRENT_MEDICATIONS={existing_meds}\n"
                f"NEW_USER_MESSAGE={user}"
            ),
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

        # Check for medications that are missing frequency — ask a targeted follow-up
        incomplete = [m for m in parsed if (m.get("name") or "").strip() and not (m.get("freq") or "").strip()]
        if incomplete:
            names = ", ".join(m["name"] for m in incomplete)
            follow_up = (
                f"Thanks for sharing that. To make sure we have the complete picture, "
                f"how often do you take {names}? "
                f"And when did you last take it?"
            )
            return {
                "medications": parsed,          # save partial — merged on next turn
                "clinical_step": "meds",        # stay in meds step
                "messages": [{"role": "assistant", "text": follow_up}],
                "current_phase": "clinical_history",
                **_failure_state_patch(meta, "clinical_history"),
            }

        return {
            "medications": parsed,
            "clinical_step": "pmh",
            "messages": [{"role": "assistant", "text": pmh_q}],
            "current_phase": "clinical_history",
            **_failure_state_patch(meta, "clinical_history"),
        }

    if step == "pmh":
        pmh_q = adapt_clinical_question("pmh", cls) or (
            "Thank you. Do you have any past medical conditions or surgeries I should be aware of? "
            "If none, just say 'none'."
        )
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": pmh_q}],
                "current_phase": "clinical_history",
                "clinical_step": "pmh",
            }
        pmh = extract_list_simple(user)
        results_q = adapt_clinical_question("results", cls) or (
            "Appreciated — almost done. Have you had any recent lab tests or imaging "
            "(bloodwork, X-ray, CT, etc.) since your last visit? If none, just say 'none'."
        )
        return {
            "pmh": pmh,
            "clinical_step": "results",
            "messages": [{"role": "assistant", "text": results_q}],
            "current_phase": "clinical_history",
        }

    if step == "results":
        results_q = adapt_clinical_question("results", cls) or (
            "Have you had any recent lab tests or imaging since your last visit? "
            "If none, just say 'none'."
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
                summary + "\n\nReply 'confirm' to generate the clinician note, "
                "or tell me what you'd like to change."}],
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
    # Crisis detection: patient may disclose self-harm intent at any phase.
    if user:
        crisis_msg = _check_crisis(user, state)
        if crisis_msg:
            return {
                "crisis_detected": True,
                "messages": [{"role": "assistant", "text": crisis_msg}],
                "current_phase": "handoff",
            }

    user = user.lower()
    if user in {"confirm", "yes", "y", "ok", "okay", "looks good", "correct", "done"}:
        return {
            "current_phase": "report",
            "messages": [{"role": "assistant", "text": "Got it — generating the clinician note now."}],
        }

    def _match(patterns: list[str]) -> bool:
        return any(re.search(p, user) for p in patterns)

    if _match([r"\ballerg", r"\bmed(ication|icine|s)?\b", r"\bpmh\b",
               r"\bhistory\b", r"\bsurgery\b", r"\btest\b", r"\blab\b", r"\bimaging\b"]):
        return {
            "current_phase": "clinical_history",
            "clinical_step": "allergies",
            "messages": [{"role": "assistant", "text":
                "Sure — let's update your medical history. "
                "Do you have any allergies?"}],
        }
    if _match([r"\bpain\b", r"\bsymptom\b", r"\bonset\b", r"\bseverity\b",
               r"\btiming\b", r"\bradiati", r"\bquality\b", r"\bprovocation\b",
               r"\bcomplaint\b"]):
        return {
            "current_phase": "subjective",
            "messages": [{"role": "assistant", "text":
                "Sure — what would you like to change about your symptoms?"}],
        }
    if _match([r"\bname\b", r"\bphone\b", r"\baddress\b", r"\bdob\b",
               r"\bdate of birth\b"]):
        return {
            "current_phase": "identity",
            "identity_attempts": 0,
            "messages": [{"role": "assistant", "text":
                "Sure — what should I update in your contact details?"}],
        }
    return {
        "current_phase": "confirm",
        "messages": [{"role": "assistant", "text":
            "Reply 'confirm' to proceed, or tell me what to change "
            "(symptoms, history, or identity details)."}],
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

    payload = {
        "identity": identity, "chief_complaint": cc,
        "opqrst": op, "allergies": allergies,
        "medications": meds, "pmh": pmh,
        "recent_results": results, "triage": validated.triage,
    }

    res = get_gemini().generate_text(
        system=report_system(),
        prompt=json.dumps(payload, indent=2),
        temperature=0.2,
        max_tokens=1300,
        response_mime_type="text/plain",
    )

    if not res.ok or not res.text.strip():
        log_event("report_fallback_used", level="warning",
                  thread_id=thread_id, error=res.error)
        report_text = _fmt_report_text(cc, op, allergies, meds, pmh, results, identity)
    else:
        report_text = res.text.strip()

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
            "Your intake is complete. Click \"View Report\" to see the clinician note."}],
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