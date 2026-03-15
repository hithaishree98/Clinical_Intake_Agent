"""
nodes.py — LangGraph conversation nodes.

Each node corresponds to one phase of the intake state machine.
All nodes return a dict of state fields to update (LangGraph merges them).

Changes from base version:
  - Guardrails integrated: prompt injection check, crisis detection,
    LLM response validation, PHI audit logging
  - New webhook module used for all outbound notifications
  - Validators used for DOB and phone before storing in state
  - Consent phase added (controlled by settings.require_consent)
  - _slack() removed — all notifications go through webhook.dispatch_*
"""
from __future__ import annotations

import json
import re
from typing import Dict

from .prompts import subjective_extract_system, meds_extract_system, report_system
from .state import IntakeState
from .schemas import SubjectiveOut, MedsOut
from .llm import run_json_step, get_gemini, validate_llm_response
from .logging_utils import log_event
from . import sqlite_db as db
from . import fhir_builder
from . import webhook
from .extract import (
    extract_identity_deterministic,
    normalize_phone,
    is_yes,
    is_no,
    is_ack,
    detect_emergency_red_flags,
    extract_allergies_simple,
    extract_list_simple,
    check_prompt_injection,
    detect_crisis,
    CRISIS_RESOURCE,
    CONSENT_MESSAGE,
    is_consent_accepted,
    is_consent_declined,
    validate_phone,
    validate_dob,
)
from .settings import settings

RESPONSE_RULES = (
    "Be concise and human. "
    "Ask ONE question only if needed. "
    "Never invent facts. "
    "No diagnosis. "
    "No medical advice."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def last_user(state: IntakeState) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if m["role"] == "user":
            return m["text"]
    return ""


def _summary_identity(x: Dict[str, str]) -> str:
    return (
        f"Name: {x.get('name') or '—'}, "
        f"DOB: {x.get('dob') or '—'}, "
        f"Phone: {x.get('phone') or '—'}, "
        f"Address: {x.get('address') or '—'}"
    )



def _check_crisis(user: str, state: IntakeState) -> str | None:
    """
    Run crisis detection. Returns crisis resource message if triggered, else None.
    """
    crisis_flags = detect_crisis(user)
    if not crisis_flags:
        return None

    thread_id    = state.get("thread_id", "")
    patient_name = (state.get("identity") or {}).get("name") or "unknown patient"

    log_event("guardrail_crisis_detected", level="warning",
              thread_id=thread_id, matched_phrases=crisis_flags)
    db.create_escalation(
        thread_id=thread_id,
        kind="crisis",
        payload={"matched_phrases": crisis_flags},
    )
    webhook.dispatch_crisis_alert(
        thread_id=thread_id,
        patient_name=patient_name,
        matched_phrases=crisis_flags,
    )
    return CRISIS_RESOURCE


def _safe_reply(text: str) -> str:
    """Run LLM response through the diagnosis-language guardrail."""
    safe, _ = validate_llm_response(text)
    return safe


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
        log_event("patient_declined_consent", thread_id=state.get("thread_id"))
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

    thread_id = state.get("thread_id", "")

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
    user   = last_user(state).strip()
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
                payload={"stored_identity": stored, "new_identity": identity},
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

def needs_ed_followup(cc: str, op: Dict[str, str]) -> bool:
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
        db.create_escalation(thread_id=thread_id, kind="emergency", payload={"triage": triage})
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
            "chief_complaint": cc,
            "opqrst": op,
            "is_complete": False,
            "reply": "When did it start, and how severe is it from 0-10?",
        },
        temperature=0.2,
    )
    log_event("llm_step", thread_id=thread_id, node="subjective", **meta)

    out    = obj.model_dump()
    new_cc = (out.get("chief_complaint") or "").strip()
    new_op = out.get("opqrst") or {}

    if new_cc and not cc:
        cc = new_cc

    for k in op.keys():
        v = (new_op.get(k) or "").strip()
        if v and not (op.get(k) or "").strip():
            op[k] = v

    complete = bool(out.get("is_complete"))
    reply    = _safe_reply((out.get("reply") or "").strip())

    if complete:
        mode     = state.get("mode") or "clinic"
        attempts = int(state.get("triage_attempts") or 0)

        if mode == "ed" and attempts < 1 and needs_ed_followup(cc, op):
            return {
                "chief_complaint": cc,
                "opqrst": op,
                "subjective_complete": False,
                "triage_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text":
                    "Quick safety check: are you having shortness of breath, "
                    "fainting, sweating, or pain spreading to your arm or jaw?"}],
                "current_phase": "subjective",
            }

        triage = compute_basic_triage(mode, cc, op)
        return {
            "chief_complaint": cc,
            "opqrst": op,
            "triage": triage,
            "needs_emergency_review": False,
            "subjective_complete": True,
            "clinical_step": "allergies",
            "messages": [{"role": "assistant", "text":
                "Do you have any allergies — especially to medications or latex? "
                "If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    return {
        "chief_complaint": cc,
        "opqrst": op,
        "subjective_complete": False,
        "messages": [{"role": "assistant",
                      "text": reply or "When did it start, and how severe is it from 0-10?"}],
        "current_phase": "subjective",
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
        f"- Quality: {op.get('quality') or '—'}",
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

    if step == "allergies":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "Do you have any allergies — especially to medications or latex? "
                    "If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        allergies = extract_allergies_simple(user)
        return {
            "allergies": allergies,
            "clinical_step": "meds",
            "messages": [{"role": "assistant", "text":
                "What medications are you currently taking? "
                "Include the name, dose, how often, and when you last took it. "
                "If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "meds":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "What medications are you currently taking? "
                    "Include dose, frequency, and last taken time if you know. "
                    "If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }
        if user.strip().lower() in {"none", "no", "no meds", "not taking anything", "nothing"}:
                return {
                "medications": [],
                "clinical_step": "pmh",
                "messages": [{"role": "assistant", "text":
                    "Any past medical conditions or surgeries? If none, say 'none'."}],
                "current_phase": "clinical_history",
            }

        obj, meta = run_json_step(
            system=meds_extract_system(RESPONSE_RULES),
            prompt=f"NEW_USER_MESSAGE={user}",
            schema=MedsOut,
            fallback={"medications": [], "reply": "Could you list the medication names you take?"},
            temperature=0.2,
        )
        log_event("llm_step", thread_id=thread_id, node="medications", **meta)

        out    = obj.model_dump()
        parsed = out.get("medications") or []
        reply  = _safe_reply((out.get("reply") or "").strip())

        if not parsed:
            return {
                "messages": [{"role": "assistant", "text":
                    reply or "Could you list the medication names you take?"}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }
        return {
            "medications": parsed,
            "clinical_step": "pmh",
            "messages": [{"role": "assistant", "text":
                "Any past medical conditions or surgeries? If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "pmh":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "Any past medical conditions or surgeries? If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "pmh",
            }
        pmh = extract_list_simple(user)
        return {
            "pmh": pmh,
            "clinical_step": "results",
            "messages": [{"role": "assistant", "text":
                "Have you had any recent lab tests or imaging (bloodwork, X-ray, CT, etc.) "
                "since your last visit? If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "results":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text":
                    "Any recent lab tests or imaging since your last visit? If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "results",
            }
        results = extract_list_simple(user)
        summary = _confirm_summary({**state, "recent_results": results})
        return {
            "recent_results": results,
            "clinical_complete": True,
            "clinical_step": "done",
            "current_phase": "confirm",
            "messages": [{"role": "assistant", "text":
                summary + "\n\nReply 'confirm' to generate the clinician note, "
                "or tell me what you'd like to change."}],
        }

    return {"current_phase": "report"}


# ---------------------------------------------------------------------------
# Confirm node
# ---------------------------------------------------------------------------

def confirm_node(state: IntakeState):
    user = last_user(state).strip().lower()
    if user in {"confirm", "yes", "y", "ok", "okay", "looks good", "correct", "done"}:
        return {
            "current_phase": "report",
            "messages": [{"role": "assistant", "text": "Got it — generating the clinician note now."}],
        }
    if any(k in user for k in ["allerg", "med", "medicine", "medication", "pmh",
                                "history", "surgery", "test", "lab", "imaging"]):
        return {
            "current_phase": "clinical_history",
            "clinical_step": "allergies",
            "messages": [{"role": "assistant", "text":
                "Sure — let's update your medical history. "
                "Do you have any allergies?"}],
        }
    if any(k in user for k in ["pain", "symptom", "onset", "severity", "timing",
                                "radiat", "quality", "provocation", "complaint"]):
        return {
            "current_phase": "subjective",
            "messages": [{"role": "assistant", "text":
                "Sure — what would you like to change about your symptoms?"}],
        }
    if any(k in user for k in ["name", "phone", "address", "dob", "date of birth"]):
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


def report_node(state: IntakeState):
    identity = state.get("identity") or {}
    cc       = state.get("chief_complaint") or "Unknown/Not provided"
    op       = state.get("opqrst") or {}
    allergies = state.get("allergies") or []
    meds      = state.get("medications") or []
    pmh       = state.get("pmh") or []
    results   = state.get("recent_results") or []
    thread_id = state.get("thread_id", "")

    payload = {
        "identity": identity, "chief_complaint": cc,
        "opqrst": op, "allergies": allergies,
        "medications": meds, "pmh": pmh,
        "recent_results": results, "triage": state.get("triage") or {},
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
        allergies_line = "NKDA" if not allergies else ", ".join(allergies)
        report_text = (
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
    else:
        report_text = res.text.strip()

    # FHIR bundle — failure never blocks the clinician note
    fhir_json: str | None = None
    try:
        bundle    = fhir_builder.build_bundle(dict(state))
        fhir_json = json.dumps(bundle, indent=2)
        log_event("fhir_bundle_built", thread_id=thread_id,
                  resource_count=len(bundle.get("entry", [])))
    except Exception as e:
        log_event("fhir_bundle_error", level="warning",
                  thread_id=thread_id, error=str(e)[:200])

    triage = state.get("triage") or {}
    db.save_report(thread_id, triage.get("risk_level") or "low",
                   triage.get("visit_type") or "routine", report_text, fhir_json)
    db.set_session_status(thread_id, "active")


    log_event(
        "phi_audit_session_complete",
        thread_id=thread_id,
        identity_status=state.get("identity_status") or "unverified",
        mode=state.get("mode") or "clinic",
    )

    patient_name = (identity or {}).get("name") or "unknown patient"
    webhook.dispatch_intake_complete(
        thread_id=thread_id,
        patient_name=patient_name,
        risk_level=triage.get("risk_level") or "low",
        fhir_json=fhir_json,
    )

    return {
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