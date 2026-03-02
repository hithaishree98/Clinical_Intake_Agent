from __future__ import annotations
import hashlib
import hmac
import urllib.request
import json
import re
from typing import Dict
from .prompts import subjective_extract_system, meds_extract_system, report_system
from .state import IntakeState
from .schemas import SubjectiveOut, MedsOut
from .llm import run_json_step, get_gemini
from .logging_utils import log_event
from . import sqlite_db as db
from . import fhir_builder
from .extract import (
    extract_identity_deterministic,
    normalize_phone,
    is_yes,
    is_no,
    is_ack,
    detect_emergency_red_flags,
    extract_allergies_simple,
    extract_list_simple,
)

RESPONSE_RULES = "Be concise and human. Ask ONE question only if needed. Never invent facts. No diagnosis."
def _fire_completion_webhook(thread_id: str, fhir_json: str | None) -> None:
    """POST the FHIR Bundle to COMPLETION_WEBHOOK_URL if configured. Failures are logged, never raised."""
    from .settings import settings
    url = (settings.completion_webhook_url or "").strip()
    if not url or not fhir_json:
        return

    payload = fhir_json.encode("utf-8")
    secret = (settings.completion_webhook_secret or "").encode("utf-8")
    sig = hmac.new(secret, payload, hashlib.sha256).hexdigest() if secret else ""

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/fhir+json",
                "X-Thread-Id": thread_id,
                "X-Signature": f"sha256={sig}",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log_event("webhook_fired", thread_id=thread_id, status=resp.status)
    except Exception as e:
        log_event("webhook_error", level="warning", thread_id=thread_id, error=str(e)[:200])

def _slack(msg: str) -> None:
    """Post a message to Slack if SLACK_WEBHOOK_URL is configured. Failures are logged, never raised."""
    from .settings import settings
    url = (settings.slack_webhook_url or "").strip()
    if not url:
        return
    try:
        payload = json.dumps({"text": msg}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        log_event("slack_sent", msg_preview=msg[:80])
    except Exception as e:
        log_event("slack_error", level="warning", error=str(e)[:200])


def needs_ed_followup(cc: str, op: Dict[str, str]) -> bool:
    t = (cc or "").lower()

    # Lightweight check — intentionally avoids over-triggering.
    # The red-flag gate in subjective_node catches real emergencies.
    chest = any(k in t for k in ["chest pain", "chest tight", "chest pressure", "pressure in chest"])
    breathing = any(k in t for k in ["shortness of breath", "sob", "difficulty breathing"])
    neuro = any(k in t for k in ["weakness", "numbness", "slurred speech", "face droop", "confusion"])
    faint = any(k in t for k in ["faint", "passed out", "syncope"])

    # If breathing/neuro/faint is already mentioned, the red-flag gate handles it.
    if breathing or neuro or faint:
        return False

    if chest:
        return True

    return False


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
    if "severe" in s or "worst" in s:
        return 9
    if "moderate" in s:
        return 5
    if "mild" in s:
        return 2
    return -1


def compute_basic_triage(mode: str, cc: str, op: Dict[str, str]) -> Dict[str, str]:
    # Not a diagnosis — a safe, low-level disposition hint for ED mode only.
    base = {
        "emergency_flag": False,
        "risk_level": "low",
        "visit_type": "routine",
        "confidence": "medium",
        "rationale": "No emergency red flags detected in the intake.",
        "red_flags": [],
    }

    if mode != "ed":
        return base

    t = (cc or "").lower()
    sev = _severity_score(op)

    concerning = any(k in t for k in [
        "chest", "short of breath", "difficulty breathing", "faint", "passed out",
        "severe", "worst headache", "blood", "bleeding", "vision", "confusion"
    ])

    if sev >= 7 or concerning:
        return {
            **base,
            "risk_level": "medium",
            "visit_type": "urgent_care_today",
            "confidence": "medium",
            "rationale": "Symptoms sound significant or severity is high. Recommend evaluation today.",
        }

    if 0 <= sev <= 3:
        return {
            **base,
            "risk_level": "low",
            "visit_type": "clinic_24_72h",
            "confidence": "medium",
            "rationale": "Lower severity, no emergency keywords. Recommend clinic follow-up within 24-72 hours.",
        }

    return {
        **base,
        "risk_level": "low",
        "visit_type": "clinic_24_72h",
        "confidence": "low",
        "rationale": "Severity unclear. Recommend clinic follow-up within 24-72 hours unless symptoms worsen.",
    }


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


def identity_node(state: IntakeState):
    user = last_user(state).strip()
    identity = dict(state.get("identity") or {"name": "", "dob": "", "phone": "", "address": ""})
    attempts = int(state.get("identity_attempts") or 0)

    if attempts == 0 and not user:
        return {
            "messages": [{"role": "assistant", "text": "Hi — I'll collect intake info for the clinician. What's your full name?"}],
            "current_phase": "identity",
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": False,
        }

    det = extract_identity_deterministic(user)
    for k in ["name", "dob", "phone", "address"]:
        if det.get(k) and not (identity.get(k) or "").strip():
            identity[k] = det[k].strip()
    if identity.get("phone"):
        identity["phone"] = normalize_phone(identity["phone"])

    attempts += 1
    missing = [k for k in ["name", "dob", "phone", "address"] if not (identity.get(k) or "").strip()]
    if missing:
        q = {
            "name": "What's your full name?",
            "dob": "What's your date of birth? (MM/DD/YYYY)",
            "phone": "What's the best phone number to reach you?",
            "address": "What's your current address?",
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
                "I found stored info on file:\n"
                f"- {_summary_identity(stored)}\n\n"
                "You provided:\n"
                f"- {_summary_identity(identity)}\n\n"
                "Should I keep the stored info, or update it with what you provided? (keep/update)"
            }],
            "current_phase": "identity_review",
        }

    return {
        "identity": identity,
        "stored_identity": None,
        "identity_status": "unverified",
        "needs_identity_review": True,
        "messages": [{"role": "assistant", "text": f"Got it. I have: {_summary_identity(identity)}. Is this correct? (yes/no)"}],
        "current_phase": "identity_review",
    }


def identity_review_node(state: IntakeState):
    user = last_user(state).strip()
    identity = dict(state.get("identity") or {})
    stored = state.get("stored_identity")

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
                thread_id=state["thread_id"],
                kind="identity_review",
                payload={"stored_identity": stored, "new_identity": identity}
            )
            name = (identity or {}).get("name") or "unknown patient"
            _slack(f"[identity review] {name} provided details that differ from the EHR record. Session: {state['thread_id'][:8]}")
            return {
                "identity": identity,
                "identity_status": "unverified",
                "needs_identity_review": True,
                "messages": [{"role": "assistant", "text": "Okay — I'll use what you provided (a nurse may review). What brings you in today?"}],
                "current_phase": "subjective",
            }
        return {"messages": [{"role": "assistant", "text": "Please reply 'keep' or 'update'."}], "current_phase": "identity_review"}

    if is_yes(user):
        return {
            "identity_status": "verified",
            "needs_identity_review": False,
            "messages": [{"role": "assistant", "text": "Thanks. What's the main reason for your visit today? (in your own words)"}],
            "current_phase": "subjective",
        }
    if is_no(user):
        return {
            "identity": {"name": "", "dob": "", "phone": "", "address": ""},
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": True,
            "messages": [{"role": "assistant", "text": "Okay — what's your full name?"}],
            "current_phase": "identity",
        }
    return {"messages": [{"role": "assistant", "text": "Just to confirm — is that correct? (yes/no)"}], "current_phase": "identity_review"}


def subjective_node(state: IntakeState):
    user = last_user(state).strip()
    cc = (state.get("chief_complaint") or "").strip()
    op = dict(state.get("opqrst") or {"onset": "", "provocation": "", "quality": "", "radiation": "", "severity": "", "timing": ""})

    if not cc and (not user or is_ack(user) or is_yes(user) or is_no(user)):
        return {
            "messages": [{"role": "assistant", "text": "What's the main reason for your visit today? (in your own words)"}],
            "current_phase": "subjective",
        }

    flags = detect_emergency_red_flags(cc, op, user)
    if flags:
        triage = {
            "emergency_flag": True,
            "risk_level": "high",
            "visit_type": "emergency",
            "red_flags": flags,
            "confidence": "high",
            "rationale": "Red-flag phrase detected.",
        }
        db.create_escalation(thread_id=state["thread_id"], kind="emergency", payload={"triage": triage})
        db.set_session_status(state["thread_id"], "escalated")
        log_event("emergency_escalation", level="warning", thread_id=state["thread_id"], red_flags=flags)
        name = (state.get("identity") or {}).get("name") or "unknown patient"
        _slack(f"[emergency] {name} triggered an emergency escalation. Flags: {', '.join(flags)}. Session: {state['thread_id'][:8]}")
        return {
            "triage": triage,
            "needs_emergency_review": True,
            "messages": [{"role": "assistant", "text": "Based on what you shared, this could be urgent. Please call 911 or go to the nearest ER now. A clinician has been notified."}],
            "current_phase": "handoff",
        }

    prompt = f"CURRENT_STATE={json.dumps({'chief_complaint': cc, 'opqrst': op})}\nNEW_USER_MESSAGE={user}"
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
    log_event("llm_step", thread_id=state.get("thread_id"), node="subjective", **meta)

    out = obj.model_dump()
    new_cc = (out.get("chief_complaint") or "").strip()
    new_op = out.get("opqrst") or {}

    if new_cc and not cc:
        cc = new_cc
    for k in op.keys():
        v = (new_op.get(k) or "").strip()
        if v and not (op.get(k) or "").strip():
            op[k] = v

    complete = bool(out.get("is_complete"))
    reply = (out.get("reply") or "").strip()

    if complete:
        mode = state.get("mode") or "clinic"
        attempts = int(state.get("triage_attempts") or 0)

        if mode == "ed" and attempts < 1 and needs_ed_followup(cc, op):
            return {
                "chief_complaint": cc,
                "opqrst": op,
                "subjective_complete": False,
                "triage_attempts": attempts + 1,
                "messages": [{"role": "assistant", "text": "Quick safety check: are you having shortness of breath, fainting, sweating, or pain spreading to your arm/jaw?"}],
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
            "messages": [{"role": "assistant", "text": "Do you have any allergies (especially medications or latex)? If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    return {
        "chief_complaint": cc,
        "opqrst": op,
        "subjective_complete": False,
        "messages": [{"role": "assistant", "text": reply or "When did it start, and how severe is it from 0-10?"}],
        "current_phase": "subjective",
    }


def _confirm_summary(state: IntakeState) -> str:
    identity = state.get("identity") or {}
    cc = state.get("chief_complaint") or "—"
    op = state.get("opqrst") or {}
    allergies = state.get("allergies") or []
    meds = state.get("medications") or []
    pmh = state.get("pmh") or []
    results = state.get("recent_results") or []
    triage = state.get("triage") or {}

    def fmt_list(xs):
        return "None" if not xs else ", ".join(xs)

    def fmt_meds(ms):
        if not ms:
            return "None"
        parts = []
        for m in ms:
            name = (m.get("name") or "").strip() or "Unknown"
            dose = (m.get("dose") or "").strip()
            freq = (m.get("freq") or "").strip()
            last = (m.get("last_taken") or "").strip()
            s = name
            if dose: s += f" {dose}"
            if freq: s += f" ({freq})"
            if last: s += f", last: {last}"
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
        lines += [
            "",
            "Triage",
            f"- Risk: {triage.get('risk_level') or '—'}",
            f"- Visit type: {triage.get('visit_type') or '—'}",
        ]
    return "\n".join(lines)


def clinical_history_node(state: IntakeState):
    user = last_user(state).strip()
    step = state.get("clinical_step") or "allergies"

    if step == "allergies":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": "Do you have any allergies (especially medications or latex)? If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "allergies",
            }
        return {
            "allergies": extract_allergies_simple(user),
            "clinical_step": "meds",
            "messages": [{"role": "assistant", "text": "What medications are you currently taking? Include dose, how often, and when you last took it (if you know). If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "meds":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": "What medications are you currently taking? Include dose, frequency, and last taken time if you know. If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }
        if user.strip().lower() in {"none", "no", "no meds", "not taking anything"}:
            return {
                "medications": [],
                "clinical_step": "pmh",
                "messages": [{"role": "assistant", "text": "Any past medical conditions or past surgeries? If none, say 'none'."}],
                "current_phase": "clinical_history",
            }

        obj, meta = run_json_step(
            system=meds_extract_system(RESPONSE_RULES),
            prompt=f"NEW_USER_MESSAGE={user}",
            schema=MedsOut,
            fallback={"medications": [], "reply": "Could you list the medication names you take?"},
            temperature=0.2,
        )
        log_event("llm_step", thread_id=state.get("thread_id"), node="medications", **meta)

        out = obj.model_dump()
        parsed = out.get("medications") or []
        reply = (out.get("reply") or "").strip()

        if not parsed:
            return {
                "messages": [{"role": "assistant", "text": reply or "Could you list the medication names you take?"}],
                "current_phase": "clinical_history",
                "clinical_step": "meds",
            }
        return {
            "medications": parsed,
            "clinical_step": "pmh",
            "messages": [{"role": "assistant", "text": "Any past medical conditions or past surgeries? If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "pmh":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": "Any past medical conditions or past surgeries? If none, say 'none'."}],
                "current_phase": "clinical_history",
                "clinical_step": "pmh",
            }
        return {
            "pmh": extract_list_simple(user),
            "clinical_step": "results",
            "messages": [{"role": "assistant", "text": "Have you had any recent lab tests or imaging (bloodwork, X-ray, CT, etc.) since your last visit? If none, say 'none'."}],
            "current_phase": "clinical_history",
        }

    if step == "results":
        if not user or is_ack(user):
            return {
                "messages": [{"role": "assistant", "text": "Any recent lab tests or imaging since your last visit? If none, say 'none'."}],
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
            "messages": [{"role": "assistant", "text": summary + "\n\nReply 'confirm' to generate the clinician note, or tell me what you want to change."}],
        }

    return {"current_phase": "report"}


def confirm_node(state: IntakeState):
    user = last_user(state).strip().lower()

    if user in {"confirm", "yes", "y", "ok", "okay", "looks good", "correct"}:
        return {
            "current_phase": "report",
            "messages": [{"role": "assistant", "text": "Got it — generating the clinician note now."}],
        }

    if any(k in user for k in ["allerg", "med", "medicine", "medication", "pmh", "history", "surgery", "test", "lab", "imaging"]):
        return {
            "current_phase": "clinical_history",
            "clinical_step": "allergies",
            "messages": [{"role": "assistant", "text": "Sure — what would you like to update in your medical history?"}],
        }

    if any(k in user for k in ["pain", "symptom", "onset", "severity", "timing", "radiat", "quality", "provocation", "complaint"]):
        return {
            "current_phase": "subjective",
            "messages": [{"role": "assistant", "text": "Sure — what would you like to change about your symptoms?"}],
        }

    if any(k in user for k in ["name", "phone", "address"]):
        return {
            "current_phase": "identity",
            "identity_attempts": 0,
            "messages": [{"role": "assistant", "text": "Sure — what should I update in your identity details?"}],
        }

    return {
        "current_phase": "confirm",
        "messages": [{"role": "assistant", "text": "Reply 'confirm' to proceed, or tell me what you want to change (symptoms, history, or identity)."}],
    }


def _fmt_meds_fallback(meds: list) -> str:
    """Format medications for the plain-text fallback report."""
    if not meds:
        return "- None/Unknown"
    lines = []
    for m in meds:
        name = (m.get("name") or "").strip() or "Unknown"
        dose = (m.get("dose") or "").strip()
        freq = (m.get("freq") or "").strip()
        last = (m.get("last_taken") or "").strip()
        line = f"- {name}"
        if dose: line += f", {dose}"
        if freq: line += f", {freq}"
        if last: line += f" (last taken: {last})"
        lines.append(line)
    return "\n".join(lines)


def report_node(state: IntakeState):
    identity = state.get("identity") or {}
    cc = state.get("chief_complaint") or "Unknown/Not provided"
    op = state.get("opqrst") or {}
    allergies = state.get("allergies") or []
    meds = state.get("medications") or []
    pmh = state.get("pmh") or []
    results = state.get("recent_results") or []

    payload = {
        "identity": identity,
        "chief_complaint": cc,
        "opqrst": op,
        "allergies": allergies,
        "medications": meds,
        "pmh": pmh,
        "recent_results": results,
        "triage": state.get("triage") or {},
    }

    res = get_gemini().generate_text(
        system=report_system(),
        prompt=json.dumps(payload, indent=2),
        temperature=0.2,
        max_tokens=1300,
        response_mime_type="text/plain",
    )

    if not res.ok or not res.text.strip():
        # LLM failed — build a plain-text report from state directly
        log_event("report_fallback_used", level="warning", thread_id=state.get("thread_id"), error=res.error)
        allergies_line = "None/Unknown" if not allergies else ", ".join(allergies)
        report_text = f"""Subjective Intake (Why)
Chief Complaint (CC): {cc}

HPI (OPQRST):
- Onset: {op.get("onset") or "Unknown/Not provided"}
- Provocation: {op.get("provocation") or "Unknown/Not provided"}
- Quality: {op.get("quality") or "Unknown/Not provided"}
- Radiation: {op.get("radiation") or "Unknown/Not provided"}
- Severity: {op.get("severity") or "Unknown/Not provided"}
- Timing: {op.get("timing") or "Unknown/Not provided"}

Clinical History & Safety
ALLERGIES (IMPORTANT): {allergies_line}

Current Medications:
{_fmt_meds_fallback(meds)}

PMH:
{", ".join(pmh) if pmh else "Unknown/Not provided"}

Recent Lab/Imaging Results:
{", ".join(results) if results else "None/Unknown"}

Identity
- Name: {identity.get("name") or "Unknown"}
- DOB: {identity.get("dob") or "Unknown"}
- Phone: {identity.get("phone") or "Unknown"}
- Address: {identity.get("address") or "Unknown"}
"""
    else:
        report_text = res.text.strip()

    # Build FHIR R4 Bundle from the same state data.
    # We do this after the plain-text report so a FHIR failure never
    # blocks the clinician note from being saved.
    fhir_json: str | None = None
    try:
        bundle = fhir_builder.build_bundle(dict(state))
        fhir_json = json.dumps(bundle, indent=2)
        log_event("fhir_bundle_built", thread_id=state.get("thread_id"),
                  resource_count=len(bundle.get("entry", [])))
    except Exception as e:
        log_event("fhir_bundle_error", level="warning",
                  thread_id=state.get("thread_id"), error=str(e)[:200])

    triage = state.get("triage") or {}
    db.save_report(
        state["thread_id"],
        triage.get("risk_level") or "low",
        triage.get("visit_type") or "routine",
        report_text,
        fhir_json,
    )
    db.set_session_status(state["thread_id"], "active")
    name = (state.get("identity") or {}).get("name") or "unknown patient"
    risk = triage.get("risk_level") or "low"
    _slack(f"[intake complete] {name} finished intake. Risk level: {risk}. Session: {state['thread_id'][:8]}")
    _fire_completion_webhook(state["thread_id"], fhir_json)

    return {
        "messages": [{"role": "assistant", "text": "Intake complete. Your report is ready. Click \"View Report\"."}],
        "current_phase": "done",
    }


def handoff_node(state: IntakeState):
    return {
        "messages": [{"role": "assistant", "text": "To prioritize your safety, please call 911 or go to the nearest ER now. A clinician has been notified."}],
        "current_phase": "handoff",
    }