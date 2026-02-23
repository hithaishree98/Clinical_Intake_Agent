from __future__ import annotations
import json
from typing import Dict
from .prompts import subjective_extract_system, meds_extract_system, report_system
from .state import IntakeState
from .schemas import SubjectiveOut, MedsOut
from .llm import run_json_step, get_gemini
from .logging_utils import log_event
from . import sqlite_db as db
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

RESPONSE_RULES = "Be concise and human. Ask ONE question only if needed. Never invent facts. No diagnosis.".strip()

def needs_ed_followup(cc: str, op: Dict[str, str]) -> bool:
    t = (cc or "").lower()

    # Simple high-risk patterns that should trigger a clarifying question in ED mode. This is intentionally lightweight to avoid over-triggering, and relies on the triage node to catch emergencies even if the user doesn't acknowledge these symptoms.
    chest = any(k in t for k in ["chest pain", "chest tight", "chest pressure", "pressure in chest"])
    breathing = any(k in t for k in ["shortness of breath", "sob", "difficulty breathing"])
    neuro = any(k in t for k in ["weakness", "numbness", "slurred speech", "face droop", "confusion"])
    faint = any(k in t for k in ["faint", "passed out", "syncope"])

    # If it already explicitly includes breathing/neuro/faint, no need to clarifier here (red-flag gate will catch it).
    if breathing or neuro or faint:
        return False

    # Chest complaint is the classic "ask one more question" situation
    if chest:
        return True

    return False

def _severity_score(op: Dict[str, str]) -> int:
    s = (op.get("severity") or "").lower()
    # try to parse a number 0-10
    import re
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        try:
            n = int(m.group(1))
            if 0 <= n <= 10:
                return n
        except:
            pass
    if "severe" in s or "worst" in s:
        return 9
    if "moderate" in s:
        return 5
    if "mild" in s:
        return 2
    return -1


def compute_basic_triage(mode: str, cc: str, op: Dict[str, str]) -> Dict[str, str]:
    """
    NOT diagnosis. Just a safe, low-level disposition suggestion.
    - Only meaningful for mode='ed'
    """
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

    # If it sounds like a potentially serious complaint but not an emergency phrase,
    # recommend evaluation today (urgent care) instead of "self care".
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
            "rationale": "Symptoms sound significant (or severity is high). Recommend evaluation today. No emergency keywords detected.",
        }

    if 0 <= sev <= 3:
        return {
            **base,
            "risk_level": "low",
            "visit_type": "clinic_24_72h",
            "confidence": "medium",
            "rationale": "Symptoms appear lower severity with no emergency keywords. Recommend clinic follow-up within 24–72 hours if symptoms persist.",
        }

    # Unknown severity → still suggest clinic soon
    return {
        **base,
        "risk_level": "low",
        "visit_type": "clinic_24_72h",
        "confidence": "low",
        "rationale": "Severity unclear. Recommend clinic follow-up within 24–72 hours unless symptoms worsen.",
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
    identity = dict(state.get("identity") or {"name":"","dob":"","phone":"","address":""})
    attempts = int(state.get("identity_attempts") or 0)

    if attempts == 0 and not user:
        return {
            "messages": [{"role":"assistant","text":"Hi — I’ll collect intake info for the clinician. What’s your full name?"}],
            "current_phase": "identity",
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": False,
        }

    det = extract_identity_deterministic(user)
    for k in ["name","dob","phone","address"]:
        if det.get(k) and not (identity.get(k) or "").strip():
            identity[k] = det[k].strip()
    if identity.get("phone"):
        identity["phone"] = normalize_phone(identity["phone"])

    attempts += 1
    missing = [k for k in ["name","dob","phone","address"] if not (identity.get(k) or "").strip()]
    if missing:
        q = {
            "name": "What’s your full name?",
            "dob": "What’s your date of birth? (MM/DD/YYYY)",
            "phone": "What’s the best phone number to reach you?",
            "address": "What’s your current address?",
        }[missing[0]]
        return {
            "identity": identity,
            "identity_attempts": attempts,
            "identity_status": "unverified",
            "messages": [{"role":"assistant","text": q}],
            "current_phase": "identity",
        }

    stored = db.get_stored_identity_by_name(identity["name"])
    if stored:
        return {
            "identity": identity,
            "stored_identity": stored,
            "messages": [{"role":"assistant","text":
                "I found stored info on file:\n"
                f"- { _summary_identity(stored) }\n\n"
                "You provided:\n"
                f"- { _summary_identity(identity) }\n\n"
                "Should I keep the stored info, or update it with what you provided? (keep/update)"
            }],
            "current_phase": "identity_review",
        }

    return {
        "identity": identity,
        "stored_identity": None,
        "identity_status": "unverified",
        "needs_identity_review": True,  # nurse can verify later if needed
        "messages": [{"role":"assistant","text": f"Got it. I have: {_summary_identity(identity)}. Is this correct? (yes/no)"}],
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
                "messages": [{"role":"assistant","text":"Thanks — I’ll keep what’s on file. What brings you in today?"}],
                "current_phase": "subjective",
            }
        if user.startswith("update"):
            db.create_escalation(
                thread_id=state["thread_id"],
                kind="identity_review",
                payload={"stored_identity": stored, "new_identity": identity}
            )
            return {
                "identity": identity,
                "identity_status": "unverified",
                "needs_identity_review": True,
                "messages": [{"role":"assistant","text":"Okay — I’ll use what you provided (a nurse may review). What brings you in today?"}],
                "current_phase": "subjective",
            }
        return {"messages":[{"role":"assistant","text":"Please reply 'keep' or 'update'."}], "current_phase":"identity_review"}

    if is_yes(user):
        return {
            "identity_status": "verified",
            "needs_identity_review": False,
            "messages": [{"role":"assistant","text":"Thanks. What’s the main reason for your visit today? (in your own words)"}],
            "current_phase": "subjective",
        }
    if is_no(user):
        
        return {
            "identity": {"name":"","dob":"","phone":"","address":""},
            "identity_attempts": 0,
            "identity_status": "unverified",
            "needs_identity_review": True,
            "messages": [{"role":"assistant","text":"Okay — what’s your full name?"}],
            "current_phase": "identity",
        }
    return {"messages":[{"role":"assistant","text":"Just to confirm — is that correct? (yes/no)"}], "current_phase":"identity_review"}


def subjective_node(state: IntakeState):
    user = last_user(state).strip()
    cc = (state.get("chief_complaint") or "").strip()
    op = dict(state.get("opqrst") or {"onset":"","provocation":"","quality":"","radiation":"","severity":"","timing":""})


    if not cc and (not user or is_ack(user) or is_yes(user) or is_no(user)):
        return {"messages":[{"role":"assistant","text":"What’s the main reason for your visit today? (in your own words)"}],
                "current_phase":"subjective"}

    flags = detect_emergency_red_flags(cc, op, user)
    if flags:
        triage = {
            "emergency_flag": True,
            "risk_level": "high",
            "visit_type": "emergency",
            "red_flags": flags,
            "confidence": "high",
            "rationale": "Red-flag phrase detected."
        }
        db.create_escalation(thread_id=state["thread_id"], kind="emergency", payload={"triage": triage})
        db.set_session_status(state["thread_id"], "escalated")
        log_event("escalation_created", level="warn", thread_id=state["thread_id"], kind="emergency", red_flags=flags)
        return {
            "triage": triage,
            "needs_emergency_review": True,
            "messages": [{"role":"assistant","text":"Based on what you shared, this could be urgent. Please call 911 or go to the nearest ER now. A clinician has been notified."}],
            "current_phase":"handoff",
        }

    system = subjective_extract_system(RESPONSE_RULES)

    prompt = f"CURRENT_STATE={json.dumps({'chief_complaint': cc, 'opqrst': op})}\nNEW_USER_MESSAGE={user}"

    obj, meta = run_json_step(
        system=system,
        prompt=prompt,
        schema=SubjectiveOut,
        fallback={
            "chief_complaint": cc,
            "opqrst": op,
            "is_complete": False,
            "reply": "When did it start, and how severe is it from 0–10?"
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

        # ED mode: ask ONE targeted clarifier if needed (only once)
        if mode == "ed" and attempts < 1 and needs_ed_followup(cc, op):
            return {
                "chief_complaint": cc,
                "opqrst": op,
                "subjective_complete": False,
                "triage_attempts": attempts + 1,
                "messages": [{
                    "role": "assistant",
                    "text": "Quick safety check: are you having shortness of breath, fainting, sweating, or pain spreading to your arm/jaw?"
                }],
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
            "messages": [{"role":"assistant","text":"Do you have any allergies (especially medications or latex)? If none, say 'none'."}],
            "current_phase":"clinical_history",
        }

    return {
        "chief_complaint": cc,
        "opqrst": op,
        "subjective_complete": False,
        "messages": [{"role":"assistant","text": reply or "When did it start, and how severe is it from 0–10?"}],
        "current_phase":"subjective",
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

    lines = []
    lines.append("Here’s what I captured:")
    lines.append("")
    lines.append("Identity")
    lines.append(f"- Name: {identity.get('name') or '—'}")
    lines.append(f"- DOB: {identity.get('dob') or '—'}")
    lines.append(f"- Phone: {identity.get('phone') or '—'}")
    lines.append(f"- Address: {identity.get('address') or '—'}")
    lines.append("")
    lines.append("Symptoms")
    lines.append(f"- Chief complaint: {cc}")
    if op:
        lines.append(f"- Onset: {op.get('onset') or '—'}")
        lines.append(f"- Provocation: {op.get('provocation') or '—'}")
        lines.append(f"- Quality: {op.get('quality') or '—'}")
        lines.append(f"- Radiation: {op.get('radiation') or '—'}")
        lines.append(f"- Severity: {op.get('severity') or '—'}")
        lines.append(f"- Timing: {op.get('timing') or '—'}")
    lines.append("")
    lines.append("History")
    lines.append(f"- Allergies: {fmt_list(allergies)}")
    lines.append(f"- Medications: {fmt_meds(meds)}")
    lines.append(f"- PMH: {fmt_list(pmh)}")
    lines.append(f"- Recent results: {fmt_list(results)}")
    if triage:
        lines.append("")
        lines.append("Triage")
        lines.append(f"- Risk: {triage.get('risk_level') or '—'}")
        lines.append(f"- Visit type: {triage.get('visit_type') or '—'}")

    return "\n".join(lines)

def clinical_history_node(state: IntakeState):
    user = last_user(state).strip()

    step = state.get("clinical_step") or "allergies"
    allergies = list(state.get("allergies") or [])
    meds = list(state.get("medications") or [])
    pmh = list(state.get("pmh") or [])
    results = list(state.get("recent_results") or [])

    if step == "allergies":
        if not user or is_ack(user):
            return {
                "messages":[{"role":"assistant","text":"Do you have any allergies (especially medications or latex)? If none, say 'none'."}],
                "current_phase":"clinical_history",
                "clinical_step":"allergies",
            }
        allergies = extract_allergies_simple(user)
        return {
            "allergies": allergies,
            "clinical_step": "meds",
            "messages":[{"role":"assistant","text":"What medications are you currently taking? Include dose, how often, and when you last took it (if you know). If none, say 'none'."}],
            "current_phase":"clinical_history",
        }

    if step == "meds":
        if not user or is_ack(user):
            return {
                "messages":[{"role":"assistant","text":"What medications are you currently taking? Include dose, frequency, and last taken time if you know. If none, say 'none'."}],
                "current_phase":"clinical_history",
                "clinical_step":"meds",
            }
        if user.strip().lower() in {"none", "no", "no meds", "not taking anything"}:
            return {
                "medications": [],
                "clinical_step": "pmh",
                "messages":[{"role":"assistant","text":"Any past medical conditions or past surgeries? If none, say 'none'."}],
                "current_phase":"clinical_history",
            }

        system = meds_extract_system(RESPONSE_RULES)

        prompt = f"NEW_USER_MESSAGE={user}"
        obj, meta = run_json_step(
            system=system,
            prompt=prompt,
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
                "messages":[{"role":"assistant","text": reply or "Could you list the medication names you take?"}],
                "current_phase":"clinical_history",
                "clinical_step":"meds",
            }

        return {
            "medications": parsed,
            "clinical_step": "pmh",
            "messages":[{"role":"assistant","text":"Any past medical conditions or past surgeries? If none, say 'none'."}],
            "current_phase":"clinical_history",
        }

    if step == "pmh":
        if not user or is_ack(user):
            return {
                "messages":[{"role":"assistant","text":"Any past medical conditions or past surgeries? If none, say 'none'."}],
                "current_phase":"clinical_history",
                "clinical_step":"pmh",
            }
        pmh = extract_list_simple(user)
        return {
            "pmh": pmh,
            "clinical_step": "results",
            "messages":[{"role":"assistant","text":"Have you had any recent lab tests or imaging (bloodwork, X-ray, CT, etc.) since your last visit? If none, say 'none'."}],
            "current_phase":"clinical_history",
        }

    if step == "results":
        if not user or is_ack(user):
            return {
                "messages":[{"role":"assistant","text":"Any recent lab tests or imaging since your last visit? If none, say 'none'."}],
                "current_phase":"clinical_history",
                "clinical_step":"results",
            }
        results = extract_list_simple(user)
              
        temp_state = dict(state)
        temp_state["recent_results"] = results

        summary = _confirm_summary(temp_state)

        return {
            "recent_results": results,
            "clinical_complete": True,
            "clinical_step": "done",
            "current_phase": "confirm",
            "messages": [{
                "role": "assistant",
                "text": summary + "\n\nReply 'confirm' to generate the clinician note, or tell me what you want to change."
            }],
        }

    return {"current_phase":"report"}

def confirm_node(state: IntakeState):
    user = last_user(state).strip().lower()

    # If user confirms, move to report generation
    if user in {"confirm", "yes", "y", "ok", "okay", "looks good", "correct"}:
        return {
            "current_phase": "report",
            "messages": [{"role": "assistant", "text": "Got it — generating the clinician note now."}],
        }

    # If user wants to change something, route them back to the right phase (simple keyword routing)
    # Keep this intentionally lightweight.
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

    # Default: ask for either confirm or a clear edit
    return {
        "current_phase": "confirm",
        "messages": [{"role": "assistant", "text": "Reply 'confirm' to proceed, or tell me what you want to change (symptoms, history, or identity)."}],
    }

def report_node(state: IntakeState):
    identity = state.get("identity") or {}
    cc = state.get("chief_complaint") or "Unknown/Not provided"
    op = state.get("opqrst") or {}

    allergies = state.get("allergies") or []
    meds = state.get("medications") or []
    pmh = state.get("pmh") or []
    results = state.get("recent_results") or []


    system = report_system()

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
        system=system,
        prompt=json.dumps(payload, indent=2),
        temperature=0.2,
        max_tokens=1300,
        response_mime_type="text/plain",
    )

    if not res.ok or not res.text.strip():
        allergies_line = "None/Unknown" if not allergies else ", ".join(allergies)

        def fmt_meds():
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
{fmt_meds()}

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


    triage = state.get("triage") or {}
    risk_level = triage.get("risk_level") or "low"
    visit_type = triage.get("visit_type") or "routine"

    db.save_report(state["thread_id"], risk_level, visit_type, report_text)
    db.set_session_status(state["thread_id"], "active")

    return {
        "messages":[{"role":"assistant","text":"Intake complete. Your report is ready. Click “View Report”."}],
        "current_phase":"done",
    }

def handoff_node(state: IntakeState):
    return {
        "messages":[{"role":"assistant","text":"To prioritize your safety, please call 911 or go to the nearest ER now. A clinician has been notified."}],
        "current_phase":"handoff",
    }
