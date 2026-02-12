from __future__ import annotations
import json
from typing import Dict

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

STYLE = """
STYLE:
- Be concise and human.
- Ask exactly ONE question if more info is needed; otherwise ask ZERO questions.
- Never invent facts.
- No diagnosis or medical advice.
""".strip()

def last_user(state: IntakeState) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if m["role"] == "user":
            return m["text"]
    return ""

def _summary_identity(x: Dict[str, str]) -> str:
    return f"Name: {x.get('name') or '—'}, Phone: {x.get('phone') or '—'}, Address: {x.get('address') or '—'}"


def identity_node(state: IntakeState):
    user = last_user(state).strip()
    identity = dict(state.get("identity") or {"name":"","phone":"","address":""})
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
    for k in ["name","phone","address"]:
        if det.get(k) and not (identity.get(k) or "").strip():
            identity[k] = det[k].strip()
    if identity.get("phone"):
        identity["phone"] = normalize_phone(identity["phone"])

    attempts += 1
    missing = [k for k in ["name","phone","address"] if not (identity.get(k) or "").strip()]
    if missing:
        q = {
            "name": "What’s your full name?",
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
            "identity": {"name":"","phone":"","address":""},
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

    system = f"""
{STYLE}

You are an intake nurse. Extract/update chief complaint and OPQRST from NEW_USER_MESSAGE.
Return ONLY JSON:
{{
  "chief_complaint": "",
  "opqrst": {{"onset":"","provocation":"","quality":"","radiation":"","severity":"","timing":""}},
  "is_complete": bool,
  "reply": ""
}}

Rules:
- Never erase existing non-empty fields.
- If message is only a number, treat it as severity if severity missing.
- Ask EXACTLY ONE best next question if incomplete.
- If complete, reply must be "".
Completion requires: chief_complaint + severity + (onset OR timing).
""".strip()

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
        return {
            "chief_complaint": cc,
            "opqrst": op,
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

        system = f"""
{STYLE}

Extract a medication list from NEW_USER_MESSAGE.
Return ONLY JSON:
{{
  "medications": [{{"name":"","dose":"","freq":"","last_taken":""}}],
  "reply": ""
}}

Rules:
- Do NOT invent dose/frequency/last_taken.
- If you can’t find any medication name, ask ONE question in reply.
- If at least one medication name is found, reply must be "".
""".strip()

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
        return {
            "recent_results": results,
            "clinical_complete": True,
            "clinical_step": "done",
            "messages":[{"role":"assistant","text":"Thanks — I’m going to generate your clinician note now."}],
            "current_phase":"report",
        }

    return {"current_phase":"report"}

def report_node(state: IntakeState):
    identity = state.get("identity") or {}
    cc = state.get("chief_complaint") or "Unknown/Not provided"
    op = state.get("opqrst") or {}

    allergies = state.get("allergies") or []
    meds = state.get("medications") or []
    pmh = state.get("pmh") or []
    results = state.get("recent_results") or []


    system = """
You are a senior clinical scribe. Produce a concise clinician note.
Do NOT diagnose. If missing, write "Unknown/Not provided".
Return plain text only.

Include sections:
1) Subjective Intake (Why)
- Chief Complaint (CC)
- HPI using OPQRST bullets

2) Clinical History & Safety (highlight allergies)
- Allergies (IMPORTANT)
- Current Medications (include dose/frequency/last taken if present)
- PMH
- Recent Lab/Imaging Results

3) Identity
""".strip()

    payload = {
        "identity": identity,
        "chief_complaint": cc,
        "opqrst": op,
        "allergies": allergies,
        "medications": meds,
        "pmh": pmh,
        "recent_results": results,
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
