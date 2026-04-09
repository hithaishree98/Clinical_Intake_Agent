"""
patient.py — Patient-facing API endpoints.

Endpoints here require only a session token (issued at /start).
No clinician JWT is needed, which means the attack surface is limited:
a patient can only read their own session data.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import sqlite_db as db
from ..extract import check_prompt_injection
from ..graph import build_graph
from ..logging_utils import log_event, log_audit, set_trace_id, set_request_id, set_job_id
from ..settings import get_settings
from .deps import limiter, require_session_token, require_clinician

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers shared across patient routes
# ---------------------------------------------------------------------------

def _issue_session_token() -> str:
    return secrets.token_hex(32)


def _compact_snapshot(output: dict) -> dict:
    """Strip the full message list and other noise before persisting state."""
    return {
        "current_phase":          output.get("current_phase"),
        "identity":               output.get("identity"),
        "stored_identity":        output.get("stored_identity"),
        "identity_status":        output.get("identity_status"),
        "needs_identity_review":  output.get("needs_identity_review"),
        "consent_given":          output.get("consent_given"),
        "chief_complaint":        output.get("chief_complaint"),
        "opqrst":                 output.get("opqrst"),
        "allergies":              output.get("allergies"),
        "medications":            output.get("medications"),
        "pmh":                    output.get("pmh"),
        "recent_results":         output.get("recent_results"),
        "triage":                 output.get("triage"),
        "needs_emergency_review": output.get("needs_emergency_review"),
        "clinical_step":          output.get("clinical_step"),
        "mode":                   output.get("mode"),
        "triage_attempts":        output.get("triage_attempts"),
        "intake_classification":       output.get("intake_classification"),
        "classification_confidence":   output.get("classification_confidence"),
        "extraction_quality_score":    output.get("extraction_quality_score"),
        "extraction_retry_count":      output.get("extraction_retry_count"),
        "validation_errors":           output.get("validation_errors"),
        "validation_target_phase":     output.get("validation_target_phase"),
        "crisis_detected":        output.get("crisis_detected"),
        "human_review_required":  output.get("human_review_required"),
        "safety_score":           output.get("safety_score"),
        "extraction_confidence":  output.get("extraction_confidence"),
        "last_failed_phase":      output.get("last_failed_phase"),
        "last_failure_reason":    output.get("last_failure_reason"),
    }


def _run_report_job(graph, thread_id: str, job_id: str) -> None:
    try:
        set_trace_id(thread_id)
        set_job_id(job_id)
        db.update_job(job_id, "running")
        config = {"configurable": {"thread_id": thread_id}}
        output = graph.invoke({"messages": []}, config)

        phase = output.get("current_phase")
        if phase == "handoff" and output.get("human_review_required"):
            err = "report_blocked_preflight: required fields missing — clinician review required"
            db.update_job(job_id, "failed", error=err)
            log_event("report_job_blocked", level="warning",
                      thread_id=thread_id, job_id=job_id,
                      safety_score=output.get("safety_score"))
        else:
            db.update_job(job_id, "done")
            log_event("report_job_done", thread_id=thread_id, job_id=job_id)

    except Exception as e:
        db.update_job(job_id, "failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        log_event("report_job_failed", level="error",
                  thread_id=thread_id, job_id=job_id, error=str(e)[:400])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start")
@limiter.limit("10/hour")
def start_session(request: Request, mode: str = Form("clinic")):
    settings = get_settings()
    thread_id = str(uuid.uuid4())
    set_trace_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "current_phase": "consent" if settings.require_consent else "identity",
        "consent_given": not settings.require_consent,
        "mode": "ed" if (mode or "").strip().lower() == "ed" else "clinic",
        "triage_attempts": 0,
        "identity": {"name": "", "phone": "", "address": "", "dob": ""},
        "stored_identity": None,
        "identity_attempts": 0,
        "identity_status": "unverified",
        "needs_identity_review": False,
        "chief_complaint": "",
        "opqrst": {"onset": "", "provocation": "", "quality": "", "radiation": "", "severity": "", "timing": ""},
        "subjective_complete": False,
        "clinical_step": "allergies",
        "allergies": [],
        "medications": [],
        "pmh": [],
        "recent_results": [],
        "clinical_complete": False,
        "triage": {"emergency_flag": False, "risk_level": "low", "visit_type": "routine",
                   "red_flags": [], "confidence": "low", "rationale": ""},
        "needs_emergency_review": False,
        "intake_classification": None,
        "classification_confidence": None,
        "extraction_quality_score": None,
        "extraction_retry_count": 0,
        "validation_errors": [],
        "validation_target_phase": None,
        "crisis_detected": False,
        "human_review_required": False,
        "human_review_reasons": [],
        "safety_score": None,
        "extraction_confidence": None,
        "last_failed_phase": None,
        "last_failure_reason": None,
        "messages": [],
    }

    session_token = _issue_session_token()
    db.create_session(thread_id, session_token=session_token)
    log_event("session_started", thread_id=thread_id, mode=initial_state["mode"])

    graph = request.app.state.graph
    t0 = time.time()
    output = graph.invoke(initial_state, config)
    db.save_session_state(thread_id, _compact_snapshot(output))
    log_event("session_ready", thread_id=thread_id, duration_ms=int((time.time() - t0) * 1000))

    messages = output.get("messages") or []
    reply = messages[-1]["text"] if messages else "Welcome. Let's begin your intake."
    db.save_message(thread_id, "assistant", reply)
    return {
        "thread_id": thread_id,
        "session_token": session_token,
        "reply": reply,
        "phase": output.get("current_phase") or "identity",
        "status": "active",
    }


@router.get("/resume/{thread_id}")
def resume_session(thread_id: str, authorization: str = Header(default="")):
    require_session_token(thread_id, authorization)
    sess = db.fetch_one(
        "SELECT thread_id, status FROM sessions WHERE thread_id=?", (thread_id,)
    )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess["status"] in ("done", "escalated", "expired"):
        raise HTTPException(status_code=410, detail=f"Session already {sess['status']}.")

    last_msg = db.fetch_one(
        "SELECT text FROM messages WHERE thread_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
        (thread_id,),
    )
    state = db.get_session_state(thread_id)
    phase = ((state or {}).get("state") or {}).get("current_phase", "identity")
    return {
        "thread_id": thread_id,
        "status": sess["status"],
        "phase": phase,
        "reply": last_msg["text"] if last_msg else "Welcome back. Let's continue your intake.",
    }


@router.post("/chat")
@limiter.limit("60/minute")
def chat(
    request: Request,
    thread_id: str = Form(...),
    message: str = Form(...),
    client_msg_id: str = Form(...),
    authorization: str = Header(default=""),
):
    require_session_token(thread_id, authorization)

    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 1200:
        raise HTTPException(status_code=400, detail="Message too long (max 1200 chars).")
    if len(client_msg_id) > 128:
        raise HTTPException(status_code=400, detail="client_msg_id too long (max 128 chars).")
    if check_prompt_injection(message):
        return {
            "reply": "I can only collect intake information for your visit. "
                     "If you have a question for your care team, they'll be happy to help when you arrive.",
            "status": "active",
            "phase": "unknown",
        }

    request_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()

    prev = db.get_idempotent_response(thread_id, client_msg_id)
    if prev:
        if (prev.get("request_hash") or "") != request_hash:
            raise HTTPException(
                status_code=409,
                detail="client_msg_id was reused for a different message.",
            )
        return json.loads(prev["response_json"])

    db.expire_stale_sessions(ttl_hours=get_settings().intake.session_ttl_hours)

    sess = db.fetch_one("SELECT thread_id, status FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found. Start a new session first.")
    if sess["status"] == "expired":
        raise HTTPException(status_code=410, detail="Session expired. Please start a new intake.")

    set_trace_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    log_event("chat_received", request_id=request_id, thread_id=thread_id, message_len=len(message))

    try:
        t0 = time.time()
        _prev = db.get_session_state(thread_id)
        prev_phase = (_prev or {}).get("state", {}).get("current_phase") if _prev else None

        graph = request.app.state.graph
        output = graph.invoke({"messages": [{"role": "user", "text": message}]}, config)
        db.save_session_state(thread_id, _compact_snapshot(output))

        new_phase = output.get("current_phase")
        if prev_phase and new_phase and prev_phase != new_phase:
            log_event("phase_transition", thread_id=thread_id,
                      from_phase=prev_phase, to_phase=new_phase)

        job_id = None
        phase = output.get("current_phase")

        if phase == "report":
            import threading
            job_id = db.create_job(thread_id, "report")
            t = threading.Thread(target=_run_report_job, args=(graph, thread_id, job_id), daemon=True)
            t.start()

        duration_ms = int((time.time() - t0) * 1000)

        messages = output.get("messages") or []
        reply = messages[-1]["text"] if messages else "Thank you. Please continue."

        triage = output.get("triage") or {}
        if phase == "done":
            status = "done"
        elif phase == "handoff" or output.get("needs_emergency_review") or triage.get("emergency_flag"):
            status = "escalated"
        else:
            current = db.fetch_one("SELECT status FROM sessions WHERE thread_id=?", (thread_id,))
            current_status = (current or {}).get("status") or "active"
            status = current_status if current_status in ("done", "escalated", "expired") else "active"

        if status not in ("done", "escalated", "expired"):
            status = "active"

        db.set_session_status(thread_id, status)

        resp_obj = {"reply": reply, "status": status, "phase": phase}
        if job_id:
            resp_obj["job_id"] = job_id
            resp_obj["phase"] = "report_generating"

        db.persist_chat_turn(
            thread_id=thread_id,
            user_message=message,
            assistant_reply=reply,
            state_snapshot=_compact_snapshot(output),
            status=status,
            client_msg_id=client_msg_id,
            request_hash=request_hash,
            response_obj=resp_obj,
            job_id=job_id,
        )

        log_event("chat_done", request_id=request_id, thread_id=thread_id,
                  duration_ms=duration_ms, phase=phase, status=status)
        return resp_obj

    except Exception as e:
        log_event("chat_error", level="error", request_id=request_id,
                  thread_id=thread_id, error=str(e)[:400])
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@router.get("/report/{thread_id}")
@limiter.limit("30/minute")
def get_report(request: Request, thread_id: str, authorization: str = Header(default="")):
    require_session_token(thread_id, authorization)
    sess = db.fetch_one("SELECT thread_id FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    rep = db.get_latest_report(thread_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    rep_dict = dict(rep)
    rep_dict["pending_review"] = bool(rep_dict.get("pending_review", 0))
    return {"latest": rep_dict}


@router.post("/report/{thread_id}/retry")
@limiter.limit("5/hour")
def retry_report(
    request: Request,
    thread_id: str,
    _: None = Depends(require_clinician),
):
    sess = db.fetch_one("SELECT status FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")

    jobs = db.get_jobs_for_thread(thread_id)
    report_jobs = [j for j in jobs if j["kind"] == "report"]
    in_flight = [j for j in report_jobs if j["status"] in ("queued", "running")]
    if in_flight:
        raise HTTPException(
            status_code=409,
            detail=f"A report job is already {in_flight[0]['status']}. "
                   f"Poll /jobs/{in_flight[0]['job_id']} for status.",
        )

    MAX_ATTEMPTS = get_settings().intake.max_report_attempts
    if len(report_jobs) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum retry attempts ({MAX_ATTEMPTS}) reached for this thread.",
        )
    if not report_jobs or report_jobs[0]["status"] != "failed":
        raise HTTPException(
            status_code=409,
            detail="No failed report job to retry.",
        )

    import threading
    attempt = len(report_jobs) + 1
    job_id = db.create_job(thread_id, "report")
    graph = request.app.state.graph
    t = threading.Thread(target=_run_report_job, args=(graph, thread_id, job_id), daemon=True)
    t.start()
    log_event("report_retry_queued", thread_id=thread_id, job_id=job_id, attempt=attempt)
    return {"job_id": job_id, "attempt": attempt, "status": "queued"}


@router.get("/jobs/{job_id}")
@limiter.limit("120/minute")
def job_status(request: Request, job_id: str, authorization: str = Header(default="")):
    """
    Poll report generation progress.

    Requires the session token of the thread that owns the job — prevents
    a patient from enumerating other patients' job statuses by guessing UUIDs.
    The job record carries a thread_id; we verify the caller owns that thread
    before returning any data.
    """
    # Fetch job first so we have the thread_id for the ownership check.
    stale_count = db.mark_stale_jobs_failed(stale_minutes=10)
    if stale_count:
        log_event("stale_jobs_expired", level="warning", count=stale_count)

    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Ownership check: the caller must hold the session token for this thread.
    require_session_token(job["thread_id"], authorization)

    report_available = False
    if job["status"] == "done":
        report_available = db.get_latest_report(job["thread_id"]) is not None

    return {
        "job_id":           job["job_id"],
        "thread_id":        job["thread_id"],
        "kind":             job["kind"],
        "status":           job["status"],
        "error":            job["error"],
        "updated_at":       job["updated_at"],
        "report_available": report_available,
    }


