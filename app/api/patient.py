"""
patient.py — Patient-facing API endpoints.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request

from .. import sqlite_db as db
from ..extract import check_prompt_injection
from ..llm import is_llm_available
from ..logging_utils import log_event, set_trace_id
from ..settings import get_settings
from .deps import limiter, require_session_token, require_clinician

router = APIRouter()


def _issue_session_token() -> str:
    return secrets.token_hex(32)


# "messages" stored separately in messages table; exclude to prevent double-storage.
_SNAPSHOT_EXCLUDE: frozenset[str] = frozenset({"messages"})


def _compact_snapshot(output: dict) -> dict:
    """Persist all state fields except those in _SNAPSHOT_EXCLUDE."""
    return {k: v for k, v in output.items() if k not in _SNAPSHOT_EXCLUDE}


def _quick_replies_for_state(output: dict) -> list[dict]:
    """Return quick-reply buttons for binary-gate phases (consent, identity_review, confirm)."""
    phase = output.get("current_phase")
    target = output.get("validation_target_phase")

    if phase == "consent":
        return [
            {"label": "Yes, I consent", "payload": "yes"},
            {"label": "No, decline",    "payload": "no"},
        ]

    if phase == "identity_review":
        # When we have a stored EHR record the choice is "keep what's on
        # file" vs. "use what I just provided" — both routed through the
        # same yes/no codepath inside identity_review_node.
        if output.get("stored_identity"):
            return [
                {"label": "Keep on file", "payload": "yes"},
                {"label": "Update",       "payload": "no"},
            ]
        return [
            {"label": "Yes, that's right", "payload": "yes"},
            {"label": "Fix it",            "payload": "no"},
        ]

    if phase == "confirm" or (phase == "validate" and target == "confirm"):
        return [
            {"label": "Confirm",        "payload": "confirm"},
            {"label": "Make a change",  "payload": "go back"},
        ]

    return []


def _build_resume_context(phase: str, state_data: dict) -> str:
    """Build a one-sentence context summary shown when a patient resumes a session."""
    _phase_context = {
        "consent":          "You were just getting started — we hadn't yet collected your information.",
        "identity":         "You were sharing your personal details.",
        "identity_review":  "You were reviewing your personal details.",
        "subjective":       "You were describing what brought you in.",
        "clinical_history": "You were sharing your health background.",
        "confirm":          "You were reviewing your intake summary.",
        "report":           "Your intake was being finalised.",
        "handoff":          "Your intake is with the care team.",
        "done":             "Your intake is complete.",
    }
    ctx = _phase_context.get(phase, "You were in the middle of your intake.")
    cc = (state_data.get("chief_complaint") or "").strip()
    if cc and phase not in ("consent", "identity", "identity_review"):
        ctx += f" Your main concern was noted as '{cc}'."
    return f"Welcome back! {ctx} Let's continue where we left off."


def _run_report_inline(graph, thread_id: str, config: dict) -> tuple[dict, str | None]:
    """
    Finalise the LangGraph state machine synchronously, returning
    (output, final_reply).  Called directly in the /chat handler when
    the graph reaches the 'report' phase — no thread pool, no job table.
    """
    output = graph.invoke({"messages": []}, config)
    messages = output.get("messages") or []
    final_reply: str | None = None
    if messages and messages[-1].get("role") == "assistant":
        final_reply = (messages[-1].get("text") or "").strip() or None
    db.persist_report_turn(
        thread_id=thread_id,
        state_snapshot=_compact_snapshot(output),
        assistant_reply=final_reply,
    )
    return output, final_reply


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start")
@limiter.limit("10/hour")
def start_session(request: Request, mode: str = Form("clinic"), clinic_id: str = Form("default")):
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
        "allergies": None,
        "medications": None,
        "pmh": None,
        "recent_results": None,
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
    db.create_session(thread_id, session_token=session_token, clinic_id=clinic_id)
    log_event("session_started", thread_id=thread_id, mode=initial_state["mode"], clinic_id=clinic_id)

    graph = request.app.state.graph
    t0 = time.time()
    output = graph.invoke(initial_state, config)
    db.save_session_state(thread_id, _compact_snapshot(output))
    log_event("session_ready", thread_id=thread_id, duration_ms=int((time.time() - t0) * 1000))

    messages = output.get("messages") or []
    reply = messages[-1]["text"] if messages else "Welcome. Let's begin your intake."
    db.save_message(thread_id, "assistant", reply)
    response = {
        "thread_id":     thread_id,
        "session_token": session_token,
        "reply":         reply,
        "phase": output.get("current_phase") or initial_state["current_phase"],
        "status":        "active",
    }
    quick_replies = _quick_replies_for_state(output)
    if quick_replies:
        response["quick_replies"] = quick_replies
    return response


@router.get("/resume/{thread_id}")
@limiter.limit("30/minute")
def resume_session(request: Request, thread_id: str, authorization: str = Header(default="")):
    require_session_token(thread_id, authorization)
    sess = db.get_session_row(thread_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess["status"] in ("done", "escalated", "expired"):
        raise HTTPException(status_code=410, detail=f"Session already {sess['status']}.")

    state_row  = db.get_session_state(thread_id)
    state_data = ((state_row or {}).get("state") or {})
    phase      = state_data.get("current_phase", "identity")
    resume_msg = _build_resume_context(phase, state_data)
    response = {
        "thread_id": thread_id,
        "status":    sess["status"],
        "phase":     phase,
        "reply":     resume_msg,
    }
    # returning patient lands on a binary-gate phase (consent / identity_review / confirm) sees the same
    # one-click choices they would on a fresh /chat turn.  state_data is the
    # snapshot persisted at the end of the previous turn; it carries the
    # `validation_target_phase` and `stored_identity` fields _quick_replies_for_state
    # uses, so no re-derivation is needed.
    quick_replies = _quick_replies_for_state(state_data)
    if quick_replies:
        response["quick_replies"] = quick_replies
    return response


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

    # TTL expiry runs in the hourly background loop (see app.main).  The old
    # 5%-per-chat-turn sweep contended with the chat-write transaction and
    # was a flaky-test source — non-deterministic UPDATEs between assertion
    # rounds.  Hourly is plenty for a 4-hour TTL.
    sess = db.get_session_row(thread_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found. Start a new session first.")
    if sess["status"] == "expired":
        raise HTTPException(status_code=410, detail="Session expired. Please start a new intake.")
    # A "done" session has already produced a clinician note — further chat
    # turns would route to END and return a stale default reply.  Surface the
    # terminal state to the caller instead so the UI can prompt for a new
    # intake.  Mirrors the /resume behaviour for already-completed sessions.
    if sess["status"] == "done":
        raise HTTPException(status_code=410, detail="Session already complete. Please start a new intake.")

    set_trace_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    # request_id is set by CorrelationMiddleware and propagates via ContextVar —
    # log_event picks it up automatically, so we don't generate or pass one here.
    log_event("chat_received", thread_id=thread_id, message_len=len(message))

    try:
        t0 = time.time()
        _prev = db.get_session_state(thread_id)
        prev_phase = (_prev or {}).get("state", {}).get("current_phase") if _prev else None

        # ── Guard 1: Circuit breaker — LLM API temporarily unavailable ──────
        if not is_llm_available():
            log_event("chat_blocked_circuit_open", level="warning", thread_id=thread_id)
            return {
                "reply": "We're experiencing a brief technical issue. "
                         "Your progress is saved — please try again in a couple of minutes.",
                "status": "active",
                "phase":  prev_phase or "unknown",
            }

        # ── Guard 2: Max session turns — prevents loops and cost runaway ─────
        turn_row = db.fetch_one(
            "SELECT COUNT(*) AS n FROM messages WHERE thread_id=? AND role='user'",
            (thread_id,),
        ) or {}
        if (turn_row.get("n") or 0) >= get_settings().intake.max_session_turns:
            log_event("chat_max_turns_reached", level="warning", thread_id=thread_id)
            db.set_session_status(thread_id, "done")
            return {
                "reply": "Your session has reached its maximum length. "
                         "Please speak with the front desk to complete your intake — "
                         "your progress has been saved.",
                "status": "done",
                "phase":  "done",
            }

        # ── Guard 2b: Session cost cap — prevents runaway LLM spend ─────────
        # Queries llm_usage (written by _track_llm_failure after every LLM call)
        # so the cap is enforced even across retries and repair calls.
        _pricing = get_settings().intake
        cost_row = db.fetch_one(
            "SELECT COALESCE("
            "  SUM(CAST(input_tokens AS REAL)/1000000.0*?"
            "    + CAST(output_tokens AS REAL)/1000000.0*?), 0.0"
            ") AS cost FROM llm_usage "
            "WHERE thread_id=? AND provider='gemini'",
            (_pricing.gemini_input_cost_per_million,
             _pricing.gemini_output_cost_per_million,
             thread_id),
        ) or {}
        session_cost = float(cost_row.get("cost") or 0.0)
        if session_cost > get_settings().intake.max_cost_usd_per_session:
            log_event("session_cost_cap_enforced", level="warning",
                      thread_id=thread_id, session_cost_usd=session_cost,
                      cap_usd=get_settings().intake.max_cost_usd_per_session)
            db.set_session_status(thread_id, "done")
            return {
                "reply": "Your session has reached its limit. "
                         "Please speak with the front desk to complete your intake — "
                         "your progress has been saved.",
                "status": "done",
                "phase":  "done",
            }

        graph = request.app.state.graph
        output = graph.invoke({"messages": [{"role": "user", "text": message}]}, config)
        db.save_session_state(thread_id, _compact_snapshot(output))

        new_phase = output.get("current_phase")
        if prev_phase and new_phase and prev_phase != new_phase:
            log_event("phase_transition", thread_id=thread_id,
                      from_phase=prev_phase, to_phase=new_phase)

        phase = output.get("current_phase")

        # When the graph reaches the report phase, run report_node synchronously
        # in the same request — it completes in ~300ms and returns the final note.
        if phase == "report":
            output, _ = _run_report_inline(graph, thread_id, config)
            phase = output.get("current_phase")

        duration_ms = int((time.time() - t0) * 1000)

        messages = output.get("messages") or []
        reply = messages[-1]["text"] if messages else "Thank you. Please continue."

        triage = output.get("triage") or {}
        if phase == "done":
            status = "done"
        elif phase == "handoff" or output.get("needs_emergency_review") or triage.get("emergency_flag"):
            status = "escalated"
        else:
            current = db.get_session_row(thread_id)
            current_status = (current or {}).get("status") or "active"
            status = current_status if current_status in ("done", "escalated", "expired") else "active"

        if status not in ("done", "escalated", "expired"):
            status = "active"

        db.set_session_status(thread_id, status)

        resp_obj = {"reply": reply, "status": status, "phase": phase}
        if output.get("validation_errors"):
            resp_obj["hint"] = "Gathering a bit more detail before moving on."
        quick_replies = _quick_replies_for_state(output)
        if quick_replies:
            resp_obj["quick_replies"] = quick_replies

        db.persist_chat_turn(
            thread_id=thread_id,
            user_message=message,
            assistant_reply=reply,
            state_snapshot=_compact_snapshot(output),
            status=status,
            client_msg_id=client_msg_id,
            request_hash=request_hash,
            response_obj=resp_obj,
            job_id=None,
        )

        log_event("chat_done", thread_id=thread_id,
                  duration_ms=duration_ms, phase=phase, status=status)
        return resp_obj

    except Exception as e:
        log_event("chat_error", level="error",
                  thread_id=thread_id, error=str(e)[:400])
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")




