from __future__ import annotations
import json
import uuid
import time
import hashlib
import logging
from contextlib import asynccontextmanager
import jwt
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .graph import build_graph
from .logging_utils import log_event
from .llm import get_gemini
from . import sqlite_db as db
from .settings import settings
from .extract import check_prompt_injection

limiter = Limiter(key_func=get_remote_address)

graph = None  # initialised in lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db.init_schema()
    if not db.get_emergency_phrases():
        from .extract import DEFAULT_EMERGENCY_PHRASES
        db.seed_emergency_phrases(DEFAULT_EMERGENCY_PHRASES)
    if not settings.debug_mode:
        get_gemini().validate()
    graph = build_graph()
    yield


app = FastAPI(title="Clinical Intake", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _compact_snapshot(output: dict) -> dict:
    return {
        "current_phase":        output.get("current_phase"),
        "identity":             output.get("identity"),
        "stored_identity":      output.get("stored_identity"),
        "identity_status":      output.get("identity_status"),
        "needs_identity_review":output.get("needs_identity_review"),
        "consent_given":        output.get("consent_given"),
        "chief_complaint":      output.get("chief_complaint"),
        "opqrst":               output.get("opqrst"),
        "allergies":            output.get("allergies"),
        "medications":          output.get("medications"),
        "pmh":                  output.get("pmh"),
        "recent_results":       output.get("recent_results"),
        "triage":               output.get("triage"),
        "needs_emergency_review":output.get("needs_emergency_review"),
        "clinical_step":        output.get("clinical_step"),
        "mode":                 output.get("mode"),
        "triage_attempts":      output.get("triage_attempts"),
        # Agentic fields
        "intake_classification":      output.get("intake_classification"),
        "classification_confidence":  output.get("classification_confidence"),
        "extraction_quality_score":   output.get("extraction_quality_score"),
        "extraction_retry_count":     output.get("extraction_retry_count"),
        "validation_errors":          output.get("validation_errors"),
        "validation_target_phase":    output.get("validation_target_phase"),
        # Safety fields
        "crisis_detected":        output.get("crisis_detected"),
        "human_review_required":  output.get("human_review_required"),
        "safety_score":           output.get("safety_score"),
        "extraction_confidence":  output.get("extraction_confidence"),
        # Failure tracking fields
        "last_failed_phase":      output.get("last_failed_phase"),
        "last_failure_reason":    output.get("last_failure_reason"),
    }

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.post("/start")
@limiter.limit("10/hour")
def start_session(request: Request, mode: str = Form("clinic")):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,

        "current_phase": "consent" if settings.require_consent else "identity",
        "consent_given": not settings.require_consent,  # pre-granted when consent gate is off
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

        "triage": {"emergency_flag": False, "risk_level": "low", "visit_type": "routine", "red_flags": [], "confidence": "low", "rationale": ""},
        "needs_emergency_review": False,

        # Agentic fields
        "intake_classification": None,
        "classification_confidence": None,
        "extraction_quality_score": None,
        "extraction_retry_count": 0,
        "validation_errors": [],
        "validation_target_phase": None,

        # Safety fields
        "crisis_detected": False,
        "human_review_required": False,
        "human_review_reasons": [],
        "safety_score": None,
        "extraction_confidence": None,

        # Failure tracking fields
        "last_failed_phase": None,
        "last_failure_reason": None,

        "messages": [],
    }

    db.create_session(thread_id)
    log_event("session_started", thread_id=thread_id, mode=initial_state["mode"])

    t0 = time.time()
    output = graph.invoke(initial_state, config)
    db.save_session_state(thread_id, _compact_snapshot(output))
    if output.get("current_phase") == "report":
        output = graph.invoke({"messages": []}, config)

    log_event("session_ready", thread_id=thread_id, duration_ms=int((time.time() - t0) * 1000))

    messages = output.get("messages") or []
    reply = messages[-1]["text"] if messages else "Welcome. Let's begin your intake."
    db.save_message(thread_id, "assistant", reply)
    return {
        "thread_id": thread_id,
        "reply": reply,
        "phase": output.get("current_phase") or "identity",
        "status": "active",
    }


@app.get("/resume/{thread_id}")
def resume_session(thread_id: str):
    sess = db.fetch_one(
        "SELECT thread_id, status FROM sessions WHERE thread_id=?",
        (thread_id,)
    )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess["status"] in ("done", "escalated"):
        raise HTTPException(status_code=410, detail=f"Session already {sess['status']}.")

    # Last assistant message tells the frontend what to re-display
    last_msg = db.fetch_one(
        "SELECT text FROM messages WHERE thread_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
        (thread_id,)
    )
    state = db.get_session_state(thread_id)
    phase = ((state or {}).get("state") or {}).get("current_phase", "identity")

    return {
        "thread_id": thread_id,
        "status": sess["status"],
        "phase": phase,
        "reply": last_msg["text"] if last_msg else "Welcome back. Let's continue your intake.",
    }


def run_report_job(thread_id: str, job_id: str):
    try:
        db.update_job(job_id, "running")
        config = {"configurable": {"thread_id": thread_id}}
        output = graph.invoke({"messages": []}, config)

        # report_node routes to "handoff" (not "done") when preflight safety check blocks.
        # Treat that as a failed job so the frontend can surface the blockage clearly.
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


def require_clinician(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token.")
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


@app.post("/report/{thread_id}/retry")
@limiter.limit("5/hour")
def retry_report(
    request: Request,
    background_tasks: BackgroundTasks,
    thread_id: str,
    _: None = Depends(require_clinician),
):
    """
    Retry report generation for a thread whose last report job failed.

    Rules:
      - Requires clinician token (prevents patient-side abuse).
      - At most 3 report jobs per thread (including the original).  If the
        original job failed that counts as attempt 1, so clinicians get 2 retries.
      - The most recent report job must be in 'failed' status.

    Returns the new job_id and attempt number so the clinician can poll /jobs/{job_id}.
    """
    sess = db.fetch_one("SELECT status FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")

    jobs = db.get_jobs_for_thread(thread_id)
    report_jobs = [j for j in jobs if j["kind"] == "report"]

    # Prevent a duplicate in-flight job
    in_flight = [j for j in report_jobs if j["status"] in ("queued", "running")]
    if in_flight:
        raise HTTPException(
            status_code=409,
            detail=f"A report job is already {in_flight[0]['status']} for this thread. "
                   f"Poll /jobs/{in_flight[0]['job_id']} for status.",
        )

    MAX_ATTEMPTS = 3
    if len(report_jobs) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum retry attempts ({MAX_ATTEMPTS}) reached for this thread.",
        )

    if not report_jobs or report_jobs[0]["status"] != "failed":
        raise HTTPException(
            status_code=409,
            detail="No failed report job to retry. "
                   "The most recent report job must be in 'failed' status.",
        )

    attempt = len(report_jobs) + 1
    job_id = db.create_job(thread_id, "report")
    background_tasks.add_task(run_report_job, thread_id, job_id)
    log_event("report_retry_queued", thread_id=thread_id,
              job_id=job_id, attempt=attempt)
    return {"job_id": job_id, "attempt": attempt, "status": "queued"}


@app.post("/chat")
@limiter.limit("60/minute")
def chat(request: Request, background_tasks: BackgroundTasks,
         thread_id: str = Form(...),
         message: str = Form(...),
         client_msg_id: str = Form(...)):

    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 1200:
        raise HTTPException(status_code=400, detail="Message too long (max 1200 chars).")
    if len(client_msg_id) > 128:
        raise HTTPException(status_code=400, detail="client_msg_id too long (max 128 chars).")
    if check_prompt_injection(message):
        return {"reply": "I can only collect intake information for your visit. If you have a question for your care team, they'll be happy to help when you arrive.", "status": "active", "phase": "unknown"}

    request_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()

    prev = db.get_idempotent_response(thread_id, client_msg_id)
    if prev:
        if (prev.get("request_hash") or "") != request_hash:
            raise HTTPException(
                status_code=409,
                detail="client_msg_id was reused for a different message.",
            )
        return json.loads(prev["response_json"])

    sess = db.fetch_one("SELECT thread_id, status FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found. Start a new session first.")

    config = {"configurable": {"thread_id": thread_id}}
    request_id = str(uuid.uuid4())
    log_event("chat_received", request_id=request_id, thread_id=thread_id, message_len=len(message))

    try:
        t0 = time.time()
        _prev = db.get_session_state(thread_id)
        prev_phase = (_prev or {}).get("state", {}).get("current_phase") if _prev else None

        output = graph.invoke({"messages": [{"role": "user", "text": message}]}, config)
        db.save_session_state(thread_id, _compact_snapshot(output))

        new_phase = output.get("current_phase")
        if prev_phase and new_phase and prev_phase != new_phase:
            log_event("phase_transition", thread_id=thread_id,
                      from_phase=prev_phase, to_phase=new_phase)

        job_id = None
        phase = output.get("current_phase")

        if phase == "report":
            job_id = db.create_job(thread_id, "report")
            background_tasks.add_task(run_report_job, thread_id, job_id)

        duration_ms = int((time.time() - t0) * 1000)

        messages = output.get("messages") or []
        reply = messages[-1]["text"] if messages else "Thank you. Please continue."
        db.save_message(thread_id, "user", message)
        db.save_message(thread_id, "assistant", reply)

        triage = output.get("triage") or {}
        status = "escalated" if triage.get("emergency_flag") else "active"
        db.set_session_status(thread_id, status)

        resp_obj = {"reply": reply, "status": status, "phase": phase}
        if job_id:
            resp_obj["job_id"] = job_id
            resp_obj["phase"] = "report_generating"
        db.save_idempotent_response(thread_id, client_msg_id, request_hash, resp_obj)

        log_event("chat_done", request_id=request_id, thread_id=thread_id, duration_ms=duration_ms, phase=phase, status=status)
        return resp_obj

    except Exception as e:
        log_event("chat_error", level="error", request_id=request_id, thread_id=thread_id, error=str(e)[:400])
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.get("/report/{thread_id}")
@limiter.limit("30/minute")
def get_report(request: Request, thread_id: str):
    # Confirm the session exists before returning any data — prevents probing
    # for reports belonging to unknown thread IDs.
    sess = db.fetch_one("SELECT thread_id FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    rep = db.get_latest_report(thread_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    rep_dict = dict(rep)
    # Normalise to bool so JSON consumers don't receive SQLite's integer 0/1.
    rep_dict["pending_review"] = bool(rep_dict.get("pending_review", 0))
    return {"latest": rep_dict}


@app.get("/report/{thread_id}/fhir")
@limiter.limit("30/minute")
def get_fhir_report(request: Request, thread_id: str, _: None = Depends(require_clinician)):
    """
    Return the FHIR R4 Bundle for the latest completed intake.

    The Bundle contains:
      - Patient           (identity)
      - Condition         (chief complaint + OPQRST note)
      - AllergyIntolerance (one per allergy)
      - MedicationStatement (one per medication)
      - Observation        (triage risk level)

    Content-Type is application/fhir+json per the FHIR R4 spec.

    If the report was saved with pending_review=True (soft safety signals were
    present), the response includes an X-Pending-Review: true header so
    clinician tooling can surface a review banner without a second request.
    """
    rep = db.get_latest_report(thread_id)
    if not rep or not rep.get("fhir_bundle"):
        raise HTTPException(
            status_code=404,
            detail="FHIR bundle not available. Complete the intake first.",
        )
    response = JSONResponse(
        content=json.loads(rep["fhir_bundle"]),
        media_type="application/fhir+json",
    )
    if rep.get("pending_review"):
        response.headers["X-Pending-Review"] = "true"
    return response


@app.post("/clinician/token")
def clinician_token(password: str = Form(...)):
    if password != settings.clinician_password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = jwt.encode(
        {"sub": "clinician", "exp": time.time() + 86400},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer"}


@app.get("/clinician/pending")
def clinician_pending(_: None = Depends(require_clinician)):
    return JSONResponse(content=db.list_pending_escalations())


@app.post("/clinician/resolve")
def clinician_resolve(thread_id: str = Form(...), esc_id: str = Form(...), nurse_note: str = Form("Resolved"), _: None = Depends(require_clinician)):
    db.resolve_escalation(thread_id, esc_id, nurse_note)
    db.set_session_status(thread_id, "active")
    return {"ok": True}


@app.get("/clinician/case/{thread_id}")
def clinician_case(thread_id: str, _: None = Depends(require_clinician)):
    msgs = db.fetch_all(
        "SELECT role, text, created_at FROM messages WHERE thread_id=? ORDER BY id ASC",
        (thread_id,),
    )
    rep   = db.get_latest_report(thread_id)
    state = db.get_session_state(thread_id)

    # Include full reason trail from escalation payload so clinicians see *why*
    esc_rows = db.fetch_all(
        "SELECT esc_id, kind, resolved, nurse_note, payload_json, created_at "
        "FROM escalations WHERE thread_id=? ORDER BY created_at DESC",
        (thread_id,),
    )
    escalations = []
    for row in esc_rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        escalations.append({
            "esc_id":      row["esc_id"],
            "kind":        row["kind"],
            "severity":    payload.get("severity", "unknown"),
            "resolved":    row["resolved"],
            "nurse_note":  row["nurse_note"],
            "created_at":  row["created_at"],
            # Clinician-facing reason trail
            "reasons":          payload.get("reasons", []),
            "safety_score":     payload.get("safety_score"),
            "review_required":  payload.get("review_required"),
            "triggered_at_phase": payload.get("triggered_at_phase"),
            "context":          payload.get("context", {}),
        })

    return {
        "thread_id":    thread_id,
        "messages":     msgs,
        "latest_report": rep,
        "escalations":  escalations,
        "state":        state,
        # Safety summary at the case level
        "safety_summary": {
            "human_review_required": ((state or {}).get("state") or {}).get("human_review_required", False),
            "safety_score":          ((state or {}).get("state") or {}).get("safety_score"),
            "crisis_detected":       ((state or {}).get("state") or {}).get("crisis_detected", False),
        },
    }


@app.get("/clinician/webhooks")
@limiter.limit("30/minute")
def list_webhook_deliveries(
    request: Request,
    thread_id: str | None = None,
    limit: int = 50,
    _: None = Depends(require_clinician),
):
    """
    List outbound webhook delivery records, optionally filtered by thread.

    Each record exposes:
      delivery_id      UUID of the delivery
      event_type       slack_emergency | slack_crisis | slack_intake_complete | fhir_completion
      status           pending | success | failed | exhausted | duplicate_skipped
      attempts         number of HTTP attempts made so far
      last_http_status HTTP response code from the most recent attempt
      last_error       Error message if the last attempt failed
      next_retry_at    ISO timestamp for the next scheduled retry (null if terminal)
      payload_hash     SHA-256 of the payload body (idempotency key)
      created_at       When the first dispatch was attempted

    Use this to confirm FHIR bundles reached the EHR, investigate failed
    deliveries, and audit which events triggered retries.
    """
    rows = db.get_webhook_deliveries(thread_id=thread_id, limit=min(limit, 200))
    return {
        "count": len(rows),
        "deliveries": [dict(r) for r in rows],
    }


@app.get("/jobs/{job_id}")
@limiter.limit("120/minute")
def job_status(request: Request, job_id: str):
    # Auto-expire any running jobs that have been stuck for more than 10 minutes
    # before we return status, so clients always see an accurate terminal state.
    stale_count = db.mark_stale_jobs_failed(stale_minutes=10)
    if stale_count:
        log_event("stale_jobs_expired", level="warning", count=stale_count)

    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Include a flag so the frontend can skip a separate report fetch when the
    # job is done but no report was persisted (e.g. preflight blocked it).
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


@app.get("/analytics")
def analytics(_: None = Depends(require_clinician)):
    """Operational metrics for the last 7 days. Requires clinician token."""
    return db.get_analytics()



@app.post("/demo/reset")
def demo_reset(_: None = Depends(require_clinician)):
    """
    Wipe all session data and re-seed mock EHR patients.
    Use before demos to get back to a clean known state.
    Requires clinician token so it can't be hit accidentally.
    """
    db.reset_demo_data()
    db.seed_demo_patients()
    log_event("demo_reset", msg="Demo data wiped and re-seeded")
    return {"ok": True, "message": "Demo data reset. 3 mock patients re-seeded."}



@app.get("/admin/emergency-phrases")
def list_emergency_phrases(_: None = Depends(require_clinician)):
    """List all active emergency escalation phrases."""
    phrases = db.get_emergency_phrases()
    # Fall back to defaults if DB is empty so the response is always useful.
    if not phrases:
        from .extract import DEFAULT_EMERGENCY_PHRASES
        phrases = DEFAULT_EMERGENCY_PHRASES
    return {"phrases": phrases, "count": len(phrases)}


@app.post("/admin/emergency-phrases")
def add_emergency_phrase(phrase: str = Form(...), _: None = Depends(require_clinician)):
    """Add a new emergency escalation phrase. Takes effect immediately."""
    phrase = phrase.strip().lower()
    if not phrase:
        raise HTTPException(status_code=400, detail="Phrase cannot be empty.")
    # Seed defaults first if table is empty, so we don't lose the built-ins.
    if not db.get_emergency_phrases():
        from .extract import DEFAULT_EMERGENCY_PHRASES
        db.seed_emergency_phrases(DEFAULT_EMERGENCY_PHRASES)
    db.add_emergency_phrase(phrase)
    log_event("emergency_phrase_added", phrase=phrase)
    return {"ok": True, "phrase": phrase}


@app.delete("/admin/emergency-phrases")
def delete_emergency_phrase(phrase: str = Form(...), _: None = Depends(require_clinician)):
    """Remove an emergency escalation phrase. Takes effect immediately."""
    phrase = phrase.strip().lower()
    deleted = db.delete_emergency_phrase(phrase)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Phrase not found: {phrase}")
    log_event("emergency_phrase_deleted", phrase=phrase)
    return {"ok": True, "phrase": phrase}