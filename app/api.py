from __future__ import annotations
import json
import uuid
import time
import hashlib
import logging
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

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Clinical Intake")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

def _compact_snapshot(output: dict) -> dict:
    return {
        "current_phase": output.get("current_phase"),
        "identity": output.get("identity"),
        "stored_identity": output.get("stored_identity"),
        "identity_status": output.get("identity_status"),
        "needs_identity_review": output.get("needs_identity_review"),
        "chief_complaint": output.get("chief_complaint"),
        "opqrst": output.get("opqrst"),
        "allergies": output.get("allergies"),
        "medications": output.get("medications"),
        "pmh": output.get("pmh"),
        "recent_results": output.get("recent_results"),
        "triage": output.get("triage"),
        "needs_emergency_review": output.get("needs_emergency_review"),
        "clinical_step": output.get("clinical_step"),
        "mode": output.get("mode"),
        "triage_attempts": output.get("triage_attempts"),
    }

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def _startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
    )
    db.init_schema()
    # Seed default emergency phrases on first boot if the table is empty.
    if not db.get_emergency_phrases():
        from .extract import DEFAULT_EMERGENCY_PHRASES
        db.seed_emergency_phrases(DEFAULT_EMERGENCY_PHRASES)
    if not settings.debug_mode:
        get_gemini().validate()

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
        "current_phase": "identity",
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

    reply = output["messages"][-1]["text"]
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
    phase = (state or {}).get("state", {}).get("current_phase", "identity")

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
        graph.invoke({"messages": []}, config)
        db.update_job(job_id, "done")
        log_event("report_job_done", thread_id=thread_id, job_id=job_id)
    except Exception as e:
        db.update_job(job_id, "failed", error=f"{type(e).__name__}: {e}")
        log_event("report_job_failed", level="error", thread_id=thread_id, job_id=job_id, error=str(e)[:400])


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
        output = graph.invoke({"messages": [{"role": "user", "text": message}]}, config)
        db.save_session_state(thread_id, _compact_snapshot(output))

        job_id = None
        phase = output.get("current_phase")

        if phase == "report":
            job_id = db.create_job(thread_id, "report")
            background_tasks.add_task(run_report_job, thread_id, job_id)

        duration_ms = int((time.time() - t0) * 1000)

        reply = output["messages"][-1]["text"]
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
        return {"reply": "Something went wrong. Please try again.", "status": "error"}


@app.get("/report/{thread_id}")
@limiter.limit("30/minute")
def get_report(request: Request, thread_id: str):
    rep = db.get_latest_report(thread_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    return {"latest": rep}


@app.get("/report/{thread_id}/fhir")
@limiter.limit("30/minute")
def get_fhir_report(request: Request, thread_id: str):
    """
    Return the FHIR R4 Bundle for the latest completed intake.

    The Bundle contains:
      - Patient           (identity)
      - Condition         (chief complaint + OPQRST note)
      - AllergyIntolerance (one per allergy)
      - MedicationStatement (one per medication)
      - Observation        (triage risk level)

    Content-Type is application/fhir+json per the FHIR R4 spec.
    """
    fhir_json = db.get_fhir_bundle(thread_id)
    if not fhir_json:
        raise HTTPException(
            status_code=404,
            detail="FHIR bundle not available. Complete the intake first.",
        )
    return JSONResponse(
        content=json.loads(fhir_json),
        media_type="application/fhir+json",
    )


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
    rep = db.get_latest_report(thread_id)
    esc = db.fetch_all(
        "SELECT esc_id, kind, resolved, nurse_note, created_at FROM escalations WHERE thread_id=? ORDER BY created_at DESC",
        (thread_id,),
    )
    state = db.get_session_state(thread_id)
    return {"thread_id": thread_id, "messages": msgs, "latest_report": rep, "escalations": esc, "state": state}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job["job_id"],
        "thread_id": job["thread_id"],
        "kind": job["kind"],
        "status": job["status"],
        "error": job["error"],
        "updated_at": job["updated_at"],
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

    # Re-seed mock EHR patients inline so this works without running
    # the seed script separately.
    from .settings import settings as _settings
    import sqlite3 as _sqlite3

    DEMO_PATIENTS = [
        {
            "patient_id": "demo-ava",
            "name": "Ava Johnson",
            "history": "Prior visit: Hypertension. Penicillin allergy.",
            "data_json": json.dumps({
                "identity": {"phone": "4125550199", "address": "100 Forbes Ave, Pittsburgh, PA"},
                "allergies": ["penicillin"],
                "medications": ["lisinopril 10mg daily"],
                "pmh": ["hypertension"],
                "recent_results": ["CBC normal (2025-11-10)"],
            }),
        },
        {
            "patient_id": "demo-marcus",
            "name": "Marcus Thorne",
            "history": "Prior cardiac stent placement in 2023.",
            "data_json": json.dumps({
                "identity": {"phone": "5550388844", "address": "12 Market St, Pittsburgh, PA"},
                "allergies": [],
                "medications": ["atorvastatin 40mg nightly"],
                "pmh": ["coronary artery disease", "cardiac stent (2023)"],
                "recent_results": [],
            }),
        },
        {
            "patient_id": "demo-nina",
            "name": "Nina Shah",
            "history": "Prior visit: Anxiety. No known drug allergies.",
            "data_json": json.dumps({
                "identity": {"phone": "5557772222", "address": "44 Walnut St, Chicago, IL"},
                "allergies": [],
                "medications": [],
                "pmh": ["anxiety"],
                "recent_results": [],
            }),
        },
    ]

    with _sqlite3.connect(_settings.app_db_path, timeout=10.0) as c:
        c.execute("DELETE FROM mock_ehr WHERE patient_id LIKE 'demo-%'")
        for p in DEMO_PATIENTS:
            c.execute(
                "INSERT INTO mock_ehr (patient_id, name, history, data_json) VALUES (?,?,?,?)",
                (p["patient_id"], p["name"], p["history"], p["data_json"]),
            )
        c.commit()

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
