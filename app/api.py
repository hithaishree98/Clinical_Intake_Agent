from __future__ import annotations
import json
import uuid
import time
import os
import hashlib
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .graph import build_graph
from .logging_utils import log_event
from .llm import get_gemini
from . import sqlite_db as db

app = FastAPI(title="Clinical Intake (Clean Repo)")

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

@app.on_event("startup")
def _startup():
    db.init_schema()
    get_gemini()

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (adjust if needed)
STATIC_DIR = BASE_DIR / "static"
CLINICIAN_TOKEN = os.getenv("CLINICIAN_TOKEN", "dev-token")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.post("/start")
def start_session(mode: str = Form("clinic")):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "current_phase": "identity",
        "mode": "ed" if (mode or "").strip().lower() == "ed" else "clinic",
        "triage_attempts": 0,
        "identity": {"name":"", "phone":"", "address":""},
        "stored_identity": None,
        "identity_attempts": 0,
        "identity_status": "unverified",
        "needs_identity_review": False,

        "chief_complaint": "",
        "opqrst": {"onset":"","provocation":"","quality":"","radiation":"","severity":"","timing":""},
        "subjective_complete": False,

        "clinical_step": "allergies",
        "allergies": [],
        "medications": [],
        "pmh": [],
        "recent_results": [],
        "clinical_complete": False,

        "triage": {"emergency_flag": False, "risk_level": "low", "visit_type": "routine", "red_flags": [], "confidence":"low", "rationale":""},
        "needs_emergency_review": False,

        "messages": [],
    }

    db.create_session(thread_id)
    log_event("api_start", thread_id=thread_id)

    t0 = time.time()
    output = graph.invoke(initial_state, config)
    db.save_session_state(thread_id, _compact_snapshot(output))
    if output.get("current_phase") == "report":
        output = graph.invoke({"messages": []}, config)

    log_event("api_start_done", thread_id=thread_id, duration_ms=int((time.time()-t0)*1000))

    reply = output["messages"][-1]["text"]
    db.save_message(thread_id, "assistant", reply)
    return {
    "thread_id": thread_id,
    "reply": reply,
    "phase": output.get("current_phase") or "identity",
    "status": "active",
}


@app.post("/chat")
def chat( background_tasks: BackgroundTasks,
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
        # If the same client_msg_id is reused for a different message, that's a bug
        if (prev.get("request_hash") or "") != request_hash:
            raise HTTPException(
                status_code=409,
                detail="client_msg_id was reused for a different message (idempotency key conflict).",
            )
        return json.loads(prev["response_json"])

    sess = db.fetch_one("SELECT thread_id, status FROM sessions WHERE thread_id=?", (thread_id,))
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown thread_id. Start a session first.")

    config = {"configurable": {"thread_id": thread_id}}
    request_id = str(uuid.uuid4())
    log_event("api_chat_start", request_id=request_id, thread_id=thread_id, message_len=len(message))


    try:
        t0 = time.time()
        output = graph.invoke({"messages": [{"role":"user","text": message}]}, config)
        db.save_session_state(thread_id, _compact_snapshot(output))

        job_id = None
        phase = output.get("current_phase")

        # If workflow reaches report phase, schedule it async
        if phase == "report":
            job_id = db.create_job(thread_id, "report")
            background_tasks.add_task(run_report_job, thread_id, job_id)
    # do not invoke report synchronously anymore
        duration_ms = int((time.time()-t0)*1000)

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

        log_event("api_chat_done", request_id=request_id, thread_id=thread_id, duration_ms=duration_ms, phase=output.get("current_phase"), status=status)
        return resp_obj

    except Exception as e:
        log_event("api_error", level="error", request_id=request_id, thread_id=thread_id, error=str(e)[:400])
        return {"reply": "System error: clinical state could not be resumed.", "status": "error"}

@app.get("/report/{thread_id}")
def get_report(thread_id: str):
    rep = db.get_latest_report(thread_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    return {"latest": rep}

def require_clinician(x_clinician_token: str = Header(default="")):
    if x_clinician_token != CLINICIAN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
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
    state = db.get_session_state(thread_id)  # added below

    return {"thread_id": thread_id, "messages": msgs, "latest_report": rep, "escalations": esc, "state": state}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "thread_id": job["thread_id"],
        "kind": job["kind"],
        "status": job["status"],
        "error": job["error"],
        "updated_at": job["updated_at"],
    }

def run_report_job(thread_id: str, job_id: str):
    try:
        db.update_job(job_id, "running")
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"messages": []}, config)  # triggers report_node
        db.update_job(job_id, "done")
    except Exception as e:
        db.update_job(job_id, "failed", error=f"{type(e).__name__}: {e}")