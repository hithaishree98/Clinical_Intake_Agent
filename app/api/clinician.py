"""
clinician.py — Clinician-gated endpoints.

All routes here require a valid short-lived JWT (POST /clinician/token).
Patients cannot reach these endpoints even if they know the URLs.
"""
from __future__ import annotations

import json
import time

import jwt
from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import sqlite_db as db
from ..logging_utils import log_event
from ..settings import get_settings
from .deps import limiter, require_clinician

router = APIRouter(prefix="/clinician")


@router.post("/token")
@limiter.limit("5/minute")
def clinician_token(request: Request, password: str = Form(...)):
    settings = get_settings()
    if password != settings.clinician_password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = jwt.encode(
        {"sub": "clinician", "exp": time.time() + 86400},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer"}


@router.get("/pending")
def clinician_pending(_: None = Depends(require_clinician)):
    return JSONResponse(content=db.list_pending_escalations())


@router.post("/resolve")
def clinician_resolve(
    thread_id: str = Form(...),
    esc_id: str = Form(...),
    nurse_note: str = Form("Resolved"),
    _: None = Depends(require_clinician),
):
    db.resolve_escalation(thread_id, esc_id, nurse_note)
    db.set_session_status(thread_id, "active")
    return {"ok": True}


@router.get("/case/{thread_id}")
def clinician_case(thread_id: str, _: None = Depends(require_clinician)):
    msgs = db.fetch_all(
        "SELECT role, text, created_at FROM messages WHERE thread_id=? ORDER BY id ASC",
        (thread_id,),
    )
    rep   = db.get_latest_report(thread_id)
    state = db.get_session_state(thread_id)

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
            "esc_id":              row["esc_id"],
            "kind":                row["kind"],
            "severity":            payload.get("severity", "unknown"),
            "resolved":            row["resolved"],
            "nurse_note":          row["nurse_note"],
            "created_at":          row["created_at"],
            "reasons":             payload.get("reasons", []),
            "safety_score":        payload.get("safety_score"),
            "review_required":     payload.get("review_required"),
            "triggered_at_phase":  payload.get("triggered_at_phase"),
            "context":             payload.get("context", {}),
        })

    return {
        "thread_id":    thread_id,
        "messages":     msgs,
        "latest_report": rep,
        "escalations":  escalations,
        "state":        state,
        "safety_summary": {
            "human_review_required": ((state or {}).get("state") or {}).get("human_review_required", False),
            "safety_score":          ((state or {}).get("state") or {}).get("safety_score"),
            "crisis_detected":       ((state or {}).get("state") or {}).get("crisis_detected", False),
        },
    }


@router.get("/webhooks")
@limiter.limit("30/minute")
def list_webhook_deliveries(
    request: Request,
    thread_id: str | None = None,
    limit: int = 50,
    _: None = Depends(require_clinician),
):
    rows = db.get_webhook_deliveries(thread_id=thread_id, limit=min(limit, 200))
    return {"count": len(rows), "deliveries": [dict(r) for r in rows]}


@router.get("/report/{thread_id}/fhir")
@limiter.limit("30/minute")
def get_fhir_report(request: Request, thread_id: str, _: None = Depends(require_clinician)):
    """
    Return the FHIR R4 Bundle for the latest completed intake.
    Requires clinician JWT — patients cannot access raw FHIR bundles.
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


# ---------------------------------------------------------------------------
# Prompt A/B experiment management
# ---------------------------------------------------------------------------

@router.post("/experiments")
def create_experiment(
    name: str = Form(...),
    prompt_key: str = Form(...),
    variant_a: str = Form(...),
    variant_b: str = Form(...),
    _: None = Depends(require_clinician),
):
    from ..prompts import PROMPT_VERSIONS
    if prompt_key not in PROMPT_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt_key '{prompt_key}'. Valid keys: {list(PROMPT_VERSIONS.keys())}",
        )
    existing = db.get_active_experiment(prompt_key)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Active experiment already exists for '{prompt_key}': {existing['experiment_id']}",
        )
    exp_id = db.create_experiment(name, prompt_key, variant_a, variant_b)
    log_event("experiment_created", experiment_id=exp_id, prompt_key=prompt_key,
              variant_a=variant_a, variant_b=variant_b)
    return {"experiment_id": exp_id, "status": "active"}


@router.get("/experiments")
def list_experiments(_: None = Depends(require_clinician)):
    return {"experiments": db.list_experiments()}


@router.patch("/experiments/{experiment_id}")
def update_experiment(
    experiment_id: str,
    status: str = Form(...),
    _: None = Depends(require_clinician),
):
    if status not in ("active", "paused", "concluded"):
        raise HTTPException(status_code=400, detail="status must be active|paused|concluded")
    db.update_experiment_status(experiment_id, status)
    log_event("experiment_updated", experiment_id=experiment_id, new_status=status)
    return {"ok": True, "experiment_id": experiment_id, "status": status}
