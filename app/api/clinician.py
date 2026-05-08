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
from ..settings import get_settings
from .deps import limiter, require_clinician

router = APIRouter(prefix="/clinician")


@router.post("/token")
@limiter.limit("5/minute")
def clinician_token(request: Request, password: str = Form(...)):
    settings = get_settings()
    if password != settings.clinician_password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    lifetime = int(settings.clinician_token_lifetime_seconds)
    token = jwt.encode(
        {"sub": "clinician", "exp": time.time() + lifetime},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer", "expires_in": lifetime}


@router.get("/pending")
@limiter.limit("60/minute")
def clinician_pending(request: Request, _: None = Depends(require_clinician)):
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

