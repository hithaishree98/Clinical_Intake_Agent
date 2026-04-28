"""
admin.py — Operational / demo endpoints.

Emergency-phrase management, demo-reset, full analytics, webhook delivery log,
and prompt A/B experiments live here because they are ops-facing, not
clinical-workflow-facing.  The clinician dashboard only needs /analytics/summary
and /clinician/*; everything else is admin-only.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from .. import sqlite_db as db
from ..logging_utils import log_event
from .deps import limiter, require_clinician

router = APIRouter(prefix="/admin")



@router.get("/emergency-phrases")
def list_emergency_phrases(_: None = Depends(require_clinician)):
    phrases = db.get_emergency_phrases()
    if not phrases:
        from ..extract import DEFAULT_EMERGENCY_PHRASES
        phrases = DEFAULT_EMERGENCY_PHRASES
    return {"phrases": phrases, "count": len(phrases)}


@router.post("/emergency-phrases")
@limiter.limit("30/minute")
def add_emergency_phrase(request: Request, phrase: str = Form(...), _: None = Depends(require_clinician)):
    phrase = phrase.strip().lower()
    if not phrase:
        raise HTTPException(status_code=400, detail="Phrase cannot be empty.")
    if not db.get_emergency_phrases():
        from ..extract import DEFAULT_EMERGENCY_PHRASES
        db.seed_emergency_phrases(DEFAULT_EMERGENCY_PHRASES)
    db.add_emergency_phrase(phrase)
    log_event("emergency_phrase_added", phrase=phrase)
    return {"ok": True, "phrase": phrase}


@router.delete("/emergency-phrases")
@limiter.limit("30/minute")
def delete_emergency_phrase(request: Request, phrase: str = Form(...), _: None = Depends(require_clinician)):
    phrase = phrase.strip().lower()
    deleted = db.delete_emergency_phrase(phrase)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Phrase not found: {phrase}")
    log_event("emergency_phrase_deleted", phrase=phrase)
    return {"ok": True, "phrase": phrase}


@router.post("/demo/reset")
@limiter.limit("10/minute")
def demo_reset(request: Request, _: None = Depends(require_clinician)):
    """Wipe all session data and re-seed mock EHR patients. Requires clinician token."""
    db.reset_demo_data()
    db.seed_demo_patients()
    log_event("demo_reset", msg="Demo data wiped and re-seeded")
    return {"ok": True, "message": "Demo data reset. 3 mock patients re-seeded."}


# ---------------------------------------------------------------------------
# Full analytics (admin-only; clinician dashboard uses /analytics/summary)
# ---------------------------------------------------------------------------

@router.get("/analytics")
def analytics(_: None = Depends(require_clinician)):
    """Full operational metrics for the last 7 days. Requires clinician token."""
    from ..llm import _breaker
    data = db.get_analytics()
    data["llm_circuit_state"] = _breaker.state
    return data


# ---------------------------------------------------------------------------
# Webhook delivery log
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt A/B experiment management
# ---------------------------------------------------------------------------

@router.post("/experiments")
@limiter.limit("30/minute")
def create_experiment(
    request: Request,
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
@limiter.limit("30/minute")
def update_experiment(
    request: Request,
    experiment_id: str,
    status: str = Form(...),
    _: None = Depends(require_clinician),
):
    if status not in ("active", "paused", "concluded"):
        raise HTTPException(status_code=400, detail="status must be active|paused|concluded")
    db.update_experiment_status(experiment_id, status)
    log_event("experiment_updated", experiment_id=experiment_id, new_status=status)
    return {"ok": True, "experiment_id": experiment_id, "status": status}
