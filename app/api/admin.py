"""
admin.py — Operational / demo endpoints.

Emergency-phrase management and demo-reset live here (not in clinician.py)
because they are primarily ops-facing, not clinical-workflow-facing.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException

from .. import sqlite_db as db
from ..logging_utils import log_event
from .deps import require_clinician

router = APIRouter(prefix="/admin")


_DEMO_SCENARIOS = [
    {
        "id": "chest_pain",
        "label": "Chest Pain (Emergency)",
        "description": "Acute crushing chest pain radiating to left arm — triggers emergency escalation",
        # Intentionally short: emergency path terminates after 4 messages (escalation triggered)
        "messages": [
            "yes",
            "John Demo, 04/15/1978, 5551230000, 123 Main St, Chicago IL",
            "yes",
            "I have severe crushing chest pain radiating to my left arm, started 20 minutes ago, 9 out of 10",
        ],
    },
    {
        "id": "routine_checkup",
        "label": "Routine Checkup",
        "description": "Standard clinic visit — tension headache, medication refill — full intake flow",
        "messages": [
            "yes",
            "Sarah Demo, 06/20/1990, 5559876543, 456 Oak Ave, Boston MA",
            "yes",
            "I've had a mild tension headache for two days, about 3 out of 10, also need a refill on my blood pressure meds",
            # clinical history
            "no known allergies",
            "lisinopril 10mg once daily, last took it this morning",
            "hypertension diagnosed 3 years ago, no surgeries",
            "blood pressure check last month, results were normal",
            "confirm",
        ],
    },
    {
        "id": "mental_health",
        "label": "Mental Health",
        "description": "Anxiety and panic attacks — mental health classification — full intake flow",
        # NOTE: avoid emergency-phrase keywords ("shortness of breath", "chest pain") so the
        # demo exercises the mental-health classification path rather than emergency escalation.
        "messages": [
            "yes",
            "Alex Demo, 03/12/1995, 5554443333, 789 Pine Rd, Seattle WA",
            "yes",
            "I've been having panic attacks almost daily for two weeks — my heart races, I feel extremely anxious and on edge, about a 7 out of 10",
            # clinical history
            "no known allergies",
            "sertraline 50mg once daily, last took it this morning",
            "generalized anxiety disorder diagnosed last year, no surgeries",
            "therapist session two weeks ago, no recent lab work",
            "confirm",
        ],
    },
]


@router.get("/emergency-phrases")
def list_emergency_phrases(_: None = Depends(require_clinician)):
    phrases = db.get_emergency_phrases()
    if not phrases:
        from ..extract import DEFAULT_EMERGENCY_PHRASES
        phrases = DEFAULT_EMERGENCY_PHRASES
    return {"phrases": phrases, "count": len(phrases)}


@router.post("/emergency-phrases")
def add_emergency_phrase(phrase: str = Form(...), _: None = Depends(require_clinician)):
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
def delete_emergency_phrase(phrase: str = Form(...), _: None = Depends(require_clinician)):
    phrase = phrase.strip().lower()
    deleted = db.delete_emergency_phrase(phrase)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Phrase not found: {phrase}")
    log_event("emergency_phrase_deleted", phrase=phrase)
    return {"ok": True, "phrase": phrase}


@router.get("/demo/scenarios")
def demo_scenarios():
    """Demo scenario presets — no auth required (public facing)."""
    return {"scenarios": _DEMO_SCENARIOS}


@router.post("/demo/reset")
def demo_reset(_: None = Depends(require_clinician)):
    """Wipe all session data and re-seed mock EHR patients. Requires clinician token."""
    db.reset_demo_data()
    db.seed_demo_patients()
    log_event("demo_reset", msg="Demo data wiped and re-seeded")
    return {"ok": True, "message": "Demo data reset. 3 mock patients re-seeded."}
