"""
memory.py — Layer-2 patient memory merge logic.

Separated from sqlite_db.py because merging is domain logic, not storage.
Each field has its own merge rule:

  identity           → replace (most recent wins)
  allergies          → union, dedup case-insensitive, cap at 20
  medications        → replace with current meds from this visit
                       (patients stop/start meds — re-asserting the list
                        each visit is safer than unioning forever)
  conditions         → union (chronic conditions accumulate)
  recent_complaints  → append (CC, date), keep last 5
  flags              → union, no cap (crisis history is important)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


_MAX_ALLERGIES = 20
_MAX_CONDITIONS = 30
_MAX_RECENT_COMPLAINTS = 5


def _dedup_lower(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = (it or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


def merge_summary(prior: dict | None, visit: dict) -> dict:
    """
    Merge a completed visit's validated state into the prior Layer-2 summary.

    `visit` is expected to look like ReportInputState.model_dump():
      { identity, chief_complaint, allergies, medications, pmh,
        recent_results, triage, crisis_detected (optional) }
    """
    prior = prior or {}
    prior_visits = int(prior.get("visit_count") or 0)

    # identity — replace (most recent reflects current contact info)
    identity = visit.get("identity") or prior.get("identity") or {}

    # allergies — union, dedup, capped
    allergies = _dedup_lower(
        list(prior.get("allergies") or []) + list(visit.get("allergies") or [])
    )[:_MAX_ALLERGIES]

    # medications — replace with this visit's current list
    # (medications are transient; re-asserted every visit)
    medications = list(visit.get("medications") or [])

    # conditions — union from PMH
    conditions = _dedup_lower(
        list(prior.get("conditions") or []) + list(visit.get("pmh") or [])
    )[:_MAX_CONDITIONS]

    # recent_complaints — append with date, keep last N
    today = datetime.utcnow().strftime("%Y-%m-%d")
    new_cc = (visit.get("chief_complaint") or "").strip()
    prior_complaints = list(prior.get("recent_complaints") or [])
    if new_cc:
        prior_complaints.append({"cc": new_cc, "date": today})
    recent_complaints = prior_complaints[-_MAX_RECENT_COMPLAINTS:]

    # flags — add crisis history if relevant
    flags = list(prior.get("flags") or [])
    if visit.get("crisis_detected"):
        flags.append({"flag": "prior_crisis_escalation", "date": today})

    return {
        "identity":          identity,
        "allergies":         allergies,
        "medications":       medications,
        "conditions":        conditions,
        "recent_complaints": recent_complaints,
        "flags":             flags,
        "visit_count":       prior_visits + 1,
    }


def format_for_prompt(summary: dict) -> str:
    """
    Render Layer-2 summary as a compact string to inject into LLM prompts.
    Kept small so it doesn't blow the context window — a few hundred tokens max.
    """
    if not summary or summary.get("visit_count", 0) == 0:
        return ""

    parts: list[str] = [f"RETURNING_PATIENT (visit #{summary['visit_count']})"]

    allergies = summary.get("allergies") or []
    if allergies:
        parts.append(f"KNOWN_ALLERGIES: {', '.join(allergies)}")
    else:
        parts.append("KNOWN_ALLERGIES: none on file (confirm with patient)")

    meds = summary.get("medications") or []
    if meds:
        med_strs = [
            f"{m.get('name','')} {m.get('dose','')}".strip()
            for m in meds if m.get("name")
        ]
        parts.append(f"CURRENT_MEDICATIONS: {', '.join(med_strs)}")

    conditions = summary.get("conditions") or []
    if conditions:
        parts.append(f"CHRONIC_CONDITIONS: {', '.join(conditions)}")

    complaints = summary.get("recent_complaints") or []
    if complaints:
        recent = [f"{c.get('cc','')} ({c.get('date','')})" for c in complaints[-3:]]
        parts.append(f"RECENT_COMPLAINTS: {'; '.join(recent)}")

    flags = summary.get("flags") or []
    if flags:
        flag_strs = [f"{f.get('flag','')} ({f.get('date','')})" for f in flags]
        parts.append(f"CLINICAL_FLAGS: {'; '.join(flag_strs)}")

    return "\n".join(parts)