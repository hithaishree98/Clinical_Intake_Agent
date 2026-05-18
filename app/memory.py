"""
memory.py — Cross-visit patient memory merge logic.

identity → replace; allergies/conditions → union; medications → replace;
recent_complaints → rolling last 5; flags → union, never dropped.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    """Merge a completed visit's ReportInputState dump into the prior summary."""
    prior = prior or {}
    prior_visits = int(prior.get("visit_count") or 0)

    identity = visit.get("identity") or prior.get("identity") or {}

    allergies = _dedup_lower(
        list(prior.get("allergies") or []) + list(visit.get("allergies") or [])
    )[:_MAX_ALLERGIES]

    medications = list(visit.get("medications") or [])

    conditions = _dedup_lower(
        list(prior.get("conditions") or []) + list(visit.get("pmh") or [])
    )[:_MAX_CONDITIONS]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_cc = (visit.get("chief_complaint") or "").strip()
    prior_complaints = list(prior.get("recent_complaints") or [])
    if new_cc:
        prior_complaints.append({"cc": new_cc, "date": today})
    recent_complaints = prior_complaints[-_MAX_RECENT_COMPLAINTS:]

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