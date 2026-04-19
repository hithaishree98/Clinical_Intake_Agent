"""
health.py — Observability endpoints.

/health  — liveness probe (always 200 if the process is up)
/ready   — readiness probe (503 if DB / graph / LLM circuit is degraded)
/analytics      — full operational metrics (clinician-gated)
/analytics/summary — dashboard summary (clinician-gated)

Keeping these in a dedicated router makes it trivial to add to a monitoring
stack: point Kubernetes / ECS health checks at /health, readiness checks at
/ready, and scrape /analytics for a Grafana dashboard.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import sqlite_db as db
from ..logging_utils import log_event
from .deps import require_clinician

router = APIRouter()


@router.get("/health")
def health(request: Request):
    """
    Liveness probe — returns 200 if the process is running.

    Also surfaces the LLM circuit breaker state so an alert can fire before
    the circuit fully opens and patient intakes start degrading.
    """
    from ..llm import _breaker
    return {
        "status": "ok",
        "llm_circuit": _breaker.state,
        "llm_failure_count": _breaker._failures,
        "version": "1.0.0",
    }


@router.get("/ready")
def ready(request: Request):
    """
    Readiness probe — returns 200 only when all subsystems are healthy.

    Kubernetes / ECS should gate traffic on this endpoint, not /health.
    Checks:
      • SQLite reachable (SELECT 1)
      • LangGraph initialised (graph not None)
      • LLM circuit breaker not open (would reject all LLM calls)
    """
    from ..llm import _breaker
    errors = []

    try:
        db.fetch_one("SELECT 1")
    except Exception as e:
        errors.append(f"db: {str(e)[:100]}")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        errors.append("graph: not initialised")

    if _breaker.state == "open":
        errors.append("llm: circuit breaker open")

    if errors:
        log_event("readiness_check_failed", level="warning", errors=errors)
        raise HTTPException(status_code=503, detail={"errors": errors})

    return {"status": "ready", "llm_circuit": _breaker.state}


@router.get("/analytics")
def analytics(_: None = Depends(require_clinician)):
    """Full operational metrics for the last 7 days. Requires clinician token."""
    from ..llm import _breaker
    data = db.get_analytics()
    data["llm_circuit_state"] = _breaker.state
    return data


@router.get("/analytics/summary")
def analytics_summary(_: None = Depends(require_clinician)):
    """Lightweight summary for the dashboard."""
    from ..llm import _breaker
    data = db.get_analytics()
    return {
        "sessions_today":          data["sessions_today"],
        "sessions_last_7_days":    data["sessions_last_7_days"],
        "completion_rate_pct":     data["completion_rate_pct"],
        "escalations_last_7_days": data["escalations_last_7_days"],
        "pending_escalations":     data["pending_escalations"],
        "llm_fallback_rate_pct":   round(
            data["llm_failures_last_7_days"]["total_llm_failures"]
            / max(data["sessions_last_7_days"], 1) * 100, 1
        ),
        "llm_circuit_state":       _breaker.state,
        "failed_report_jobs":      data["failed_report_jobs_last_7_days"],
        "llm_cost_today_usd":      data.get("llm_cost_today_usd", 0.0),
        "repair_rate_last_1h":     data.get("repair_rate_last_1h", 0.0),
        "repair_rate_alert":       data.get("repair_rate_alert", False),
    }
