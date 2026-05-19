"""
main.py — Application factory.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Architecture:
    Each concern lives in its own router module under app/api/:
      patient.py   — patient-facing intake flow (/start, /chat, /resume)
      clinician.py — clinician-gated workflow (/clinician/token, /pending, /resolve, /case/*, /report/*/fhir)
      admin.py     — operational tooling (/admin/emergency-phrases, /demo/*, /analytics, /webhooks, /experiments)
      health.py    — observability (/health, /ready, /analytics/summary)

    Shared dependencies (rate limiter, auth guards) live in app/api/deps.py
    so every router imports from one place instead of re-declaring them.
"""
import asyncio
import logging
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest

from . import sqlite_db as db
from .api.deps import limiter, RateLimitExceeded, _rate_limit_exceeded_handler
from .api.patient import router as patient_router
from .api.clinician import router as clinician_router
from .api.admin import router as admin_router
from .api.health import router as health_router
from .api.voice import router as voice_router
from .graph import build_graph
from .llm import get_provider
from .logging_utils import log_event, set_request_id
from .settings import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Request correlation middleware
# ---------------------------------------------------------------------------

class CorrelationMiddleware:
    """
    Propagate (or generate) X-Request-Id on every request.

    Pure ASGI middleware — avoids BaseHTTPMiddleware's child-task isolation so
    ContextVars set inside endpoint handlers (e.g. set_trace_id) are visible
    here when the http_response line is logged.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request = StarletteRequest(scope, receive)
        req_id = (
            request.headers.get("X-Request-Id")
            or request.headers.get("X-Correlation-Id")
            or str(uuid.uuid4())
        )
        set_request_id(req_id)

        t0 = time.perf_counter()
        log_event("http_request", method=request.method, path=request.url.path)

        status_code = 500

        async def send_with_header(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", req_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_header)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        log_event(
            "http_response",
            method=request.method,
            path=request.url.path,
            status=status_code,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db.init_schema()

    if not db.get_emergency_phrases():
        from .extract import DEFAULT_EMERGENCY_PHRASES
        db.seed_emergency_phrases(DEFAULT_EMERGENCY_PHRASES)

    if not settings.debug_mode:
        get_provider().validate()

    app.state.graph = build_graph()

    expired = db.expire_stale_sessions(ttl_hours=settings.intake.session_ttl_hours)
    if expired:
        log_event("startup_sessions_expired", count=expired)

    pruned = db.prune_old_checkpoints(days=settings.intake.checkpoint_retention_days)
    if pruned:
        log_event("checkpoints_pruned", count=pruned)

    # Dead-letter recovery at startup: re-queue deliveries that exhausted all
    # retries while the downstream endpoint was down between restarts.
    from .webhook import retry_exhausted_webhooks
    requeued = retry_exhausted_webhooks()
    if requeued:
        log_event("dead_letter_requeued_on_startup", count=requeued)

    # Hourly background loop: dead-letter recovery + stale-job sweep.
    # Both used to be triggered inline (dead_letter on startup only,
    # stale-jobs on every /jobs poll); the loop centralises the cadence so
    # neither contends with request-path writes and ops can grep one log
    # for periodic-housekeeping events.
    async def _hourly_background_loop():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            try:
                n = retry_exhausted_webhooks()
                if n:
                    log_event("dead_letter_requeued_hourly", count=n)
            except Exception as exc:
                log_event("dead_letter_loop_error", level="warning", error=str(exc)[:200])
            try:
                stale = db.mark_stale_jobs_failed(stale_minutes=10)
                if stale:
                    log_event("stale_jobs_expired_hourly", level="warning", count=stale)
            except Exception as exc:
                log_event("stale_jobs_loop_error", level="warning", error=str(exc)[:200])
            try:
                expired_sessions = db.expire_stale_sessions(
                    ttl_hours=settings.intake.session_ttl_hours,
                )
                if expired_sessions:
                    log_event("sessions_expired_hourly", count=expired_sessions)
            except Exception as exc:
                log_event("session_expiry_loop_error", level="warning", error=str(exc)[:200])

    task = asyncio.create_task(_hourly_background_loop())

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Clinical Intake", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Correlation middleware must be added BEFORE CORSMiddleware so the
    # X-Request-Id header is present on CORS pre-flight responses too.
    app.add_middleware(CorrelationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        return (STATIC_DIR / "admin.html").read_text(encoding="utf-8")

    app.include_router(patient_router)
    app.include_router(clinician_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    app.include_router(voice_router)
    return app


app = create_app()
