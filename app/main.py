"""
main.py — Application factory.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Architecture:
    Each concern lives in its own router module under app/api/:
      patient.py   — patient-facing intake flow (/start, /chat, /report, /jobs)
      clinician.py — clinician-gated workflow (/clinician/*, /experiments)
      admin.py     — operational tooling (/admin/emergency-phrases, /demo/*)
      health.py    — observability (/health, /ready, /analytics)

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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from . import sqlite_db as db
from .api.deps import limiter, RateLimitExceeded, _rate_limit_exceeded_handler
from .api.patient import router as patient_router
from .api.clinician import router as clinician_router
from .api.admin import router as admin_router
from .api.health import router as health_router
from .graph import build_graph
from .llm import get_gemini
from .logging_utils import log_event, set_request_id
from .settings import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Request correlation middleware
# ---------------------------------------------------------------------------

class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Propagate (or generate) X-Request-Id on every request.

    • If the client sends X-Request-Id or X-Correlation-Id, that value is reused
      so client logs and server logs share a common ID.
    • If no ID is supplied, a new UUID is generated.
    • The ID is stored in a ContextVar so log_event() picks it up automatically
      without any manual threading — structured logs always contain the ID.
    • The ID is echoed back on the response so clients can match server log lines
      to specific requests.
    • Request start and end are logged with method, path, status, and duration_ms
      so slow or failing endpoints are immediately visible in the log stream.
    """
    async def dispatch(self, request: StarletteRequest, call_next):
        req_id = (
            request.headers.get("X-Request-Id")
            or request.headers.get("X-Correlation-Id")
            or str(uuid.uuid4())
        )
        set_request_id(req_id)

        t0 = time.perf_counter()
        log_event("http_request", method=request.method, path=request.url.path)

        response: StarletteResponse = await call_next(request)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-Id"] = req_id
        log_event(
            "http_response",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response


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
        get_gemini().validate()

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

    # Hourly background task: keeps trying exhausted webhooks without waiting
    # for a process restart.  Uses asyncio so no extra thread or process needed.
    async def _dead_letter_loop():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            try:
                n = retry_exhausted_webhooks()
                if n:
                    log_event("dead_letter_requeued_hourly", count=n)
            except Exception as exc:
                log_event("dead_letter_loop_error", level="warning", error=str(exc)[:200])

    task = asyncio.create_task(_dead_letter_loop())

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

    app.include_router(patient_router)
    app.include_router(clinician_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    return app


app = create_app()
