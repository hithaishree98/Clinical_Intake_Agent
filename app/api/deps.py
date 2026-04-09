"""
deps.py — Shared FastAPI dependencies and middleware helpers.

Kept in one place so every router imports from here rather than re-declaring
the same auth logic.  Easier to swap implementations (e.g. switch rate-limiter
library) without touching individual route files.
"""
from __future__ import annotations

import time

import jwt
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    class RateLimitExceeded(Exception):
        pass

    def get_remote_address(request):
        return "test-client"

    def _rate_limit_exceeded_handler(request, exc):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    class Limiter:
        def __init__(self, key_func=None):
            self.key_func = key_func

        def limit(self, _rule: str):
            def decorator(fn):
                return fn
            return decorator

from .. import sqlite_db as db
from ..settings import get_settings


limiter = Limiter(key_func=get_remote_address)


def require_session_token(thread_id: str, authorization: str = Header(default="")) -> None:
    """
    Verify the patient's bearer session token.

    Issued at POST /start, must be supplied as:
        Authorization: Bearer <session_token>

    Constant-time comparison prevents timing-oracle attacks on the token.
    """
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing session token.")
    if not db.verify_session_token(thread_id, token):
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")


def require_clinician(authorization: str = Header(default="")) -> None:
    """
    Validate a short-lived JWT issued by POST /clinician/token.

    Clinician-only endpoints call this as a dependency so patients cannot
    access case data, escalation lists, or admin operations.
    """
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token.")
    settings = get_settings()
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
