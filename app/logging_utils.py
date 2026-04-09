import json
import logging
import datetime
import re
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger("intake")

_trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)
_node_id_ctx: ContextVar[str | None] = ContextVar("node_id", default=None)

def set_trace_id(trace_id: str) -> None:
    _trace_id_ctx.set(trace_id)

def get_trace_id() -> str | None:
    return _trace_id_ctx.get()

def set_request_id(request_id: str) -> None:
    _request_id_ctx.set(request_id)

def get_request_id() -> str | None:
    return _request_id_ctx.get()

def set_job_id(job_id: str) -> None:
    _job_id_ctx.set(job_id)

def get_job_id() -> str | None:
    return _job_id_ctx.get()

def set_node_id(node_id: str) -> None:
    """Track which graph node is currently executing for distributed tracing."""
    _node_id_ctx.set(node_id)

def get_node_id() -> str | None:
    return _node_id_ctx.get()


# ---------------------------------------------------------------------------
# PHI masking — HIPAA compliance
# ---------------------------------------------------------------------------
# Fields whose values may contain patient-identifiable information.
# Values are replaced with "[REDACTED]" before any log is emitted so that
# application logs never contain PII regardless of call site.
_PHI_FIELDS: frozenset[str] = frozenset({
    "name", "patient_name", "dob", "date_of_birth", "phone",
    "address", "identity", "stored_identity",
})

# Patterns that indicate a value is a PHI data point even when the key name
# is generic.  Conservative: false positives (redacting a non-PHI field) are
# acceptable; false negatives (logging real PHI) are not.
_PHI_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),          # phone
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                 # DOB MM/DD/YYYY
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                        # ISO date YYYY-MM-DD
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        # SSN XXX-XX-XXXX
)


def mask_phi(fields: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of *fields* with PHI values replaced by "[REDACTED]".

    Rules (applied in order):
      1. If a top-level key is in _PHI_FIELDS, redact the entire value.
      2. If a string value matches a phone / DOB regex, redact it.
      3. Nested dicts are recursively masked (e.g. identity sub-dict).

    This is intentionally conservative: false positives (redacting a
    non-PHI field) are acceptable; false negatives (logging real PHI) are not.
    """
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if k in _PHI_FIELDS:
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = mask_phi(v)
        elif isinstance(v, str) and any(p.search(v) for p in _PHI_PATTERNS):
            out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

def _base_payload(event: str, level: str, **fields) -> dict:
    payload = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "event": event,
        **{k: v for k, v in fields.items() if v is not None},
    }
    trace_id  = get_trace_id()
    request_id = get_request_id()
    job_id    = get_job_id()
    node_id   = get_node_id()

    if trace_id   and "trace_id"   not in payload: payload["trace_id"]   = trace_id
    if request_id and "request_id" not in payload: payload["request_id"] = request_id
    if job_id     and "job_id"     not in payload: payload["job_id"]     = job_id
    if node_id    and "node_id"    not in payload: payload["node_id"]    = node_id

    return payload


def log_event(event: str, level: str = "info", **fields) -> None:
    """
    Emit a structured JSON log line.

    PHI fields are automatically redacted so callers do not need to
    sanitise values before logging — the masking happens here unconditionally.
    """
    payload = mask_phi(_base_payload(event, level, **fields))
    log = getattr(logger, level if level in ("debug", "info", "warning", "error", "critical") else "info")
    log(json.dumps(payload, ensure_ascii=False))


def log_audit(event: str, **fields) -> None:
    """
    Emit an audit-flagged log line.

    Audit events are always at INFO level and carry ``audit=True`` so they
    can be routed to a dedicated audit sink by the log aggregator.
    PHI is redacted identically to log_event.
    """
    payload = mask_phi(_base_payload(event, "info", audit=True, **fields))
    logger.info(json.dumps(payload, ensure_ascii=False))
