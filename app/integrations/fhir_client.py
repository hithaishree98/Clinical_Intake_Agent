"""
fhir_client.py — Direct FHIR R4 server integration.

Supports any FHIR R4-compliant server (HAPI FHIR, Azure Health Data Services,
Google Cloud Healthcare API, Epic sandbox).

Usage
─────
Configure FHIR_SERVER_URL in .env to point at your server:

    # Local HAPI FHIR (docker-compose)
    FHIR_SERVER_URL=http://hapi-fhir:8080/fhir

    # Azure Health Data Services
    FHIR_SERVER_URL=https://<workspace>.fhir.azurehealthcareapis.com

    # Google Cloud Healthcare API
    FHIR_SERVER_URL=https://healthcare.googleapis.com/v1/projects/<proj>/locations/<loc>/datasets/<ds>/fhirStores/<store>/fhir

The client sends a FHIR Transaction Bundle (type=transaction) so all resources
are created atomically.  Each resource gets a PUT with a conditional-create URL
so duplicate intakes are idempotent (same patient, same encounter).

For authenticated servers (Azure, GCP, Epic) add FHIR_SERVER_BEARER_TOKEN to
.env.  For SMART on FHIR flows, swap in your OAuth2 token here.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..logging_utils import log_event
from ..settings import get_settings as settings


class FhirServerError(Exception):
    """Raised when the FHIR server returns a non-2xx or OperationOutcome error."""
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"FHIR server {status}: {body[:200]}")


def _transaction_bundle(document_bundle: dict) -> dict:
    """
    Convert a document Bundle into a transaction Bundle so the server persists
    each resource individually (searchable, patchable, etc.).

    Document bundles contain the full resources; transaction bundles wrap them
    in request entries so the server knows what HTTP method to use.
    """
    entries = []
    for entry in document_bundle.get("entry", []):
        resource = entry.get("resource", {})
        r_type   = resource.get("resourceType", "Resource")
        r_id     = resource.get("id", "")
        entries.append({
            "fullUrl": entry.get("fullUrl", f"urn:uuid:{r_id}"),
            "resource": resource,
            "request": {
                "method": "PUT",
                "url":    f"{r_type}/{r_id}",
            },
        })
    return {
        "resourceType": "Bundle",
        "type":         "transaction",
        "entry":        entries,
    }


def push_bundle(fhir_bundle_json: str, thread_id: str) -> dict[str, Any]:
    """
    POST a FHIR R4 Bundle to the configured FHIR server.

    Returns a dict with keys:
      ok          bool
      status      HTTP status code (or 0 on network error)
      response    parsed JSON response body (or {})
      error       error message string (empty on success)

    Callers should treat this as best-effort — a failed push does not block
    the patient-facing intake flow.  The original bundle is always saved to the
    local DB regardless of FHIR server availability.
    """
    url = getattr(settings, "fhir_server_url", "")
    if not url:
        return {"ok": False, "status": 0, "response": {}, "error": "FHIR_SERVER_URL not configured"}

    try:
        doc_bundle  = json.loads(fhir_bundle_json)
        txn_bundle  = _transaction_bundle(doc_bundle)
        payload     = json.dumps(txn_bundle).encode("utf-8")
    except Exception as e:
        return {"ok": False, "status": 0, "response": {}, "error": f"bundle_parse: {e}"}

    headers: dict[str, str] = {
        "Content-Type": "application/fhir+json",
        "Accept":       "application/fhir+json",
        "X-Thread-Id":  thread_id,
    }
    # Optional bearer token for authenticated FHIR servers
    bearer = getattr(settings, "fhir_server_bearer_token", "")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    try:
        req  = urllib.request.Request(url, data=payload, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                resp_json = json.loads(body)
            except Exception:
                resp_json = {"raw": body[:500]}
            log_event(
                "fhir_server_push_success",
                thread_id=thread_id,
                http_status=resp.status,
                resource_count=len(doc_bundle.get("entry", [])),
            )
            return {"ok": True, "status": resp.status, "response": resp_json, "error": ""}

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        # FHIR servers return OperationOutcome on errors — surface the issue text
        try:
            outcome = json.loads(body)
            issues  = outcome.get("issue", [{}])
            detail  = issues[0].get("diagnostics") or issues[0].get("details", {}).get("text", "")
        except Exception:
            detail = body[:300]
        log_event(
            "fhir_server_push_failed",
            level="warning",
            thread_id=thread_id,
            http_status=e.code,
            detail=detail,
        )
        return {"ok": False, "status": e.code, "response": {}, "error": detail}

    except Exception as exc:
        log_event(
            "fhir_server_push_error",
            level="warning",
            thread_id=thread_id,
            error=str(exc)[:300],
        )
        return {"ok": False, "status": 0, "response": {}, "error": str(exc)[:300]}
