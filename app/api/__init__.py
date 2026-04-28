# app/api — router package.
# Each sub-module owns one concern:
#   deps.py      — shared dependencies (rate limiter, auth guards)
#   patient.py   — patient-facing endpoints (/start, /chat, /resume)
#   clinician.py — clinician-gated workflow (/clinician/token, /pending, /resolve, /case/*, /report/*/fhir)
#   admin.py     — ops-facing endpoints (/admin/emergency-phrases, /demo/*, /analytics, /webhooks, /experiments)
#   health.py    — observability endpoints (/health, /ready, /analytics/summary)
