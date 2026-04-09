# app/api — router package.
# Each sub-module owns one concern:
#   deps.py      — shared dependencies (rate limiter, auth guards)
#   patient.py   — patient-facing endpoints (/start, /chat, /report, /jobs)
#   clinician.py — clinician-gated endpoints (/clinician/*, /experiments)
#   admin.py     — operational endpoints (/admin/emergency-phrases, /demo/*)
#   health.py    — observability endpoints (/health, /ready, /analytics)
