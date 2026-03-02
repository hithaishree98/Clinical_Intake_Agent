# Architecture and Design Decisions

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP (REST, FormData)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Rate Limiter│  │  JWT Auth    │  │ Idempotency Layer  │ │
│  │  (slowapi)  │  │  (clinician) │  │  (thread+msg_id)   │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│                    Background Task Queue                     │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                     │
│                                                              │
│  route() ──► identity_node                                   │
│          ──► identity_review_node                            │
│          ──► subjective_node ──► [red flag → handoff_node]   │
│          ──► clinical_history_node                           │
│          ──► confirm_node                                    │
│          ──► report_node                                     │
│          ──► handoff_node (emergency terminal)               │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌─────────────────────────────────────┐
│   Gemini 2.0     │   │          SQLite (WAL mode)           │
│   Flash (LLM)    │   │                                      │
│                  │   │  app.db            checkpoints.db    │
│  extract.py      │   │  ├─ sessions       └─ LangGraph      │
│  (regex only,    │   │  ├─ messages           snapshots     │
│   no LLM)        │   │  ├─ reports                         │
│                  │   │  ├─ escalations                      │
│  fhir_builder.py │   │  ├─ jobs                            │
│  (pure Python,   │   │  ├─ mock_ehr                        │
│   no LLM)        │   │  ├─ idempotency                     │
└──────────────────┘   │  ├─ session_state                   │
                       │  └─ emergency_phrases               │
                       └─────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│   Outbound Integrations          │
│   Slack webhook  (3 events)      │
│   FHIR webhook   (on completion) │
└──────────────────────────────────┘
```
