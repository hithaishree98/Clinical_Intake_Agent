
# System Design & Architecture

The system has four main layers:

- API Layer (FastAPI)

- Workflow Layer (LangGraph state machine)

- Business Logic Nodes

- Persistence Layer (SQLite + WAL)

- LLM Integration Layer

Everything revolves around a thread_id, which represents a single intake session.
