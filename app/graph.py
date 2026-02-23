import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
from .settings import settings
from .state import IntakeState
from . import nodes

def route(state: IntakeState):
    phase = state.get("current_phase") or "identity"
    return {
        "identity": "identity_node",
        "identity_review": "identity_review_node",
        "subjective": "subjective_node",
        "clinical_history": "clinical_history_node",
        "report": "report_node",
        "handoff": "handoff_node",
        "confirm": "confirm_node",
        "done": END,
    }.get(phase, "identity_node")

def build_graph():
    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    cp = sqlite3.connect(settings.checkpoint_db_path, timeout=30.0, check_same_thread=False)
    cp.execute("PRAGMA journal_mode=WAL;")
    cp.execute("PRAGMA synchronous=NORMAL;")
    cp.execute("PRAGMA busy_timeout=30000;")
    checkpointer = SqliteSaver(cp)

    g = StateGraph(IntakeState)
    g.add_node("identity_node", nodes.identity_node)
    g.add_node("identity_review_node", nodes.identity_review_node)
    g.add_node("subjective_node", nodes.subjective_node)
    g.add_node("clinical_history_node", nodes.clinical_history_node)
    g.add_node("report_node", nodes.report_node)
    g.add_node("handoff_node", nodes.handoff_node)
    g.add_node("confirm_node", nodes.confirm_node)
    g.add_conditional_edges(START, route)

    for n in ["identity_node", "identity_review_node", "subjective_node", "clinical_history_node", "confirm_node"]:
        g.add_conditional_edges(n, route)

    g.add_edge("handoff_node", END)
    g.add_edge("report_node", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_after=[
            "identity_node",
            "identity_review_node",
            "subjective_node",
            "clinical_history_node",
            "confirm_node"
        ],
    )
