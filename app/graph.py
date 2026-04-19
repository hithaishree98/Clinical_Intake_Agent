import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
from .settings import get_settings as settings
from .state import IntakeState
from . import nodes


def route(state: IntakeState):
    phase = state.get("current_phase") or "identity"
    return {
        "consent":          "consent_node",
        "identity":         "identity_node",
        "identity_review":  "identity_review_node",
        "subjective":       "subjective_node",
        "validate":         "validate_node",
        "clinical_history": "clinical_history_node",
        "report":           "report_node",
        "handoff":          "handoff_node",
        "confirm":          "confirm_node",
        "done":             END,
    }.get(phase, "identity_node")


def route_after_guard(state: IntakeState):
    """
    After guard_node runs:
    - Crisis detected → END directly. guard_node already set the patient-facing
      message (CRISIS_RESOURCE) and fired all side effects (DB escalation, webhook).
      Routing through handoff_node would add a redundant message.
    - No crisis → route normally by phase (guard_node returned {} unchanged).
    """
    if state.get("crisis_detected") and state.get("current_phase") == "handoff":
        return END
    return route(state)


def build_graph():
    Path(settings().checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    cp = sqlite3.connect(settings().checkpoint_db_path, timeout=30.0, check_same_thread=False)
    cp.execute("PRAGMA journal_mode=WAL;")
    cp.execute("PRAGMA synchronous=NORMAL;")
    cp.execute("PRAGMA busy_timeout=30000;")
    checkpointer = SqliteSaver(cp)

    g = StateGraph(IntakeState)

    # guard_node is the single entry point for every user message.
    # It runs crisis detection once, centrally, before any business node.
    g.add_node("guard_node",            nodes.guard_node)
    g.add_node("consent_node",          nodes.consent_node)
    g.add_node("identity_node",         nodes.identity_node)
    g.add_node("identity_review_node",  nodes.identity_review_node)
    g.add_node("subjective_node",       nodes.subjective_node)
    g.add_node("validate_node",         nodes.validate_node)
    g.add_node("clinical_history_node", nodes.clinical_history_node)
    g.add_node("report_node",           nodes.report_node)
    g.add_node("handoff_node",          nodes.handoff_node)
    g.add_node("confirm_node",          nodes.confirm_node)

    # Every message enters through guard_node first.
    g.add_conditional_edges(START, lambda _: "guard_node")
    g.add_conditional_edges("guard_node", route_after_guard)

    for n in [
        "consent_node",
        "identity_node",
        "identity_review_node",
        "subjective_node",
        "validate_node",
        "clinical_history_node",
        "confirm_node",
    ]:
        g.add_conditional_edges(n, route)

    g.add_edge("handoff_node", END)
    g.add_edge("report_node", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_after=[
            # guard_node intentionally NOT here — it must be transparent:
            # run safety check then immediately continue to the business node
            # in the same graph.invoke() call. Adding it here would consume
            # the user message in the guard pause and starve the business node.
            "consent_node",
            "identity_node",
            "identity_review_node",
            "subjective_node",
            "clinical_history_node",
            "confirm_node",
        ],
    )