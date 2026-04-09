"""
events.py — Lightweight in-process event bus.

Why this exists
───────────────
webhook.py dispatches specific function calls in daemon threads.  That works
but couples the call sites (nodes.py) tightly to delivery mechanism.  Adding a
new downstream system (email, pager, audit stream) requires touching every node.

The EventBus decouples producers (nodes) from consumers (webhook handlers,
FHIR push, audit log).  Nodes publish a typed event; any number of handlers
receive it — each in its own daemon thread so no handler can block another.

Design constraints (no over-engineering)
─────────────────────────────────────────
• Zero external dependencies — stdlib only.
• Handlers run in daemon threads (fire-and-forget, same guarantee as before).
• No persistent queue — if the process crashes mid-delivery that delivery is
  lost (same as the old bare-thread approach).  The DB-backed retry in
  webhook.py handles persistence for the webhook layer.
• Subscriptions are global (module-level singleton) because all intake sessions
  share the same process and the same set of handlers.

Usage
─────
    # subscribe (at startup / module import time)
    from .events import bus, IntakeCompletedEvent
    bus.subscribe(IntakeCompletedEvent, my_handler)

    # publish (in a node)
    bus.publish(IntakeCompletedEvent(thread_id=..., patient_name=..., ...))

    # handlers receive the event object and run in a daemon thread
    def my_handler(event: IntakeCompletedEvent) -> None:
        slack_alert(...)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type

from .logging_utils import log_event


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class BaseEvent:
    thread_id: str


@dataclass
class IntakeCompletedEvent(BaseEvent):
    patient_name: str
    risk_level:   str
    fhir_json:    str | None = None


@dataclass
class EmergencyFlaggedEvent(BaseEvent):
    patient_name:  str
    red_flags:     List[str] = field(default_factory=list)
    session_short: str       = ""


@dataclass
class CrisisDetectedEvent(BaseEvent):
    patient_name:   str
    matched_phrases: List[str] = field(default_factory=list)


@dataclass
class IdentityReviewNeededEvent(BaseEvent):
    patient_name: str
    discrepancy:  str = ""


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

Handler = Callable[[Any], None]


class EventBus:
    """
    Simple pub/sub bus.  Each handler runs in its own daemon thread so a slow
    or failing handler cannot block graph execution or other handlers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[Type[BaseEvent], List[Handler]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: Type[BaseEvent], handler: Handler) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: BaseEvent) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
        if not handlers:
            return
        for handler in handlers:
            t = threading.Thread(
                target=self._run,
                kwargs={"handler": handler, "event": event},
                daemon=True,
            )
            t.start()

    @staticmethod
    def _run(handler: Handler, event: BaseEvent) -> None:
        try:
            handler(event)
        except Exception as exc:
            log_event(
                "event_handler_error",
                level="error",
                event_type=type(event).__name__,
                handler=getattr(handler, "__name__", str(handler)),
                error=str(exc)[:300],
            )


# Module-level singleton — import and use directly.
bus = EventBus()
