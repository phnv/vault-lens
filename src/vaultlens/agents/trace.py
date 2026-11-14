"""TraceSink collector and optional UI callback forwarder (ADR-0009 section 6.11)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ..models import TraceEvent

TraceCallback = Callable[[TraceEvent], None]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class TraceSink:
    """Collects `TraceEvent` items and optionally forwards each event to a live callback."""

    def __init__(self, on_event: TraceCallback | None = None) -> None:
        self.events: list[TraceEvent] = []
        self.on_event = on_event

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def emit(
        self,
        step: str,
        kind: str,
        summary: str,
        ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        ev = TraceEvent(
            ts=now_iso(),
            step=step,
            kind=kind,
            summary=summary,
            ms=ms,
            data=dict(data or {}),
        )
        self(ev)
        return ev

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.events)
