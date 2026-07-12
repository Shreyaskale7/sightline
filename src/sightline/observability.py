"""Tracing helper.

Wrap each pipeline stage in `span(...)` so a single trace flows through the whole request
(router -> retrieval -> rerank -> answerer -> verifier). Two collectors:

  - Always: a per-request `trace(...)` context (contextvar-based) gathers every span into an
    ordered list, so the API can return "here is exactly what happened, stage by stage, with
    timings" — deliverable #3, the rendered trace, with no external service required.
  - Optionally: if Langfuse is configured, spans also push there. Otherwise spans print to logs.

Why this matters: observability is a core AI-engineering skill. You want to see, per request,
which pages were retrieved, what each stage cost, and where latency went.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from .config import settings

# The active per-request trace (a list of stage records), or None outside a trace() block.
_active_trace: ContextVar[list[dict] | None] = ContextVar("_active_trace", default=None)


@contextmanager
def trace(name: str, **metadata: Any) -> Iterator[list[dict]]:
    """Collect all spans emitted inside this block into one ordered list of stage records."""
    records: list[dict] = []
    token = _active_trace.set(records)
    start = time.perf_counter()
    try:
        yield records
    finally:
        _active_trace.reset(token)
        # A trace is itself printed as a summary line for the logs.
        total = round((time.perf_counter() - start) * 1000, 1)
        if not settings.tracing_enabled:
            print(f"[trace {name}] {len(records)} stages, {total}ms")


@contextmanager
def span(name: str, **metadata: Any) -> Iterator[dict]:
    """Time a stage; append it to the active trace and (optionally) emit to Langfuse.

    Usage:
        with span("retrieval", query=q) as s:
            hits = retrieve(q)
            s["result_count"] = len(hits)
    """
    record: dict[str, Any] = {"name": name, **metadata}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        active = _active_trace.get()
        if active is not None:
            active.append(record)
        if settings.tracing_enabled:
            # TODO(M4): push `record` to Langfuse as a nested span on the current trace.
            pass
        elif active is None:
            print(f"[trace] {record}")  # standalone span outside a trace -> log it
