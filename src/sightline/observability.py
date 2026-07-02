"""Tracing helper.

Wrap each pipeline stage in `span(...)` so a single trace_id flows through the whole
request (router -> retrieval -> answerer -> verifier). If Langfuse isn't configured,
this degrades to a no-op that still prints timing to the logs, so nothing breaks in dev.

Why this matters: observability is a core AI-engineering skill. You want to see, per
request, which pages were retrieved, what each stage cost, and where latency went.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings


@contextmanager
def span(name: str, **metadata: Any) -> Iterator[dict]:
    """Time a stage and (later) emit it to Langfuse.

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
        if settings.tracing_enabled:
            # TODO(M3/M4): push `record` to Langfuse as a nested span on the current trace.
            pass
        else:
            print(f"[trace] {record}")
