"""Trace-collection tests — the rendered trace (deliverable #3) must capture every stage."""
from __future__ import annotations

from sightline.observability import span, trace


def test_trace_collects_nested_spans_in_order():
    with trace("query") as stages:
        with span("route") as s:
            s["decision"] = "simple"
        with span("retrieve") as s:
            s["n"] = 5
        with span("answer"):
            pass
    assert [s["name"] for s in stages] == ["route", "retrieve", "answer"]
    assert stages[0]["decision"] == "simple"
    assert all("latency_ms" in s for s in stages)


def test_spans_outside_trace_do_not_error():
    with span("standalone") as s:  # no active trace -> logs, doesn't collect
        s["x"] = 1
    # nothing to assert beyond "no exception"; a fresh trace starts empty
    with trace("q") as stages:
        pass
    assert stages == []


def test_nested_traces_isolate():
    with trace("outer") as outer:
        with span("a"):
            pass
        with trace("inner") as inner:
            with span("b"):
                pass
        with span("c"):
            pass
    assert [s["name"] for s in inner] == ["b"]
    assert [s["name"] for s in outer] == ["a", "c"]  # inner's span not leaked to outer
