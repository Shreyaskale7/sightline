"""Router tests — including measured routing accuracy against the labeled golden set."""
from __future__ import annotations

from pathlib import Path

from sightline.agents.router import Route, route
from sightline.eval.dataset import load_golden_set

_GOLDEN = Path(__file__).parent.parent / "src" / "sightline" / "eval" / "golden_set.yaml"


def test_comparison_when_two_companies():
    d = route("Which spent more on R&D in its most recent 10-K, NVIDIA or Intel?")
    assert d.route is Route.COMPARISON and set(d.tickers) == {"NVDA", "INTC"}


def test_multi_hop_on_trend_phrasing():
    assert route("How did AMD's quarterly net revenue change across its last three 10-Q filings?").route \
        is Route.MULTI_HOP


def test_simple_single_fact():
    d = route("What was Micron's total revenue in its most recent 10-K?")
    assert d.route is Route.SIMPLE and d.tickers == ["MU"]


def test_router_matches_every_golden_slice():
    """Routing accuracy pinned at 100% on the golden set — a new case that breaks routing
    should fail here loudly instead of silently taking the wrong (or costlier) path."""
    expected = {"basic": Route.SIMPLE, "cross_company": Route.COMPARISON,
                "multi_hop": Route.MULTI_HOP}
    for c in load_golden_set(_GOLDEN):
        want = expected.get(c.slice)
        if want is not None:
            assert route(c.question).route is want, f"{c.id}: misrouted"
