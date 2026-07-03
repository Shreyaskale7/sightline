"""Router (M3 floor): classify a question so simple ones skip the expensive path.

The router exists as a COST decision, not an intelligence one: a simple factual lookup
("NVIDIA's revenue?") needs one retrieval + one answer call, while a comparison or a trend
question needs decomposition into sub-questions (M3 planner). Routing everything through the
planner would multiply cost for no quality gain on the easy 80%.

This version is deterministic — the same string signals the metadata filter already trusts:
  - two or more corpus companies mentioned  -> comparison  (decompose per company)
  - multi-period/trend phrasing             -> multi_hop   (decompose per period)
  - everything else                         -> simple      (one-shot path)

Out-of-domain detection is deliberately NOT routed here: recognizing every non-corpus entity
deterministically is impossible, and the verifier already forces abstention when retrieval
can't support an answer. The M3 upgrade is an LLM classifier (cheap model) — which must beat
this zero-cost baseline on routing accuracy before it ships (same rule as everything else).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..retrieval.filters import parse_query_filters

_MULTI_HOP = re.compile(
    r"\b(across|over the (last|past)|last (two|three|four|\d+)|trend|"
    r"quarter[- ]over[- ]quarter|year[- ]over[- ]year|each (quarter|year)|"
    r"change[ds]? (across|over|between)|how did .+ change)\b",
    re.IGNORECASE,
)


class Route(str, Enum):
    SIMPLE = "simple"          # one-shot: retrieve -> answer -> verify
    COMPARISON = "comparison"  # decompose per company, then synthesize
    MULTI_HOP = "multi_hop"    # decompose per period/aspect, then synthesize


@dataclass
class RouteDecision:
    route: Route
    tickers: list[str]
    reason: str


def route(question: str) -> RouteDecision:
    filters = parse_query_filters(question)
    if len(filters.tickers) >= 2:
        return RouteDecision(
            route=Route.COMPARISON,
            tickers=filters.tickers,
            reason=f"mentions {len(filters.tickers)} corpus companies",
        )
    if _MULTI_HOP.search(question):
        return RouteDecision(
            route=Route.MULTI_HOP,
            tickers=filters.tickers,
            reason="multi-period/trend phrasing",
        )
    return RouteDecision(route=Route.SIMPLE, tickers=filters.tickers, reason="single-fact shape")
