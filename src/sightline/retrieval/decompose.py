"""Deterministic query decomposition for retrieval (the planner's free floor).

The measured failures this attacks:
  - cross_company R@5 = 0.25 — one search can't surface two companies' statement pages;
    whichever company embeds closer to the query hogs the top-k.
  - multi_hop R@5 = 0.22 — "across the last three 10-Qs" needs a page from EACH filing,
    but one search returns near-duplicates from whichever filing ranks best.

The fix needs no LLM: the router already classifies these shapes deterministically, and the
metadata filter already knows which companies are mentioned. So:
  comparison -> run the SAME query once per company (ticker-filtered), interleave results
  multi_hop  -> run it once per recent filing of the company (accession-filtered), interleave
  simple     -> plain filtered retrieval (the current champion config)

Interleaving (round-robin) rather than RRF is deliberate: for these questions we don't want
the "globally best" pages — we want GUARANTEED representation from each company/filing in
the top-k, because the answer needs all of them. An LLM planner (M3) must beat this at $0.
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..agents.router import Route, route
from .filters import QueryFilters, parse_query_filters, to_qdrant_filter
from .text_baseline import Hit

# retrieve_fn signature: (query, k, query_filter) -> list[Hit]
RetrieveFn = Callable[..., list[Hit]]
# list_accessions signature: (ticker, form, limit) -> list[str]
ListAccessionsFn = Callable[..., list[str]]


def interleave(lists: Sequence[list[Hit]], top_n: int) -> list[Hit]:
    """Round-robin merge, deduped — guarantees each source list representation up front."""
    out: list[Hit] = []
    seen: set[tuple[str, int]] = set()
    for i in range(max((len(hits) for hits in lists), default=0)):
        for hits in lists:
            if i < len(hits):
                key = (hits[i].accession, hits[i].page_no)
                if key not in seen:
                    seen.add(key)
                    out.append(hits[i])
                    if len(out) >= top_n:
                        return out
    return out


def _accession_filter(accession: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="accession", match=MatchValue(value=accession))])


def decomposed_retrieve(
    query: str,
    k: int,
    retrieve_fn: RetrieveFn,
    list_accessions: ListAccessionsFn,
    filter_override: QueryFilters | None = None,
) -> list[Hit]:
    """Route the query and fan retrieval out per company or per filing when that shape needs it.

    filter_override: an explicit scope from the serving layer (e.g. "search only my uploaded
    documents"). When set, it REPLACES the query-parsed company/form filter and takes the plain
    single-query path — SEC cross-company decomposition doesn't apply to a user's own corpus.
    The eval harness never passes it, so the benchmark path is provably unchanged.
    """
    if filter_override is not None:
        return retrieve_fn(query, k=k, query_filter=to_qdrant_filter(filter_override))

    decision = route(query)
    filters = parse_query_filters(query)

    if decision.route is Route.COMPARISON and len(decision.tickers) >= 2:
        per_company = [
            retrieve_fn(
                query, k=k,
                query_filter=to_qdrant_filter(QueryFilters(tickers=[t], form=filters.form)),
            )
            for t in decision.tickers
        ]
        return interleave(per_company, top_n=k)

    if decision.route is Route.MULTI_HOP and len(decision.tickers) == 1:
        accessions = list_accessions(decision.tickers[0], filters.form or "10-Q", 3)
        if len(accessions) >= 2:
            per_filing = [
                retrieve_fn(query, k=k, query_filter=_accession_filter(a)) for a in accessions
            ]
            return interleave(per_filing, top_n=k)

    # simple (or under-specified) -> the champion single-query path
    return retrieve_fn(query, k=k, query_filter=to_qdrant_filter(filters))
