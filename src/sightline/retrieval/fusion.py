"""Reciprocal Rank Fusion (RRF): combine ranked lists from different retrievers.

The idea: each retriever votes with *ranks*, not raw scores. A page at rank r in one list
earns 1/(K + r); its fused score is the sum of those across all lists. Raw scores from
different systems (cosine similarity vs BM25) live on incomparable scales — ranks don't,
which is why RRF needs no per-system calibration and is a famously strong baseline.

K (default 60, from the original RRF paper) damps the gap between rank 1 and rank 2 so a
single retriever's top hit can't dominate; pages that MULTIPLE retrievers like float up.

In M2 a third list (visual/ColModernVBERT) joins the same fusion — this function already
takes any number of lists.
"""
from __future__ import annotations

from .text_baseline import Hit

_K = 60


def rrf(*ranked_lists: list[Hit], k: int = _K, top_n: int | None = None) -> list[Hit]:
    """Fuse ranked Hit lists. Returns hits sorted by fused score (best first)."""
    fused: dict[tuple[str, int], float] = {}
    best_hit: dict[tuple[str, int], Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            key = (h.accession, h.page_no)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best_hit:
                best_hit[key] = h
    ordered = sorted(fused, key=fused.get, reverse=True)
    if top_n is not None:
        ordered = ordered[:top_n]
    return [
        Hit(
            accession=key[0],
            page_no=key[1],
            score=fused[key],
            ticker=best_hit[key].ticker,
            form=best_hit[key].form,
        )
        for key in ordered
    ]
