"""RoutedRetriever branch logic — comparison skips rerank, everything else reranks.

Built with fakes (no Qdrant, no models) by bypassing __init__, so the routing decision itself
is what's under test — the measured reason the router exists on the retrieval path.
"""
from __future__ import annotations

from sightline.retrieval.routed import RoutedRetriever
from sightline.retrieval.text_baseline import Hit


class _FakeChunked:
    def retrieve(self, query, k=5, query_filter=None):
        return [Hit("acc", i, 1.0, ticker="NVDA", form="10-K") for i in range(1, k + 1)]


class _FakeStore:
    def get_page(self, accession, page_no):
        return None

    def list_accessions(self, ticker, form=None, limit=3):
        return []


class _FakeReranker:
    def __init__(self):
        self.called = False

    def rerank(self, query, pairs, k=5):
        self.called = True
        return [h for h, _ in pairs][:k]


def _routed() -> tuple[RoutedRetriever, _FakeReranker]:
    r = object.__new__(RoutedRetriever)  # bypass heavy __init__
    r._chunked = _FakeChunked()
    r._store = _FakeStore()
    r._reranker = _FakeReranker()
    return r, r._reranker


def test_comparison_skips_rerank():
    r, reranker = _routed()
    hits = r.retrieve("Which spent more on R&D, NVIDIA or Intel?", k=5)
    assert not reranker.called          # per-company interleave must be preserved
    assert len(hits) == 5


def test_simple_question_reranks():
    r, reranker = _routed()
    r.retrieve("What was Micron's total revenue in its most recent 10-K?", k=5)
    assert reranker.called              # rerank helps where it doesn't re-crowd


def test_multi_hop_reranks():
    r, reranker = _routed()
    r.retrieve("How did NVIDIA's revenue change across its last three 10-Q filings?", k=5)
    assert reranker.called
