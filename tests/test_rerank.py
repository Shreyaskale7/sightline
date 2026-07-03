"""Reranker tests with a fake cross-encoder — the ordering logic is ours, the scores aren't."""
from __future__ import annotations

from sightline.retrieval.rerank import Reranker
from sightline.retrieval.text_baseline import Hit


class _FakeCrossEncoder:
    """Scores by how many query words appear in the passage — deterministic and obvious."""

    def predict(self, pairs):
        return [sum(w in text.lower() for w in q.lower().split()) for q, text in pairs]


def _reranker() -> Reranker:
    r = Reranker()
    r._model = _FakeCrossEncoder()
    return r


def test_reranker_reorders_by_score():
    hits = [
        (Hit("a", 1, 0.9), "nothing relevant here"),
        (Hit("a", 2, 0.5), "total revenue for the fiscal year"),
    ]
    ranked = _reranker().rerank("total revenue", hits, k=2)
    assert (ranked[0].accession, ranked[0].page_no) == ("a", 2)  # beat the higher fused score


def test_reranker_respects_k():
    hits = [(Hit("a", i, 1.0), f"text {i}") for i in range(1, 6)]
    assert len(_reranker().rerank("text", hits, k=3)) == 3


def test_empty_text_sinks_to_bottom():
    hits = [(Hit("a", 1, 0.9), "   "), (Hit("a", 2, 0.1), "revenue")]
    ranked = _reranker().rerank("revenue", hits, k=2)
    assert (ranked[0].page_no, ranked[1].page_no) == (2, 1)


def test_empty_input():
    assert _reranker().rerank("q", [], k=5) == []
