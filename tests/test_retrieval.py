"""Tests for the BM25 leg and RRF fusion — the scorers of the M2 ablation must be trustworthy."""
from __future__ import annotations

from dataclasses import dataclass

from sightline.retrieval.bm25 import BM25Retriever, tokenize
from sightline.retrieval.fusion import rrf
from sightline.retrieval.text_baseline import Hit


@dataclass
class _Page:
    accession: str
    page_no: int
    text: str
    ticker: str = "T"
    form: str = "10-K"


def test_tokenize_lowercases_and_splits():
    assert tokenize("Intel 18A ramped; R&D $13,774M!") == [
        "intel", "18a", "ramped", "r", "d", "13", "774m",
    ]


def test_bm25_prefers_exact_term_match():
    pages = [
        _Page("a", 1, "revenue increased due to data center demand"),
        _Page("a", 2, "Intel 18A process technology reached high-volume production"),
        _Page("a", 3, "employees and human capital disclosures"),
    ]
    hits = BM25Retriever(pages).retrieve("Intel 18A process", k=1)
    assert (hits[0].accession, hits[0].page_no) == ("a", 2)


def test_bm25_skips_empty_pages():
    pages = [_Page("a", 1, "   "), _Page("a", 2, "real text here")]
    r = BM25Retriever(pages)
    assert len(r.retrieve("real text", k=5)) == 1


def test_rrf_rewards_agreement():
    """A page both retrievers rank mid-list should beat pages only one list contains."""
    list_a = [Hit("x", 1, 9.0), Hit("shared", 5, 8.0), Hit("x", 2, 7.0)]
    list_b = [Hit("y", 1, 0.9), Hit("shared", 5, 0.8), Hit("y", 2, 0.7)]
    fused = rrf(list_a, list_b)
    assert (fused[0].accession, fused[0].page_no) == ("shared", 5)


def test_rrf_top_n_and_score_order():
    list_a = [Hit("a", i, 1.0) for i in range(1, 6)]
    fused = rrf(list_a, top_n=3)
    assert len(fused) == 3
    assert fused[0].score >= fused[1].score >= fused[2].score


def test_rrf_preserves_metadata():
    fused = rrf([Hit("a", 1, 1.0, ticker="NVDA", form="10-K")])
    assert fused[0].ticker == "NVDA" and fused[0].form == "10-K"
