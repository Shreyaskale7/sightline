"""BM25 (sparse/keyword) retrieval leg.

BM25 is the classic exact-match ranking function search engines used for decades: it scores a
page by how often the query's literal words appear in it (weighted so rare words count more,
and long pages don't win just by being long). No neural net, no embeddings.

Why keep it next to dense retrieval: the two fail differently. Dense embeddings capture
meaning but blur exact anchors — tickers, defined terms ("Intel 18A", "QCT"), GAAP line items.
BM25 nails exactly those. Hybrid = ask both, fuse the rankings (see fusion.py).

For ~1-2k pages, rank_bm25 in memory is plenty; the index rebuilds from SQLite in well under a
second at startup. Swap for Pyserini/OpenSearch only if the corpus outgrows RAM.
"""
from __future__ import annotations

import re
from typing import Iterable

from .text_baseline import Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately dumb — keep it deterministic and fast;
    if eval ever shows stemming/stopwords would help, prove it with a measured lift first."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 over page text. Build once from the store, query many times."""

    def __init__(self, pages: Iterable) -> None:
        from rank_bm25 import BM25Okapi

        self._pages = [p for p in pages if getattr(p, "text", "").strip()]
        if not self._pages:
            raise ValueError("BM25Retriever needs at least one non-empty page")
        self._bm25 = BM25Okapi([tokenize(p.text) for p in self._pages])

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(
                accession=self._pages[i].accession,
                page_no=self._pages[i].page_no,
                score=float(scores[i]),
                ticker=getattr(self._pages[i], "ticker", ""),
                form=getattr(self._pages[i], "form", ""),
            )
            for i in top
        ]
