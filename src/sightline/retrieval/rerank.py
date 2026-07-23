"""Cross-encoder reranker (Milestone 2, final retrieval stage).

Bi-encoders (our dense/visual legs) embed the query and the page SEPARATELY and compare
vectors — fast enough to scan the whole corpus, but they can't model fine interactions
between query and page. A cross-encoder reads (query, page_text) as ONE input and outputs a
single relevance score — much more accurate, and far too slow to run corpus-wide.

Hence the standard two-stage pattern: retrievers (fused via RRF) nominate a small candidate
set (~20); the cross-encoder re-orders just those. Wide net first, careful judge second —
the same cheap-then-expensive shape as the router/VLM split on the generation side.

Model: BGE-reranker-base (~280M, runs on CPU for 20 pairs in ~a second). Fine-tuning it on
mined hard negatives is the M6 stretch goal.
"""
from __future__ import annotations

from typing import Sequence

from .text_baseline import Hit

_MODEL_NAME = "BAAI/bge-reranker-base"
# Score on the page's first ~2k chars. Measured: shrinking this (1500 or 1000 chars) dropped
# Recall@5 0.731→0.690 for ~40% less rerank time — the reranker needs the full page to order
# financial pages correctly, so quality wins. Efficiency is bought elsewhere (statement fast-path).
_MAX_CHARS = 2000


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Cross-encoder relevance scores aligned to the input order (empty text -> -inf).

        Exposes the raw scores (rather than a sorted list) so callers can FUSE the cross-encoder
        signal with the dense first-stage rank instead of trusting it outright — the cross-encoder
        reads only the page head (_MAX_CHARS) and sometimes demotes a page whose relevant sentence
        the dense retriever found deeper in the page."""
        if not texts:
            return []
        self._ensure()
        idx = [i for i, t in enumerate(texts) if t.strip()]
        out = [float("-inf")] * len(texts)
        if idx:
            preds = self._model.predict([(query, texts[i][:_MAX_CHARS]) for i in idx])
            for i, s in zip(idx, preds):
                out[i] = float(s)
        return out

    def rerank(self, query: str, hits_with_text: Sequence[tuple[Hit, str]], k: int = 5) -> list[Hit]:
        """Re-order candidate hits by cross-encoder score; return the top-k.

        Caller supplies (hit, page_text) pairs — the reranker stays storage-agnostic and
        trivially testable. Hits with empty text keep their place at the bottom.
        """
        if not hits_with_text:
            return []
        self._ensure()
        scored = [(h, t) for h, t in hits_with_text if t.strip()]
        unscored = [h for h, t in hits_with_text if not t.strip()]
        if scored:
            scores = self._model.predict([(query, t[:_MAX_CHARS]) for _, t in scored])
            ranked = [
                Hit(accession=h.accession, page_no=h.page_no, score=float(s),
                    ticker=h.ticker, form=h.form)
                for (h, _), s in sorted(zip(scored, scores), key=lambda x: x[1], reverse=True)
            ]
        else:
            ranked = []
        return (ranked + unscored)[:k]
