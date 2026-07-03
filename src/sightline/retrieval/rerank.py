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
_MAX_CHARS = 2000  # score on the page's first ~2k chars; plenty for relevance, caps latency


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

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
