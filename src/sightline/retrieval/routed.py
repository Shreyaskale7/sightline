"""Routed retrieval: pick the retrieval config that the ablation proved best PER question type.

The M2 ablation surfaced a real tension no single config resolves:
  - `grand` (chunk + filter + decompose + rerank) tops basic/multi_hop (0.656 / 0.333) but the
    reranker re-crowds the candidate pool and craters cross_company back to 0.125.
  - `champion` (chunk + filter + decompose, NO rerank) keeps cross_company's per-company
    guarantee (0.375) but trails elsewhere.

So route by question shape (the same router that hits 100% on the golden slices):
  - COMPARISON  -> decomposition WITHOUT rerank  (protect the per-company interleave)
  - everything  -> grand (rerank)                (max precision where rerank helps)

This is "agents where decomposition earns its cost" made literal — the router is a cost/quality
switch backed by measured per-slice numbers, not a guess. It shares one Qdrant client and one
reranker across both paths so the API pays the model-load cost once.
"""
from __future__ import annotations

from pathlib import Path

from ..agents.router import Route, route
from ..config import settings
from ..observability import span
from .decompose import decomposed_retrieve
from .text_baseline import Hit, TextRetriever


class RoutedRetriever:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        from ..ingest.store import MetadataStore
        from .rerank import Reranker

        qdrant_path = str(Path(settings.data_dir) / "qdrant")
        try:
            self._client = QdrantClient(path=qdrant_path)
        except RuntimeError as e:
            if "already accessed" in str(e):
                # Embedded Qdrant is one-process-per-path. The usual cause: a demo server
                # (make api / uvicorn) is running. Say so plainly instead of a portalocker dump.
                raise SystemExit(
                    "The Sightline database is already in use by another process — usually a "
                    "running server (`make api` / uvicorn). Stop it and try again.\n"
                    "  Windows:  Get-Process python | Stop-Process -Force\n"
                    "  macOS/Linux:  pkill -f 'uvicorn sightline'"
                ) from e
            raise
        self._chunked = TextRetriever(chunked=True, client=self._client)
        self._store = MetadataStore(settings.data_dir / "sightline.db")
        self._reranker = Reranker()

    def _rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        pairs = [
            (h, (p.text if (p := self._store.get_page(h.accession, h.page_no)) else ""))
            for h in hits
        ]
        return self._reranker.rerank(query, pairs, k=k)

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        with span("route") as s:
            decision = route(query)
            s["decision"] = decision.route.value
            s["reason"] = decision.reason
            s["tickers"] = decision.tickers
        # Fan out per company/filing regardless — decomposition never hurts recall of candidates.
        with span("retrieve.candidates") as s:
            candidates = decomposed_retrieve(
                query, 20, self._chunked.retrieve, self._store.list_accessions
            )
            s["n_candidates"] = len(candidates)
        if decision.route is Route.COMPARISON:
            # Reranking would re-crowd one company out; keep the interleaved order, take top-k.
            with span("rerank") as s:
                s["skipped"] = "comparison — preserving per-company interleave"
            return candidates[:k]
        with span("rerank") as s:
            out = self._rerank(query, candidates, k)
            s["reranked_to"] = len(out)
        return out

    def index_pages(self, pages: list) -> int:
        """Index freshly ingested pages (e.g. an upload) into the chunked collection.

        Lives on the routed retriever because embedded Qdrant allows ONE client per path —
        any indexing at serving time must reuse this instance's client, never open its own.
        """
        return self._chunked.index(pages)

    def close(self) -> None:
        self._chunked.close()
        self._store.close()
        self._client.close()

    def __enter__(self) -> "RoutedRetriever":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
