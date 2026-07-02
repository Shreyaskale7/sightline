"""Text-only baseline retriever (Milestone 1).

Build THIS first and measure it, before any visual retrieval. It gives you the number that
everything in M2 must beat. Keep it deliberately simple.

Design for M1:
  - Embed each page's extracted text with a small dense model (BGE) and store the vectors in
    Qdrant. At query time, embed the question and return the top-k nearest pages by cosine.
  - Later (M2) this becomes one leg of hybrid retrieval (dense text + BM25 + visual, fused
    via Reciprocal Rank Fusion).

Two pragmatic, budget-driven choices (both easy to swap later):
  - Embeddings via **fastembed** (ONNX) instead of sentence-transformers: same BGE model, but
    no torch dependency, so it stays light and fast on a laptop CPU. torch arrives in M2 for
    the visual retriever anyway.
  - Qdrant in **embedded/local mode** (a path on disk) when no server is running. It's the
    identical qdrant-client API; point `qdrant_location` at http://localhost:6333 once you
    `docker compose up qdrant`.

BGE is an *asymmetric* retriever: the query gets a short instruction prefix that the passages
don't. We use fastembed's `query_embed` for that (falling back to a manual prefix).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import settings

_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, strong for its size, CPU-friendly
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Stable namespace so a page's point id is deterministic -> re-indexing is idempotent.
_ID_NS = uuid.UUID("00000000-0000-0000-0000-0000515111e0")  # "sightline"-ish, arbitrary but fixed


@dataclass
class Hit:
    accession: str
    page_no: int
    score: float
    ticker: str = ""
    form: str = ""


def _point_id(accession: str, page_no: int) -> str:
    """Deterministic UUID for a page, so upserts overwrite rather than duplicate."""
    return str(uuid.uuid5(_ID_NS, f"{accession}#{page_no}"))


class TextRetriever:
    def __init__(
        self,
        collection: str = "sightline_text",
        qdrant_location: str | None = None,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        # Default to a local on-disk Qdrant next to the rest of the data.
        self._location = qdrant_location or str(Path(settings.data_dir) / "qdrant")
        self._client = None
        self._model = None
        self._dim: int | None = None

    # --- lazy heavy deps -----------------------------------------------------
    def _ensure(self) -> None:
        if self._client is not None:
            return
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient

        self._model = TextEmbedding(self.model_name)
        # A URL means a running server; anything else is treated as a local path (embedded).
        if self._location.startswith(("http://", "https://")):
            self._client = QdrantClient(url=self._location)
        else:
            Path(self._location).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=self._location)

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        return [list(v) for v in self._model.embed(texts)]

    def _embed_query(self, text: str) -> list[float]:
        self._ensure()
        try:
            return list(next(iter(self._model.query_embed(text))))
        except AttributeError:  # older fastembed: apply the BGE query instruction manually
            return list(next(iter(self._model.embed([_BGE_QUERY_PREFIX + text]))))

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    # --- API -----------------------------------------------------------------
    def index(self, pages: Iterable) -> int:
        """Embed page text and upsert into Qdrant. Idempotent on (accession, page_no).

        `pages` is any iterable of objects with .accession, .page_no, .text, and optionally
        .ticker / .form (e.g. store.StoredPage). Empty-text pages are skipped.
        """
        from qdrant_client.models import PointStruct

        self._ensure()
        pages = [p for p in pages if getattr(p, "text", "").strip()]
        if not pages:
            return 0

        vectors = self._embed_passages([p.text for p in pages])
        self._dim = len(vectors[0])
        self._ensure_collection(self._dim)

        points = [
            PointStruct(
                id=_point_id(p.accession, p.page_no),
                vector=vec,
                payload={
                    "accession": p.accession,
                    "page_no": p.page_no,
                    "ticker": getattr(p, "ticker", ""),
                    "form": getattr(p, "form", ""),
                },
            )
            for p, vec in zip(pages, vectors)
        ]
        self._client.upsert(self.collection, points=points)
        return len(points)

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        """Return the top-k pages for a query, best first."""
        self._ensure()
        if not self._client.collection_exists(self.collection):
            return []
        qvec = self._embed_query(query)
        res = self._client.query_points(self.collection, query=qvec, limit=k).points
        return [
            Hit(
                accession=p.payload["accession"],
                page_no=int(p.payload["page_no"]),
                score=float(p.score),
                ticker=p.payload.get("ticker", ""),
                form=p.payload.get("form", ""),
            )
            for p in res
        ]

    def count(self) -> int:
        self._ensure()
        if not self._client.collection_exists(self.collection):
            return 0
        return self._client.count(self.collection).count

    def close(self) -> None:
        """Release the Qdrant client. In embedded mode this frees the on-disk file lock;
        closing explicitly also avoids a noisy __del__ traceback at interpreter shutdown."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "TextRetriever":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
