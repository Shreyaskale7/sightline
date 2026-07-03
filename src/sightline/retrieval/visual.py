"""Visual page-image retriever (Milestone 2) — the differentiator.

Embeds each page IMAGE (not its text) with ColModernVBERT, a ~250M late-interaction visual
retriever that runs on CPU. "Late interaction" means a page becomes MANY vectors (one per
image patch) instead of one; a query becomes one vector per token; relevance is MaxSim —
each query token takes its best-matching patch score, summed. This preserves layout/table
structure that OCR-and-chunk destroys, and lets short queries hit the exact region of a page.

Qdrant supports this natively (multivector config with the MAX_SIM comparator), including in
embedded/local mode — same client API as the dense collection.

Cost note (why M2 is careful): hundreds of vectors per page is the real scaling cost of
late interaction. For ~1.3k pages this fits comfortably on disk; at 100k pages you'd add
token pooling / quantization / a two-stage retrieve (see docs). Measure before optimizing.

Interface mirrors TextRetriever (index/retrieve/count/close, same Hit type) so the eval
runner and RRF fusion treat all retrieval legs identically.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from ..config import settings
from .text_baseline import Hit

_MODEL_NAME = "ModernVBERT/colmodernvbert"
# The processor (tokenizer + image preprocessing) loads from the BASE repo: the adapter repo
# ships no config.json, which breaks offline processor resolution. Weights still come from
# the adapter on top of the base — only the processor artifacts differ in location.
_PROCESSOR_NAME = "ModernVBERT/colmodernvbert-base"
_ID_NS = uuid.UUID("00000000-0000-0000-0000-0000515111e1")  # distinct from the text namespace


def _point_id(accession: str, page_no: int) -> str:
    return str(uuid.uuid5(_ID_NS, f"{accession}#{page_no}"))


class VisualRetriever:
    def __init__(
        self,
        collection: str = "sightline_visual",
        qdrant_location: str | None = None,
        model_name: str = _MODEL_NAME,
        batch_size: int = 4,  # CPU-friendly
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self.batch_size = batch_size
        self._location = qdrant_location or str(Path(settings.data_dir) / "qdrant")
        self._client = None
        self._model = None
        self._processor = None

    # --- lazy heavy deps -----------------------------------------------------
    def _ensure_client(self) -> None:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self._location.startswith(("http://", "https://")):
                self._client = QdrantClient(url=self._location)
            else:
                Path(self._location).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=self._location)

    def _ensure_model(self) -> None:
        if self._model is None:
            import torch
            from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

            self._model = ColModernVBert.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            ).eval()
            self._processor = ColModernVBertProcessor.from_pretrained(_PROCESSOR_NAME)

    def _embed_images(self, image_paths: list[Path]) -> list[list[list[float]]]:
        """PNG paths -> one multivector (list of patch vectors) per page."""
        import torch
        from PIL import Image

        self._ensure_model()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        batch = self._processor.process_images(images)
        with torch.no_grad():
            embs = self._model(**batch)  # (batch, n_patches, dim)
        return [e.cpu().float().tolist() for e in embs]

    def _embed_query(self, text: str) -> list[list[float]]:
        """Query -> one vector per token (padding stripped via the attention mask)."""
        import torch

        self._ensure_model()
        batch = self._processor.process_queries([text])
        with torch.no_grad():
            emb = self._model(**batch)[0]  # (seq, dim)
        mask = batch["attention_mask"][0].bool()
        return emb[mask].cpu().float().tolist()

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import (
            Distance,
            MultiVectorComparator,
            MultiVectorConfig,
            VectorParams,
        )

        self._ensure_client()
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

    # --- API -----------------------------------------------------------------
    def index(self, pages: Iterable, progress_every: int = 25) -> int:
        """Embed page images and upsert. Idempotent on (accession, page_no).

        `pages` = objects with .accession, .page_no, .image_path (+ optional .ticker/.form),
        e.g. store.StoredPage. Batches keep CPU memory bounded; upserting per batch means an
        interrupted run resumes where it left off (already-upserted pages are just re-skipped
        by deterministic ids).
        """
        from qdrant_client.models import PointStruct

        pages = [p for p in pages if Path(p.image_path).exists()]
        if not pages:
            return 0

        done = 0
        for i in range(0, len(pages), self.batch_size):
            chunk = pages[i : i + self.batch_size]
            vectors = self._embed_images([Path(p.image_path) for p in chunk])
            if done == 0:
                self._ensure_collection(dim=len(vectors[0][0]))
            self._client.upsert(
                self.collection,
                points=[
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
                    for p, vec in zip(chunk, vectors)
                ],
            )
            done += len(chunk)
            if done % progress_every < self.batch_size:
                print(f"[visual-index] {done}/{len(pages)} pages", flush=True)
        return done

    def retrieve(self, query: str, k: int = 5, query_filter: object | None = None) -> list[Hit]:
        """Top-k pages by MaxSim; `query_filter` = optional Qdrant payload filter
        (retrieval/filters.py) so the visual leg gets the same metadata advantage as dense."""
        self._ensure_client()
        if not self._client.collection_exists(self.collection):
            return []
        qvec = self._embed_query(query)
        res = self._client.query_points(
            self.collection, query=qvec, limit=k, query_filter=query_filter
        ).points
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
        self._ensure_client()
        if not self._client.collection_exists(self.collection):
            return 0
        return self._client.count(self.collection).count

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "VisualRetriever":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
