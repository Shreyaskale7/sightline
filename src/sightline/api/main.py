"""FastAPI app. Minimal now; grows into the async + SSE serving layer in M5."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..observability import span

app = FastAPI(title="Sightline", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class Citation(BaseModel):
    accession: str
    page_no: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Pipeline: retrieve top-k pages -> answer from their text -> verify citations.

    The verifier (M3 floor) guarantees no uncited or falsely-cited answer leaves the API.
    M2 switches the answer step to page IMAGES once the ablation proves the visual index.
    Each stage runs inside a `span(...)` so the whole request is one trace.
    """
    from ..answerer import Answerer
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..retrieval.text_baseline import TextRetriever
    from ..verify import verify

    with span("query", question=req.question, k=req.k):
        with span("retrieve", k=req.k), TextRetriever() as retriever:
            hits = retriever.retrieve(req.question, k=req.k)
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
        result = Answerer().answer(req.question, pages)
        with span("verify") as s:
            verdict = verify(result, {(p.accession, p.page_no) for p in pages})
            s["forced_abstain"] = verdict.forced_abstain
            s["dropped_citations"] = verdict.dropped_citations
        result = verdict.result
        return QueryResponse(
            answer=result.answer,
            citations=[Citation(accession=c.accession, page_no=c.page_no) for c in result.citations],
            abstained=result.abstained,
        )
