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
    """M1 pipeline: retrieve top-k pages -> Claude answers from their text, with citations.

    M2 switches the answer step to page IMAGES; M3 adds router + verifier. Each stage runs
    inside a `span(...)` so the whole request is one trace.
    """
    from ..answerer import Answerer
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..retrieval.text_baseline import TextRetriever

    with span("query", question=req.question, k=req.k):
        with span("retrieve", k=req.k), TextRetriever() as retriever:
            hits = retriever.retrieve(req.question, k=req.k)
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
        result = Answerer().answer(req.question, pages)
        return QueryResponse(
            answer=result.answer,
            citations=[Citation(accession=c.accession, page_no=c.page_no) for c in result.citations],
            abstained=result.abstained,
        )
