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
    """Placeholder pipeline. Wire the real stages as you build M1->M3.

    Target flow: router -> retrieve -> answer (VLM over page images) -> verify -> respond.
    Each stage should run inside a `span(...)` so the whole request is one trace.
    """
    with span("query", question=req.question, k=req.k) as s:
        # TODO(M1): retrieve = TextRetriever().retrieve(req.question, k=req.k)
        # TODO(M2): answer over the retrieved PAGE IMAGES with the VLM
        # TODO(M3): verify every claim is cited; abstain if not
        s["stub"] = True
        return QueryResponse(
            answer="Not implemented yet — wire the pipeline stages (see CLAUDE.md milestones).",
            citations=[],
            abstained=True,
        )
