"""FastAPI app: /query pipeline, cited-page-image serving, and a minimal demo page.

The demo's core promise (and the M5 killer feature in embryo): every answer arrives with the
ACTUAL page image it was cited from, rendered inline — you never have to trust a citation,
you can look at it. Region highlighting joins in M5 proper (ColPali patch attention).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..observability import span

app = FastAPI(title="Sightline", version="0.1.0")

_LANDING_HTML = Path(__file__).parent / "landing.html"
_APP_HTML = Path(__file__).parent / "app.html"


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class Citation(BaseModel):
    accession: str
    page_no: int
    image_url: str = ""  # where the UI can fetch the cited page's PNG


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Landing page: what Sightline is, with the real measured numbers."""
    return _LANDING_HTML.read_text(encoding="utf-8")


@app.get("/app", response_class=HTMLResponse)
def console() -> str:
    """The analyst console: ask a question, see the cited page images inline."""
    return _APP_HTML.read_text(encoding="utf-8")


@app.get("/pages/{accession}/{page_no}")
def page_image(accession: str, page_no: int) -> FileResponse:
    """Serve a cited page's PNG so the UI can show the evidence, not just claim it."""
    from ..config import settings
    from ..ingest.store import MetadataStore

    with MetadataStore(settings.data_dir / "sightline.db") as store:
        page = store.get_page(accession, page_no)
    if page is None or not Path(page.image_path).exists():
        raise HTTPException(status_code=404, detail="page not found")
    return FileResponse(page.image_path, media_type="image/png")


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
    from ..retrieval.routed import RoutedRetriever
    from ..verify import verify

    with span("query", question=req.question, k=req.k):
        # Champion retrieval config (measured Recall@5 0.603): the router picks the best-per-slice
        # config — decomposition for comparisons, chunk+rerank for everything else.
        with RoutedRetriever() as retriever:
            with span("retrieve", k=req.k):
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
            citations=[
                Citation(accession=c.accession, page_no=c.page_no,
                         image_url=f"/pages/{c.accession}/{c.page_no}")
                for c in result.citations
            ],
            abstained=result.abstained,
        )
