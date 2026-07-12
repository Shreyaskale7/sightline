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
    boxes: list[list[float]] = []  # normalized [x0,y0,x1,y1] regions to highlight on the image


class Stage(BaseModel):
    name: str
    latency_ms: float = 0.0
    detail: dict = {}


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    trace: list[Stage] = []      # the rendered pipeline: every stage + timing
    latency_ms: float = 0.0


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
    import time

    from ..answerer import Answerer
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..observability import trace
    from ..retrieval.routed import RoutedRetriever
    from ..verify import verify

    t0 = time.perf_counter()
    with trace("query", question=req.question, k=req.k) as stages:
        # Champion retrieval config (measured Recall@5 0.603): the router picks the best-per-slice
        # config — decomposition for comparisons, chunk+rerank for everything else.
        with RoutedRetriever() as retriever:
            hits = retriever.retrieve(req.question, k=req.k)
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            with span("fetch_pages") as s:
                pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
                s["n_pages"] = len(pages)
            result = Answerer().answer(req.question, pages)
            with span("verify") as s:
                verdict = verify(result, {(p.accession, p.page_no) for p in pages})
                s["forced_abstain"] = verdict.forced_abstain
                s["dropped_citations"] = verdict.dropped_citations
            result = verdict.result

            # Region highlighting: locate the answer's salient figures on each cited page.
            from ..highlight import page_boxes, salient_terms

            with span("highlight") as s:
                terms = salient_terms(result.answer)
                page_by_id = {(p.accession, p.page_no): p for p in pages}
                citations = []
                for c in result.citations:
                    p = page_by_id.get((c.accession, c.page_no))
                    boxes = page_boxes(p.image_path, c.page_no, terms) if p else []
                    citations.append(Citation(
                        accession=c.accession, page_no=c.page_no,
                        image_url=f"/pages/{c.accession}/{c.page_no}", boxes=boxes))
                s["n_boxes"] = sum(len(c.boxes) for c in citations)

    trace_stages = [
        Stage(name=r["name"], latency_ms=r.get("latency_ms", 0.0),
              detail={k: v for k, v in r.items() if k not in ("name", "latency_ms")})
        for r in stages
    ]
    return QueryResponse(
        answer=result.answer, citations=citations, abstained=result.abstained,
        trace=trace_stages, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
