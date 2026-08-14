"""FastAPI app: /query pipeline, cited-page-image serving, landing + console pages.

The demo's core promise: every answer arrives with the ACTUAL page image it was cited from,
rendered inline with the answer's figures highlighted in place (see highlight.py) — you never
have to trust a citation, you can look at exactly where it came from. And the full pipeline
trace (router decision → retrieval → rerank → verify, with timings) rides along in the
response, so the demo shows its work.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..observability import span

app = FastAPI(title="Sightline", version="0.1.0")

_LANDING_HTML = Path(__file__).parent / "landing.html"
_APP_HTML = Path(__file__).parent / "app.html"

# Load the retriever (embedding + reranker models, Qdrant client) ONCE and reuse it — otherwise
# every request pays ~20s of model-load. Embedded Qdrant is single-client-per-path, so one
# shared instance is also the only correct option.
#
# Two separate locks (must not be one, or /query would deadlock calling _get_retriever while
# holding it):
#   _build_lock — guards single construction (double-checked). Construction is slow (~20s), and
#     assignment happens only after it finishes, so warmup and the first request would otherwise
#     BOTH try to build and collide on the Qdrant path lock.
#   _query_lock — serializes queries (the embedded DB + CPU models aren't built for concurrency).
_retriever = None
_build_lock = threading.Lock()
_query_lock = threading.Lock()


def _get_retriever():
    global _retriever
    if _retriever is None:
        with _build_lock:
            if _retriever is None:  # double-checked: only the first caller builds
                from ..retrieval.routed import RoutedRetriever

                _retriever = RoutedRetriever()
    return _retriever


def _warmup_target() -> None:
    try:
        # A real dummy query forces EVERY lazy model to load now (the embedder AND the
        # cross-encoder reranker, which only loads on first .rerank()) — otherwise the first
        # real user's query pays ~40s of reranker load instead of the ~5s warm compute.
        _get_retriever().retrieve("warmup: total revenue", k=5)
        print("[warmup] models hot")
    except Exception as e:  # missing index/models (e.g. CI) shouldn't crash a background thread
        print(f"[warmup] skipped: {e}")


@app.on_event("startup")
def _warmup() -> None:
    """Load the models in the background at boot so the first real query is warm (~5s), not
    cold (~27s incl. model load). Non-blocking: the server accepts connections immediately."""
    threading.Thread(target=_warmup_target, daemon=True).start()


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    # Search scope. "auto" (product default): if you've uploaded documents and your question
    # names no company in the sample corpus, search YOUR documents; otherwise the sample corpus.
    # "uploads" forces your documents; "corpus" forces the SEC sample set.
    scope: str = "auto"


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


class UploadedDoc(BaseModel):
    filename: str
    label: str = ""
    accession: str = ""
    pages: int = 0
    indexed: int = 0
    already_known: bool = False
    error: str = ""      # per-file: one bad PDF must not fail the whole batch


class UploadResponse(BaseModel):
    documents: list[UploadedDoc] = []
    total_pages: int = 0
    total_indexed: int = 0


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile]) -> UploadResponse:
    """Upload one or more PDFs: rasterize -> store -> index. Then they're askable like any filing.

    Accepts a batch so you can bring a whole set of reports, not one file at a time. Each file is
    handled independently — a rejected PDF reports its own error and the rest still land.
    Idempotent on content (same bytes -> same synthetic accession -> re-upload is a no-op).
    Serialized under the query lock: indexing shares the retriever's embedded-Qdrant client.
    """
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..ingest.upload import UploadError, ingest_upload, synthetic_accession

    payloads = [(f.filename or "document.pdf", await f.read()) for f in files]
    docs: list[UploadedDoc] = []

    with _query_lock, span("upload", n_files=len(payloads)) as s:
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            for filename, data in payloads:
                try:
                    acc = synthetic_accession(data) if data else ""
                    if acc and store.is_ingested(acc):
                        docs.append(UploadedDoc(filename=filename, accession=acc,
                                                already_known=True))
                        continue
                    filing, pages = ingest_upload(data, filename, store,
                                                  Path(settings.data_dir))
                    indexed = _get_retriever().index_pages(
                        list(store.iter_pages_for(filing.accession))
                    )
                    docs.append(UploadedDoc(
                        filename=filename, label=filing.ticker, accession=filing.accession,
                        pages=len(pages), indexed=indexed,
                    ))
                except UploadError as e:
                    docs.append(UploadedDoc(filename=filename, error=str(e)))
        s["accepted"] = sum(1 for d in docs if d.pages)
        s["rejected"] = sum(1 for d in docs if d.error)

    if docs and all(d.error for d in docs):  # nothing usable -> surface it as a client error
        raise HTTPException(status_code=400, detail=docs[0].error)
    return UploadResponse(
        documents=docs,
        total_pages=sum(d.pages for d in docs),
        total_indexed=sum(d.indexed for d in docs),
    )


@app.get("/pages/{accession}/{page_no}")
def page_image(accession: str, page_no: int) -> FileResponse:
    """Serve a cited page's PNG so the UI can show the evidence, not just claim it."""
    from ..config import settings
    from ..ingest.rasterize import ensure_page_image
    from ..ingest.store import MetadataStore

    with MetadataStore(settings.data_dir / "sightline.db") as store:
        page = store.get_page(accession, page_no)
    if page is None:
        raise HTTPException(status_code=404, detail="page not found")
    # Renders from the stored filing PDF on first request if the PNG isn't on disk — the
    # deployed image ships PDFs only (see ensure_page_image), so this is the normal path there.
    img = ensure_page_image(Path(page.image_path), page_no)
    if img is None:
        raise HTTPException(status_code=404, detail="page image unavailable")
    return FileResponse(img, media_type="image/png")


class CompareRequest(BaseModel):
    question: str                       # the figure to put in the column, e.g. "R&D spend"
    tickers: list[str] = []             # empty -> every company in the corpus (capped)
    k: int = 5


class CompareCell(BaseModel):
    ticker: str
    value: str = ""
    citations: list[Citation] = []
    abstained: bool = False
    error: str = ""


class CompareResponse(BaseModel):
    question: str
    rows: list[CompareCell] = []
    answered: int = 0
    truncated_to: int = 0
    latency_ms: float = 0.0


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest) -> CompareResponse:
    """Answer one question once PER COMPANY and return a table of individually-cited cells.

    The capability a chat window can't match: the corpus is far past any context window, and
    every cell is checkable on its own page rather than buried in prose. Costs one LLM call per
    company, so the company list is capped (compare.MAX_COMPANIES).
    """
    import time

    from ..compare import compare, resolve_companies
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..observability import trace

    t0 = time.perf_counter()
    with _query_lock, trace("compare", question=req.question) as stages:
        retriever = _get_retriever()
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            tickers, truncated = resolve_companies(
                req.question, store.list_tickers(), named=req.tickers
            )
            result = compare(req.question, tickers, retriever, store, k=req.k)

            rows = []
            for r in result.rows:
                rows.append(CompareCell(
                    ticker=r.ticker, value=r.value, abstained=r.abstained, error=r.error,
                    citations=[
                        Citation(accession=c.accession, page_no=c.page_no,
                                 image_url=f"/pages/{c.accession}/{c.page_no}")
                        for c in r.citations
                    ],
                ))
    _ = stages
    return CompareResponse(
        question=req.question, rows=rows, answered=result.answered,
        truncated_to=truncated, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


class ScreenRequest(BaseModel):
    criterion: str                      # e.g. "depends on TSMC for manufacturing"
    tickers: list[str] = []
    k: int = 5


class ScreenHit(BaseModel):
    ticker: str
    matched: bool = False
    evidence: str = ""
    citations: list[Citation] = []
    error: str = ""


class ScreenResponse(BaseModel):
    criterion: str
    rows: list[ScreenHit] = []
    matched: int = 0
    screened: int = 0
    latency_ms: float = 0.0


@app.post("/screen", response_model=ScreenResponse)
def screen_endpoint(req: ScreenRequest) -> ScreenResponse:
    """Ask a yes/no of every company and return the hits with their evidence.

    A corpus scan, not a lookup: answering it touches every filing, and each hit quotes and
    cites the sentence that justifies it — so the shortlist is auditable row by row.
    """
    import time

    from ..compare import resolve_companies, screen
    from ..config import settings
    from ..ingest.store import MetadataStore
    from ..observability import trace

    t0 = time.perf_counter()
    with _query_lock, trace("screen", criterion=req.criterion):
        retriever = _get_retriever()
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            tickers, _ = resolve_companies(
                req.criterion, store.list_tickers(), named=req.tickers
            )
            result = screen(req.criterion, tickers, retriever, store, k=req.k)
            rows = [
                ScreenHit(
                    ticker=r.ticker, matched=r.matched, evidence=r.evidence, error=r.error,
                    citations=[
                        Citation(accession=c.accession, page_no=c.page_no,
                                 image_url=f"/pages/{c.accession}/{c.page_no}")
                        for c in r.citations
                    ],
                )
                for r in result.rows
            ]
    return ScreenResponse(
        criterion=req.criterion, rows=rows, matched=len(result.matches),
        screened=len(rows), latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


class DiffRequest(BaseModel):
    topic: str                          # e.g. "total revenue for the quarter"
    ticker: str
    form: str = "10-Q"
    k: int = 4


class DiffResponse(BaseModel):
    ticker: str
    topic: str
    older: str = ""                     # human label, e.g. "10-Q filed 2025-11-19"
    newer: str = ""
    summary: str = ""
    citations: list[Citation] = []
    no_material_change: bool = False
    error: str = ""
    latency_ms: float = 0.0


@app.post("/diff", response_model=DiffResponse)
def diff_endpoint(req: DiffRequest) -> DiffResponse:
    """What changed on a topic between a company's two most recent filings of one form.

    Each side is retrieved pinned to its own accession, so the two periods stay genuinely
    separate, and every claim about change is verified against the union of both page sets.
    """
    import time

    from ..config import settings
    from ..diff import diff_filings
    from ..ingest.store import MetadataStore
    from ..observability import trace

    t0 = time.perf_counter()
    with _query_lock, trace("diff", ticker=req.ticker, topic=req.topic):
        retriever = _get_retriever()
        with MetadataStore(settings.data_dir / "sightline.db") as store:
            r = diff_filings(req.topic, req.ticker.upper(), retriever, store,
                             form=req.form, k=req.k)
    return DiffResponse(
        ticker=r.ticker, topic=r.topic,
        older=r.older.label if r.older else "", newer=r.newer.label if r.newer else "",
        summary=r.summary, no_material_change=r.no_material_change, error=r.error,
        citations=[
            Citation(accession=c.accession, page_no=c.page_no,
                     image_url=f"/pages/{c.accession}/{c.page_no}")
            for c in r.citations
        ],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def _resolve_scope(scope: str, question: str, data_dir: Path):
    """Turn a requested scope into a retrieval filter override (or None for the sample corpus).

    The product default ("auto") makes a user's own uploads the search scope: nobody comes to
    this platform to look up NVIDIA's revenue — they bring their own documents and want answers
    from THOSE. So if any upload exists and the question names no sample-corpus company, restrict
    retrieval to uploaded documents. Naming a company (or scope="corpus") searches the SEC set.
    """
    from ..ingest.store import MetadataStore
    from ..retrieval.filters import QueryFilters, parse_query_filters

    scope = (scope or "auto").lower()
    if scope == "corpus":
        return None
    if scope == "uploads":
        return QueryFilters(form="UPLOAD")
    # auto
    if parse_query_filters(question).tickers:
        return None  # explicitly about a sample company -> search the sample corpus
    with MetadataStore(data_dir / "sightline.db") as store:
        return QueryFilters(form="UPLOAD") if store.has_uploads() else None


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
    from ..verify import verify

    t0 = time.perf_counter()
    # Serialize: the shared retriever holds an embedded-Qdrant client + CPU models, not built
    # for concurrent access. A demo is low-QPS; correctness beats throughput here.
    with _query_lock, trace("query", question=req.question, k=req.k) as stages:
        # Champion retrieval config (measured Recall@5 0.603): the router picks the best-per-slice
        # config — decomposition for comparisons, chunk+rerank for everything else.
        retriever = _get_retriever()
        with span("scope") as s:
            override = _resolve_scope(req.scope, req.question, settings.data_dir)
            s["scope"] = req.scope
            s["searched"] = "uploads" if override is not None else "corpus"
        hits = retriever.retrieve(req.question, k=req.k, filter_override=override)
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
