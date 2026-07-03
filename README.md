# Sightline

A visual-first document-intelligence system over SEC filings. It answers analyst-grade questions
about 10-Ks and 10-Qs by retrieving and reasoning over **page images** (OCR-free, layout-preserving)
and returns answers with page-level citations — or honestly abstains when the corpus can't support one.

> New here? Read `docs/DESIGN_DOC.md` (the what/why) and `docs/IMPLEMENTATION_PLAN.md` (the how).
> If you're using Claude Code, `CLAUDE.md` has the full context and build order.

## Why this exists
Financial filings are visually dense — nested tables, footnotes, charts. OCR-and-chunk pipelines
mangle exactly those. Sightline treats each page as an image and retrieves on layout + content using
late-interaction visual retrieval (the ColPali / ColQwen family). The differentiator over a typical
RAG project is the **evaluation harness**: everything is measured, and every improvement is proven.

## Status
**M1 (text baseline + eval harness): built and measured.** Corpus: 20 SEC filings / 1,329 page
images across NVDA, AMD, INTC, MU, QCOM. Golden set: 38 hand-verified cases. Text-only dense
retrieval baseline: **Recall@5 0.313** — with two measured negative findings along the way
(BM25 hurts on table-heavy gold pages; the text baseline degrades 0.567→0.313 as the corpus
grows 2.6×). Generation (cited answers + abstention) runs on a $0 free-model stack with a
response cache. **M2 (visual retrieval) in progress:** ColModernVBERT late-interaction
retriever + cross-encoder reranker scaffolded; the six-way ablation is next.
Full numbers and history: `docs/RESULTS.md`.

## Quickstart
```bash
# 1. Python env (uv recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# (or: pip install -e ".[dev]")

# 2. Install the headless browser used to render EDGAR HTML -> PDF
#   (PyMuPDF handles PDF -> PNG + text, so no poppler needed)
playwright install chromium

# 3. Config
cp .env.example .env
# edit .env — set SEC_USER_AGENT to "Your Name your@email.com" (required by the SEC)

# 4. Infra (Qdrant + Langfuse)
docker compose up -d

# 5. Try the ingestion CLI (pulls a couple of filings)
python scripts/ingest.py --tickers NVDA AMD --forms 10-K --limit 1

# 6. Run the API and the eval harness
make api
make eval
```

## Layout
```
src/sightline/
  config.py          # settings (pydantic-settings, reads .env)
  observability.py   # tracing helper (Langfuse if configured, else no-op)
  ingest/            # EDGAR client + PDF→image rasterization
  retrieval/         # text baseline now; visual + hybrid later
  agents/            # LangGraph router/planner/answerer/verifier (M3)
  eval/              # eval harness + golden set
  api/               # FastAPI app
scripts/             # CLI entry points
tests/               # pytest
docs/                # design doc + implementation plan
.github/workflows/   # CI eval gate
```

## The one rule
Measure the baseline before you build anything clever. That discipline is what turns this from a
student project into a system you designed and evaluated.
