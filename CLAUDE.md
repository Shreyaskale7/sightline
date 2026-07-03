# CLAUDE.md — context for Claude Code

You are helping build **Sightline**, a visual-first document-intelligence system over SEC filings.
Read this whole file before making changes. The full design lives in `docs/DESIGN_DOC.md` and
`docs/IMPLEMENTATION_PLAN.md` — consult them when you need detail.

## Who you're working with
A capable engineer who just finished their 2nd year of university. Solid Python, some prior RAG
exposure, learning the AI-engineering patterns as they build. Explain non-obvious choices briefly
as you go — the goal is for the human to *learn*, not just to receive working code. When you
introduce a new concept (late-interaction retrieval, RRF, LLM-as-judge, etc.), add a one- or
two-line comment or docstring explaining what it is and why it's here.

## What Sightline does (one sentence)
It answers analyst-grade questions about SEC filings (10-Ks, 10-Qs) by retrieving and reasoning over
**page images** — not OCR'd text — and returns answers with page-level citations, or abstains when
the corpus can't support an answer.

## The prime directive (do not violate)
**Build the ugly working version before the impressive one, and measure before you optimize.**
- The eval harness (Milestone 1) is built *before* any visual retrieval. Never skip it.
- Every "clever" upgrade must be justified by a measured lift on the eval set. If a change can't be
  measured, it doesn't ship.
- Do not jump ahead to later milestones. If asked to, note that earlier milestones aren't done and
  recommend finishing them first.
- Prefer a smaller, correct, fully-understood implementation over a larger one the human can't explain.

## Milestone roadmap and current status
- [x] **M0 — Scaffold**: repo structure, docker-compose, config, tracing hook, FastAPI stub, EDGAR
  client skeleton, eval-harness skeleton. (This scaffold delivers most of M0.)
- [x] **M1 — Text-only baseline + eval harness**: ✅ done 2026-07-03. 20 filings / 1,329 pages
  (NVDA/AMD/INTC/MU/QCOM 10-Ks + 10-Qs); 38-case golden set; retrieval R@5 0.313 dense /
  **0.505 dense+metadata-filter**; generation correctness 0.65, citation 0.43, abstention
  recall 0.80 — all on a $0 free-model stack. Full history in `docs/RESULTS.md`.
- [~] **M2 — Visual + hybrid retrieval**: ColModernVBERT retriever, RRF fusion, cross-encoder
  rerank, and the visual answer path are built and smoke-tested; full visual index + the
  ablation table are in progress. Bar to beat: R@5 0.505.
- [ ] **M3 — Agents + grounding**: LangGraph router → planner → answerer → verifier → synthesis;
  table-QA path; NER for citation anchoring; LLM-as-judge calibrated against human labels (Cohen's κ).
- [ ] **M4 — LLMOps**: cost/latency dashboard + money-shot metric, prompt registry, CI eval gate,
  semantic + embedding caches.
- [ ] **M5 — Serving + demo UI**: async FastAPI + SSE streaming; UI with inline cited page images and
  region highlighting; 3-minute Loom.
- [ ] **M6 — Stretch**: fine-tune reranker, self-growing eval set, multi-tenant corpora, filing diffs.

Update the checkboxes as milestones complete. Tag `v1.0` after M5 before touching M6.

## Architecture (target)
`question → router (easy/hard/tabular/out-of-domain) → retrieval (text prefilter → visual rerank on
top-N → RRF fusion → cross-encoder rerank) → answerer (VLM reads page images) → verifier (every claim
cited, else abstain) → synthesis`. A single `trace_id` flows through every stage into Langfuse.

## Tech stack (verify current versions at build time — the visual-retrieval space moves fast)
- Python 3.11+. Package manager: `uv` (fallback: pip + `requirements.txt`).
- Data: SEC EDGAR (free, no key). See constraints below.
- Vector DB: **Qdrant** (native multi-vector / late-interaction support), via `docker compose`.
- Visual retriever: **ColModernVBERT** (~250M, starter — runs on free/CPU) or **ColQwen2.5**, via
  `colpali-engine`. Check the ViDoRe v3 leaderboard before committing to a model.
- Text embeddings: a strong open dense model (BGE / e5 family). Sparse: BM25 (`rank_bm25` for now).
- Reranker: a cross-encoder (BGE-reranker family).
- VLM (answers): a **hosted** vision-LLM API early (cheap, no infra); self-host a small open VLM only
  in M4+ for the cost comparison. Never block M1 on self-hosting a big model.
- Orchestration: LangGraph. Eval: Ragas + the custom harness in `src/sightline/eval/`.
- Observability: Langfuse + OpenTelemetry (see `src/sightline/observability.py`).
- Serving: FastAPI (async, SSE in M5). PDF→image: `pdf2image` (needs poppler).

## Hard constraints
- **SEC EDGAR etiquette:** every request MUST send a `User-Agent` header with a real name + email
  (see `SEC_USER_AGENT` in `.env`). Stay under **10 requests/second** — the built-in rate limiter in
  `edgar.py` enforces this; do not remove it. Requests without a User-Agent get 403'd.
- **Cost:** keep a hard budget alert on any paid LLM API. Use a cheap/small model for the router and
  the eval judge; reserve the good VLM for final answers. Cache aggressively (M4).
- **Idempotency:** ingestion must be safe to re-run — key on filing accession number, never duplicate.
- **Answer path uses page images, not extracted text.** Extracted text is only for the BM25/dense
  prefilter side of hybrid retrieval. Don't let text leak into the final answer generation in M2+.
- **Abstention is a feature.** The verifier should force "I don't know" rather than let the model
  guess. Unanswerable questions are part of the eval set on purpose.

## Coding conventions
- Type hints everywhere. `ruff` for lint/format. `pytest` for tests. Small, composable modules.
- Config comes from `src/sightline/config.py` (pydantic-settings, reads `.env`). No hard-coded secrets
  or paths.
- Every pipeline stage emits a trace span (use the helper in `observability.py`).
- Keep functions pure where possible; push side effects (network, disk, model calls) to the edges.
- Write a test alongside new non-trivial logic. The EDGAR client and eval scorers especially.

## What NOT to do
- Don't skip or defer the eval harness.
- Don't build the full 5-agent graph before evals show you need it — router + verifier is the floor.
- Don't self-host a large VLM in M1.
- Don't OCR-and-chunk as the primary retrieval path — the whole point is visual retrieval.
- Don't add dependencies casually; prefer the stack above. Explain any new heavy dependency.
- Don't expand corpus size prematurely — late-interaction index size is the real scaling cost.

## Handy commands
- `make up` — start Qdrant + Langfuse + API via docker-compose.
- `make ingest` — run the EDGAR ingestion CLI (`scripts/ingest.py`).
- `make eval` — run the eval harness over the golden set.
- `make test` — run pytest.
- `make api` — run the FastAPI app locally.
