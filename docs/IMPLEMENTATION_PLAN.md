# Sightline — Implementation Plan (student edition)

Companion to `SIGHTLINE_DESIGN_DOC.md`. That doc is the *what/why*. This is the *how*, tuned for a capable 2nd-year with solid Python, some RAG exposure, on a near-zero budget. Read the design doc first; this fills in tools, data, accounts, setup, and a milestone-by-milestone build with learning resources.

> **Golden rule for this build:** build the ugly working version before the impressive one. A text-only pipeline you fully understand beats a half-working ColQwen setup you copied. Every "clever" upgrade must show a measured lift on your eval set — otherwise it doesn't ship.

---

## 0. Cost & compute reality (read this first)

You can build the core (Milestones 1–3) for **$0–30 total**. Here's how.

**Data:** SEC EDGAR is 100% free — no API key, no registration. Just a required `User-Agent` header (your name + email) and a 10 requests/second cap. That's your entire "data cost."

**Compute — pick per task:**
- **Free GPU:** Kaggle Notebooks is the most reliable free option — a guaranteed ~30 GPU-hours/week on a T4/P100 (16 GB VRAM), 9–12 hour sessions. Use this for embedding your corpus. Google Colab free works too but disconnects unpredictably; some universities give students free Colab Pro, so check yours.
- **Local (your laptop):** for a small corpus with a small retriever model, CPU is fine for development. Qdrant, FastAPI, and the whole orchestration layer run locally in Docker.
- **When you outgrow free:** Modal (serverless GPU, generous free credits, great for batch embedding jobs), Lightning AI (student credits), or RunPod/Vast.ai (A100 under ~$1/hr, pay per second). You likely won't need these until the stretch phases.

**LLM/VLM for generation:** don't self-host a big VLM early — it's the expensive, fiddly part. Use a hosted VLM API (any major provider's vision model) for the answer step and cap your spend with a hard budget alert at $10. For a small corpus and a few hundred eval queries, generation cost is a few dollars. Later, self-host a small open VLM (e.g. a Qwen-VL variant) on Kaggle/Modal to show you *can*, and to power your cost comparison.

**Efficiency habits that keep you free:** quantize embeddings, cap corpus size in v1, checkpoint long jobs (free sessions time out), and always shut down idle GPU sessions.

---

## 1. Accounts & tools to set up (all have free tiers)

| Purpose | Tool | Notes |
|---|---|---|
| Code host + CI | GitHub | CI eval gate runs in GitHub Actions (free minutes) |
| Free GPU | Kaggle (primary), Colab (backup) | Kaggle for embedding jobs |
| Vector DB | Qdrant (Docker, local) | Native multi-vector support for late-interaction |
| Tracing / cost / prompts | Langfuse (free cloud tier or self-host via Docker) | One `trace_id` through every stage |
| VLM generation | A hosted vision-LLM API | Set a $10 hard budget alert immediately |
| Optional serverless GPU | Modal | Free credits; use for batch embedding if Kaggle isn't enough |
| Env management | `uv` or `conda` + Docker | Reproducibility matters for the "engineer" signal |

Create the GitHub repo and a `docker-compose.yml` (Qdrant + Langfuse + your API) on day one. Wire tracing before you write features — retrofitting observability is miserable.

---

## 2. The data pipeline in detail (SEC EDGAR)

### 2.1 What to pull
Start narrow: **one sector, ~15 companies, last 3 years of 10-Ks and 10-Qs.** Semiconductors is a clean choice (NVDA, AMD, INTC, MU, QCOM, AVGO, TXN, and TSMC's SEC filings). That's a few thousand pages — real enough to be impressive, small enough to iterate on free compute.

### 2.2 Endpoints (all free, all require a `User-Agent: YourName your@email` header)
- **Company list / CIK lookup:** `https://www.sec.gov/files/company_tickers.json` — maps tickers → CIK. CIKs are zero-padded to 10 digits in API URLs.
- **All filings for a company:** `https://data.sec.gov/submissions/CIK##########.json` — every filing with form type, date, and accession number. Filter to `10-K` / `10-Q`.
- **The filing documents themselves:** live under `https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/` — grab the primary document (usually HTML, sometimes a PDF).
- **Structured financials (optional, for the table-QA path / eval ground truth):** `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` gives machine-readable XBRL facts (revenue, margins, etc.) — gold for building numeric eval cases you can verify automatically.
- **Bulk alternative:** `https://www.sec.gov/files/` has daily-updated `submissions.zip` and `companyfacts.zip` if you'd rather batch-download than hit the API repeatedly.

A maintained Python wrapper (`sec-edgar-api` on PyPI) handles the submissions/companyfacts endpoints and pagination if you don't want to write the HTTP layer yourself — but writing your own thin client with `requests` + a rate limiter is a good learning exercise and shows you understand the constraints.

### 2.3 Ingestion steps
1. Resolve tickers → CIKs from `company_tickers.json`.
2. For each CIK, pull submissions JSON, filter to 10-K/10-Q in your date range, collect accession numbers.
3. Download each primary document. Normalize everything to PDF (HTML → PDF via a headless renderer if needed).
4. **Rasterize each page to PNG** with `pdf2image` at ~150–200 DPI. This is the core move — the page *image* is what you retrieve and reason over.
5. Store, per page: the PNG, plus metadata (CIK, ticker, form type, fiscal period, filing date, accession no., page number). SQLite is plenty for metadata in v1.
6. Also extract page text (via the HTML or a text layer) — **only** for the text/BM25 side of hybrid retrieval, never for the answer path.
7. Make ingestion **idempotent** keyed on accession number, so re-runs don't duplicate.

---

## 3. Models & libraries (current as of mid-2026 — verify at build time)

The visual-retrieval space moves fast; check the **ViDoRe (v3) leaderboard** before committing.

| Role | Recommended | Why for a student |
|---|---|---|
| Visual retriever | **ColQwen2.5** (workhorse) or **ColModernVBERT** (~250M params) | ColModernVBERT is ~10× smaller, runs on free/CPU, near-SOTA — ideal starter. Upgrade to ColQwen2.5/3 later. |
| Library for it | `colpali-engine` | Standard tooling for the ColPali family |
| PDF → image | `pdf2image` (+ poppler) | Simple, reliable rasterization |
| Text embeddings | a strong open dense model (BGE / e5 family) | For the hybrid text side |
| Sparse | BM25 (`rank_bm25` for small corpus; Pyserini/OpenSearch later) | Exact-match anchors (tickers, line items) |
| Vector DB | **Qdrant** | Native multi-vector (late-interaction) support |
| Reranker | a cross-encoder reranker (BGE-reranker family) | Off-the-shelf first; fine-tune as a stretch goal |
| VLM (answers) | hosted vision-LLM API first; small open VLM (Qwen-VL) later | Keeps early cost near zero |
| Orchestration | LangGraph | Explicit, traceable agent graph |
| Eval | Ragas + a thin custom harness | Standard metrics + your domain ones |
| Serving | FastAPI (async, SSE streaming) | Production-grade Python serving |

> Late-interaction models emit *hundreds of vectors per page*, so index size is the real scaling cost. Mitigations you'll implement/write about: patch pooling, binary/scalar quantization, and the two-stage retrieve (cheap text prefilter → visual rerank on top-N only).

---

## 4. Milestone-by-milestone build

Each milestone has a **definition of done** and **what you learn**. Don't move on until DoD is met.

### Milestone 0 — Scaffold (a few days)
- GitHub repo, sensible package structure (`ingest/`, `retrieval/`, `agents/`, `eval/`, `api/`, `ui/`).
- `docker-compose.yml`: Qdrant + Langfuse + FastAPI stub.
- Tracing wired: a hello-world request produces a span in Langfuse.
- **DoD:** `docker compose up` runs; a dummy `/query` endpoint logs a trace.
- **Learn:** reproducible AI infra, OpenTelemetry/Langfuse tracing.

### Milestone 1 — Working text-only baseline + eval harness (the most important milestone)
- Ingest 5 companies. Rasterize + extract text.
- Text-only RAG: dense retrieval over page text → stuff top-k page text into a chat LLM → answer with page citations.
- **Build the eval harness now.** Hand-label ~30 `(question → correct page)` pairs to start (grow to 150). Add ~10 deliberately unanswerable questions.
- Compute Recall@k, nDCG@10, MRR for retrieval; faithfulness, answer-correctness, citation-accuracy, and abstention for generation.
- **DoD:** one command runs the eval set and prints a metrics table; baseline numbers recorded.
- **Learn:** retrieval metrics, "measure the dumb baseline before optimizing" discipline, LLM-as-judge basics.
- **Resources:** Ragas docs (metrics); the ColPali/ViDoRe papers for context (read *after* the baseline works, not before).

### Milestone 2 — Visual retrieval + hybrid (the differentiator)
- Embed page **images** with ColModernVBERT/ColQwen2.5 via `colpali-engine` (batch job on Kaggle). Store multi-vectors in Qdrant.
- Add BM25 + dense text; fuse the three lists with **Reciprocal Rank Fusion**; add a cross-encoder reranker.
- Implement the **two-stage retrieve**: text prefilter → top-N → visual rerank on those N only.
- Produce the **ablation table**: BM25 / dense / visual / hybrid / hybrid+rerank, with real Recall@k, nDCG, MRR.
- Switch the answer path to a **VLM reading the page images** (not text).
- **DoD:** ablation table shows a measurable lift from visual/hybrid over the text baseline; answers cite the correct page image.
- **Learn:** late-interaction retrieval, hybrid fusion, index cost/size tradeoffs, proving lift with numbers.

### Milestone 3 — Agents + grounding
- LangGraph graph: **router** (cheap classifier → simple vs multi-hop vs tabular vs out-of-domain) → **planner** (decompose multi-hop) → **answerer** (VLM) → **verifier** (every claim backed by a cited page, else abstain / flag for review) → **synthesis**.
- Add the **table-QA path** for numeric questions (verify against XBRL companyfacts where possible).
- Add NER (Token Classification) to anchor citations and enable metadata filters ("FY2025 only").
- **Calibrate your LLM-as-judge:** hand-label a sample of the judge's verdicts, report **Cohen's κ** (judge–human agreement), pick the judge prompt that maximizes it.
- **DoD:** a multi-hop cross-company question returns a correct, fully-cited answer; verifier forces abstention on the unanswerable set; κ reported.
- **Learn:** multi-agent orchestration justified by cost, hallucination/citation verification, judge calibration.

### Milestone 4 — LLMOps
- Cost + latency dashboard broken down by stage. Ship the **money-shot metric**: "$X/query with two-stage visual retrieval vs $Y for naive VLM-over-every-page — Z% cheaper."
- Prompt registry with versions; eval results tied to prompt version.
- **CI eval gate:** GitHub Action runs the eval set on every PR and **blocks merges that regress a metric past threshold.** Keep the commit where a regression actually got caught.
- Semantic cache (repeated queries) + embedding cache (ingestion).
- **DoD:** dashboard live; a PR that regresses a metric is blocked by CI (screenshot it).
- **Learn:** observability, cost accounting, CI-for-model-behavior.

### Milestone 5 — Serving + demo UI
- FastAPI async + SSE streaming. Endpoints: `/ingest`, `/query`, `/eval/run`, `/traces`.
- Frontend (Streamlit is fastest; Next.js if you want to show frontend chops). **Killer feature:** render the answer with the cited page image inline and **highlight the region** the claim came from (ColPali gives you patch-level attention you can visualize).
- Record a **3-minute Loom** walkthrough.
- **DoD:** a stranger can ask a question and see a cited, highlighted answer; Loom recorded.
- **Learn:** production serving, demo storytelling.

### Milestone 6 — Stretch (grow into these over time)
Fine-tune the reranker on mined hard negatives · self-growing eval set from failure traces · multi-sector / multi-tenant corpora · "diff two filings" period-over-period feature · quantized multi-vector index + written cost/recall analysis · A/B two VLMs and chart the quality/cost frontier.

---

## 5. Learning resources by concept (learn each the week you need it)

Don't front-load theory — you'll stall. Learn just-in-time:
- **RAG fundamentals & metrics:** Ragas documentation; any current "RAG evaluation" guide.
- **Late-interaction / ColPali:** the ColPali paper (arXiv 2407.01449) + the `colpali-engine` README + a ColQwen-on-Qdrant tutorial. Read *after* your text baseline works.
- **Hybrid search & RRF:** Qdrant's hybrid-search docs.
- **Reranking:** the cross-encoder reranker model card + a "why rerank" explainer.
- **Agents:** LangGraph docs (build the router + verifier first, add the planner only when evals demand it).
- **LLMOps / tracing:** Langfuse docs (tracing, prompt management, cost tracking).
- **Eval / LLM-as-judge calibration:** search current "LLM-as-judge Cohen's kappa agreement" write-ups; this is the rare, high-signal skill.

> Model names, leaderboards, and free-tier limits change fast. Before committing to a specific retriever or provider, check the current ViDoRe v3 leaderboard and the provider's current pricing/limits page.

---

## 6. Common pitfalls (that sink student projects)

1. **Skipping the baseline.** If you start with ColQwen you can't tell whether it helped. Build text-only first, measure, *then* improve.
2. **No eval set.** Without it, "it seems better" is your only evidence — worthless in an interview. The eval harness is Milestone 1, not an afterthought.
3. **Agent soup.** Five agents that flake beat nothing, but they lose to a solid router + verifier. Add agents only when evals show the failure they'd fix.
4. **Ignoring index size.** Late-interaction blows up storage. Cap corpus size in v1; quantize; use the two-stage retrieve.
5. **Burning API budget.** Set a $10 hard alert. Cache. Use a cheap model for the router and eval-judge, the good VLM only for final answers.
6. **Infinite scope (your specific risk — no deadline).** Tag a `v1.0` release after Milestone 5 *before* touching Milestone 6. Ship the coherent thing first.

---

## 7. First week, concretely
1. Create GitHub repo + `docker-compose.yml` (Qdrant + Langfuse + FastAPI stub). Verify a trace appears in Langfuse.
2. Write the EDGAR client (with `User-Agent` header + a 10 req/s rate limiter). Pull submissions for 5 semiconductor companies.
3. Download 10-K/10-Q primary docs; rasterize pages to PNG with `pdf2image`; store metadata in SQLite.
4. Stand up the text-only baseline end-to-end (retrieve page text → LLM answer → citation).
5. Hand-label your first 30 `(question → page)` eval pairs and run the harness against the baseline. Record the numbers.

Measure the baseline before you build anything visual. That discipline is the senior signal — and it's what turns this from "a student project" into "a system I designed and evaluated."
