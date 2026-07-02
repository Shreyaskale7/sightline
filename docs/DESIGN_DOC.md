# Sightline — Visual-First Financial Filing Analyst

> A production-grade, OCR-free document-intelligence system that answers analyst-grade
> questions over SEC filings by retrieving and reasoning over **page images**, not
> extracted text. Built to demonstrate system design, evaluation maturity, and LLMOps —
> the three things that separate an AI engineer from an "API wrapper."

**Target role:** mid-level SWE (2–6 YOE) pivoting into AI engineering, 2026 job market.
**Scope:** single flagship, built to depth. No hard deadline — so this doc includes core (must-ship) and stretch (differentiators) explicitly marked.

---

## 1. The problem (and why it's the right one)

Financial filings — 10-Ks, 10-Qs, S-1s — are **visually dense**: nested tables, footnotes, multi-column layouts, charts. The standard portfolio project (OCR → text-chunk → embed → RAG) quietly fails on exactly the questions that matter, because OCR mangles table structure and drops chart data entirely.

Sightline skips OCR in the hot path. It encodes each **page image** with a late-interaction visual retriever (ColPali / ColQwen family), preserving layout and enabling questions like:

> *"What did NVIDIA say about supply constraints across its last three 10-Qs, and how did gross-margin guidance change quarter over quarter?"*

That is a multi-hop, cross-document, table-plus-prose question — the kind that breaks naive RAG and therefore justifies real system design.

**Why it stands out in 2026:** OCR-free visual retrieval is current frontier tooling most candidates haven't touched, the data (SEC EDGAR) is free/real/messy, and the domain is legible to a non-technical interviewer in one sentence.

---

## 2. What each HuggingFace task maps to

| System component | HF task | Competency demonstrated |
|---|---|---|
| Visual page retriever | Visual Document Retrieval, Feature Extraction | Late-interaction (MaxSim) embeddings; OCR-free retrieval |
| Text retriever + sparse | Sentence Similarity | Dense vs sparse tradeoffs, hybrid fusion |
| Reranker | Text Ranking | Cross-encoder reranking; optionally fine-tuning one |
| Answerer over page images | Image-Text-to-Text, Document Question Answering | VLM prompting, grounded citations, abstention |
| Numeric / tabular path | Table Question Answering | Routing around text-pipeline hallucination |
| Entity / citation anchoring | Token Classification (NER) | Grounding claims to a page; metadata filtering |
| Query router + guardrails | Zero-Shot / Text Classification | Cheap-classifier routing as a **cost** decision |
| Section digests | Summarization | Long-context section summarization |

---

## 3. Architecture

```
                         ┌─────────────────────────────────────────────┐
   User query ──────────▶│  ROUTER  (cheap zero-shot classifier)        │
                         │  → simple lookup | multi-hop | tabular | OOD  │
                         └───────┬───────────────────┬──────────────────┘
                                 │                   │
              simple lookup      │                   │  multi-hop / comparison
                                 ▼                   ▼
                        ┌────────────────┐   ┌──────────────────────────┐
                        │  one-shot path │   │  PLANNER (decompose into  │
                        │                │   │  sub-questions)           │
                        └───────┬────────┘   └───────────┬──────────────┘
                                │                        │ (per sub-question)
                                ▼                        ▼
              ┌───────────────────────────────────────────────────────────┐
              │                  HYBRID RETRIEVAL                           │
              │                                                             │
              │  ┌── text prefilter (BM25 + dense) → top-N pages ──┐        │
              │  │                                                 ▼        │
              │  │            visual rerank (ColQwen MaxSim) on top-N       │
              │  └────────────────────┬────────────────────────────┘       │
              │                       ▼                                     │
              │        Reciprocal Rank Fusion → cross-encoder rerank        │
              └───────────────────────┬─────────────────────────────────────┘
                                      ▼
                        ┌──────────────────────────────┐
                        │  ANSWERER (VLM over page      │
                        │  images) → answer + citations │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  VERIFIER: every claim backed │
                        │  by a cited page? else ABSTAIN│
                        │  / route to human review      │
                        └──────────────┬───────────────┘
                                       ▼
                              SYNTHESIS (merge sub-answers)
                                       │
   Every stage emits a span to a single trace_id ──▶ Langfuse / OTel
```

**The two-stage retrieve is the crux of both quality and cost:** a cheap text prefilter narrows thousands of pages to top-N (~50), and the expensive visual model reranks only those. This is what makes the "money-shot" cost number possible (see §7).

---

## 4. Data pipeline (SEC EDGAR)

EDGAR is free, no API key, rate-limited to ~10 req/s (set a descriptive `User-Agent`).

**Ingestion steps:**
1. **Discover filings** — `https://data.sec.gov/submissions/CIK{cik}.json` gives every filing for a company. Filter by form type (10-K, 10-Q, S-1) and date.
2. **Fetch documents** — pull the primary document (HTML or PDF) from the filing index.
3. **Render pages to images** — normalize everything to PDF, then rasterize each page to PNG (~150–200 DPI). Store page images + page-level metadata (CIK, ticker, form type, fiscal period, filing date, page number).
4. **Embed** —
   - *Visual:* ColQwen2 → multi-vector embeddings per page.
   - *Text:* extract text (for the prefilter/hybrid only, not the answer path) → chunk → dense embeddings (BGE/e5) + BM25 index.
5. **Index** — visual multi-vectors into Qdrant (native multivector / late-interaction support); text vectors + BM25 alongside.
6. **Incremental updates** — poll EDGAR's daily index; embed only new filings. Idempotent by accession number.

**Corpus scope (core):** ~15–20 companies (e.g., a sector like semiconductors — NVDA, AMD, INTC, TSM ADR, MU) × last 3 years. That's a few thousand pages: big enough to be real, small enough to iterate.

**Stretch:** multi-sector, multi-tenant (separate namespaces per user/corpus), and a "diff two filings" feature (period-over-period change detection).

---

## 5. Retrieval design (the technical centerpiece)

- **Sparse (BM25):** exact-match anchors — ticker symbols, defined terms, GAAP line items.
- **Dense text:** semantic recall over extracted text.
- **Visual (ColQwen2, MaxSim late interaction):** the differentiator — retrieves on layout + visual content, no OCR.
- **Fusion:** Reciprocal Rank Fusion over the three candidate lists.
- **Rerank:** cross-encoder (BGE-reranker) over the fused top-K.

**The ablation table is your portfolio centerpiece.** Build a hand-verified golden set of ~150 `(question → relevant page)` pairs, then publish real numbers:

| Config | Recall@5 | nDCG@10 | MRR |
|---|---|---|---|
| BM25 only | … | … | … |
| Dense text only | … | … | … |
| Visual only | … | … | … |
| Hybrid (RRF) | … | … | … |
| Hybrid + rerank | … | … | … |

**Stretch — fine-tune the reranker:** mine hard negatives from your logs and fine-tune a small cross-encoder on in-domain (filing) query/page pairs. Showing a *trained* component (not just off-the-shelf) is a strong senior signal.

**Scalability honesty:** late-interaction produces many vectors per page, so index size is the real cost. Mitigations to implement and *write about*: token pooling, binary/scalar quantization of the multi-vectors, and the two-stage retrieve so the visual model only ever scores top-N.

---

## 6. Agent orchestration (justified by cost, not hype)

Use **LangGraph** (explicit state graph — easy to trace and reason about). Agents earn their place only where decomposition pays for itself:

- **Router** — a cheap zero-shot/small-model classifier. Simple factual lookups take the one-shot path; only multi-hop/comparison queries pay for the planner. *Frame this as a latency/cost optimization — that's the tell that you've thought about production.*
- **Planner** — decomposes a multi-hop question into independent sub-questions, each retrieved separately.
- **Answerer** — VLM reads retrieved page images, answers with page-level citations.
- **Verifier** — checks that every claim is supported by a cited page; forces **abstention** or **human review** if not.
- **Synthesis** — merges sub-answers into one cited response.

> Red flag to avoid: "agents everywhere." Green flag: "agents where decomposition earns its cost." A router + verifier alone is already a defensible design.

---

## 7. Evaluation harness (highest-signal part — build it FIRST)

**Golden dataset:**
- Retrieval: ~150 hand-verified `(question → page)` pairs.
- Generation: gold answers + the specific page(s) that justify them.
- Adversarial slices: multi-hop, cross-company, chart-only answers, and **deliberately unanswerable** questions (to test abstention).

**Retrieval metrics:** Recall@k, nDCG@10, MRR (the ablation table).

**Generation metrics:**
- **Faithfulness / groundedness** — is each claim backed by retrieved context?
- **Answer correctness** — vs gold.
- **Citation accuracy** — does the cited page actually contain the fact?
- **Refusal calibration** — does it abstain when the corpus has no answer?

**LLM-as-judge, calibrated (the rare, impressive part):** don't trust a judge model blindly. Hand-label a subset of the judge's verdicts and report **judge–human agreement (Cohen's κ)**. Pick the judge prompt that maximizes agreement. Almost no portfolio does this.

**Self-growing eval set (the flywheel):** log every query; cluster the ones where the verifier abstained or citation-check failed; auto-draft new eval cases from those failures; route low-confidence auto-labels to human review (you). Write-up line: *"eval set grew from 150 hand-labeled → N cases mined from failure traces."*

**Tooling:** Ragas for standard metrics + a thin custom harness for the domain-specific ones. Optionally promptfoo for prompt-level regression.

---

## 8. LLMOps (make it look like a real system)

- **Tracing:** OpenTelemetry + **Langfuse**. A single `trace_id` flows through router → retrieval (with candidate scores) → rerank → agent tool calls → generation → verification.
- **Cost & latency dashboard, broken down by stage.** Ship the **money-shot metric**: *"visual retrieval costs \$X/query; naive send-every-page-to-a-VLM would cost \$Y — a Z% reduction."*
- **Prompt registry with versions;** eval results tied to prompt version.
- **CI eval gate (GitHub Actions):** run the eval set on every PR; **block merges that regress any metric past threshold.** Keep the git history where a regression *actually got caught* — that's a story, not a claim.
- **Caching:** semantic cache for repeated queries + embedding cache for ingestion.
- **Guardrails:** input classification (out-of-scope / injection) + output verification (the verifier agent).

---

## 9. Serving & frontend

- **Backend:** FastAPI, fully async, streaming responses (SSE). Endpoints: `/ingest`, `/query`, `/eval/run`, `/traces`.
- **Frontend (stretch, but recommended given no time limit):** Next.js or Streamlit. The killer demo feature: **render the answer with the cited page image inline, bounding-box or highlight the region the claim came from.** Interviewers remember visual citation grounding.
- **Packaging:** Docker Compose for local (Qdrant + API + UI + Langfuse). GPU work (embedding + optional local VLM) on Modal or RunPod; or hosted VLM APIs to keep infra light.

---

## 10. Tech stack (concrete)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem |
| Visual retriever | `colpali-engine` (ColQwen2) | Late-interaction, OCR-free |
| Text embeddings | BGE / e5 | Strong open dense retrievers |
| Sparse | BM25 (Pyserini or `rank_bm25`) | Exact-match anchors |
| Vector store | **Qdrant** | Native multivector/late-interaction support |
| Reranker | BGE-reranker (cross-encoder) | Off-the-shelf, fine-tunable |
| Generator (VLM) | Hosted (GPT-4o / Claude) + optional local Qwen2-VL | Show both for cost comparison |
| Orchestration | LangGraph | Explicit, traceable state graph |
| Eval | Ragas + custom harness | Standard + domain metrics |
| Observability | Langfuse + OpenTelemetry | Tracing, cost, prompt registry |
| Serving | FastAPI (async, SSE) | Production-grade Python serving |
| Frontend | Next.js or Streamlit | Demo + visual citation grounding |
| CI | GitHub Actions | Eval regression gate |
| Infra | Docker Compose; Modal/RunPod for GPU | Reproducible, cheap |

> Note: exact model names/versions move fast. Before you commit, check current SOTA on the ColPali/ColQwen leaderboard and the MTEB retrieval/reranking boards, and confirm Qdrant's current multivector API — pick the best-available at build time rather than what's written here.

---

## 11. Full roadmap (phase = one competency; no hard deadline)

**Phase 0 — Foundations.** Repo scaffold, Docker Compose, EDGAR ingestion for ~5 companies, page rasterization, tracing wired from commit #1. *Learn: reproducible AI infra, OTel tracing.*

**Phase 1 — Baseline + eval harness (do this before anything clever).** Text-only RAG baseline. Hand-label the 150-pair golden set. Build the eval harness and measure the baseline. *Learn: retrieval metrics, "measure the dumb baseline first" discipline.*

**Phase 2 — Visual + hybrid retrieval.** Add ColQwen visual retriever, RRF fusion, cross-encoder rerank, two-stage retrieve. Produce the ablation table. *Learn: late-interaction retrieval, hybrid fusion, index cost tradeoffs, proving lift with numbers.*

**Phase 3 — Agents + grounding.** Router → planner → answerer → verifier → synthesis in LangGraph. Table-QA path. Citation grounding. Calibrated LLM-as-judge (report κ). *Learn: multi-agent orchestration, hallucination/citation checking, judge calibration.*

**Phase 4 — LLMOps.** Cost dashboard + money-shot metric, semantic + embedding caches, prompt registry, CI eval gate that catches a real regression. *Learn: observability, cost accounting, CI-for-model-behavior.*

**Phase 5 — Serving + demo UI.** Async FastAPI with streaming; frontend with inline visual citation highlighting. Record a 3-minute Loom. *Learn: production serving, demo storytelling.*

**Phase 6 — Stretch differentiators (this is where "no time limit" pays off):**
- Fine-tune the reranker on mined hard negatives.
- Self-growing eval flywheel from failure traces.
- Multi-sector / multi-tenant corpora with namespace isolation.
- Period-over-period "diff two filings" feature.
- Quantized multi-vector index + a written cost/recall tradeoff analysis.
- A/B two VLM generators and report the quality/cost frontier.

---

## 12. What you learn (competency map)

By the end you can speak fluently, with your own numbers, about: late-interaction visual retrieval; hybrid search + RRF + cross-encoder reranking; building and calibrating an eval harness (retrieval + generation metrics, LLM-as-judge with human agreement); multi-agent orchestration justified by cost; hallucination/citation verification and abstention; LLMOps (tracing, cost dashboards, prompt versioning, CI regression gates, caching); and honest scalability tradeoffs (index size, quantization, two-stage retrieval). That set *is* the AI-engineer job description.

---

## 13. Portfolio deliverables (what you hand a hiring manager)

1. This design doc (architecture + tradeoffs) in the repo root.
2. The **ablation table** with real numbers.
3. A **rendered trace** of one hard multi-hop query (router decision → candidates → rerank → generation → verification).
4. The **cost-per-query money-shot** number.
5. The **regression-caught-in-CI** git history.
6. A **3-minute Loom** walkthrough (more persuasive than any README).
7. A README that leads with a number: *"a filing analyst hitting X% citation accuracy and Y% faithfulness on an N-question eval set, at \$Z/query — a W% reduction over naive VLM-over-every-page."*

---

## 14. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Scope creep (no deadline = infinite scope) | Ship Phases 0–5 as v1.0 *before* touching Phase 6. Tag the release. |
| Late-interaction index too large | Two-stage retrieve; quantize multi-vectors; cap corpus size in v1. |
| Building features before evals | Eval harness is Phase 1. Nothing "clever" ships without a measured lift. |
| Over-engineering the agent layer | Router + verifier is the floor; add planner only when eval shows multi-hop failures. |
| GPU cost | Batch embedding jobs; hosted VLM APIs for generation; local model optional. |
| Model/library churn | Pin versions; verify current SOTA at build time (§10 note). |

---

## 15. First week, concretely

1. `git init`, repo scaffold, Docker Compose (Qdrant + Langfuse + FastAPI stub).
2. EDGAR client: pull 10-Ks/10-Qs for 5 semiconductor companies, last 3 years.
3. Rasterize pages → PNG + metadata store.
4. Text-only baseline retriever + generator, end-to-end, traced.
5. Start hand-labeling the golden set (aim: 30 pairs in week 1, 150 by end of Phase 1).

Measure the baseline before you build anything visual. That discipline is itself the senior signal.
