# Learning Sightline — from "I ran it" to "I built it"

This project was scaffolded fast. This file is the plan to *own* it: every component mapped to
the concept behind it, the best resource to learn that concept, and where (if anywhere) the
**IBM AI Engineering** specialization covers it. Work top to bottom. Check the boxes.

## The rule that makes this work

> **Rebuild each piece yourself. Do not let an AI write the code.** Use AI only as a tutor —
> "explain a cross-encoder", "review my function" — never as the author.

The loop for every item below: **read the concept → close the tab → write it from scratch →
diff against the file here → explain out loud why they differ.** If you can't explain a line,
you don't own it yet.

Two tracks, run in parallel:
- **Understand** (2–4 weeks part-time) — read one file/day + its concept. This is what
  interviews test. Do this first.
- **Rebuild** (3–5 months part-time) — reconstruct it milestone by milestone. Real mastery.

IBM course note: the *front half* (ML with Python, SVMs, Keras/TF, PyTorch, CNNs) is
**foundations** — valuable, but it won't look like Sightline. The parts that overlap this
project (transformers, LLMs, RAG, agents) are the *later* courses. Don't measure your Sightline
understanding against the early courses.

---

## Phase 0 — Python the codebase actually uses
Before the AI stuff, the patterns that appear everywhere here.

- [ ] **Type hints** — every function signature. → *Real Python: type checking* · `mypy` docs
- [ ] **`@dataclass`** — `Filing`, `Hit`, `Page`, `EvalCase`. → Python docs: `dataclasses`
- [ ] **Context managers (`with`)** — `MetadataStore`, retrievers, `span()`. → *Real Python: context managers*
- [ ] **Generators / `Iterator`** — `store.iter_pages()`. → *Fluent Python* ch. on iterables
- [ ] **`pydantic` models** — `config.py`, the API request/response models. → pydantic docs (v2)
- **Watch:** ArjanCodes (YouTube) for clean-Python patterns.
- **IBM:** partially (Course 1 uses Python for ML) · **Est: 1–2 weeks**

## Phase 1 — Ingestion: getting documents in
Files: `ingest/edgar.py`, `ingest/rasterize.py`, `ingest/store.py`, `ingest/pipeline.py`

- [ ] **HTTP clients + rate limiting** — why `edgar.py` throttles to 8 req/s. → `httpx` docs · HTTP basics (MDN)
- [ ] **PDF → image rasterization** — `rasterize.py` (PyMuPDF). → PyMuPDF docs
- [ ] **SQLite + idempotency** — `store.py`, keying on accession number. → `sqlite3` docs · "what is idempotency"
- **Do:** rebuild ingestion for ONE filing, no AI writing it. Watch `data/pages/` fill.
- **IBM:** not covered (this is data engineering, not ML) · **Est: 1 week**

## Phase 2 — Retrieval: the heart of the system (60% of the project)
Files: `retrieval/text_baseline.py`, `bm25.py`, `fusion.py`, `rerank.py`, `filters.py`

- [ ] **What an embedding IS** — vectors as "coordinates of meaning". → **Jay Alammar's illustrated blog** · Karpathy "Zero to Hero" (for the deep version)
- [ ] **Dense retrieval + cosine similarity** — `text_baseline.py` (BGE via fastembed). → *Hands-On Large Language Models* (Alammar & Grootendorst) — **the book for this project**
- [ ] **Vector databases** — Qdrant: collections, upsert, search. → Qdrant docs + tutorials
- [ ] **BM25 / sparse retrieval** — `bm25.py`, why keyword search still matters. → *Intro to Information Retrieval* (Manning, free online) ch. 6
- [ ] **Reciprocal Rank Fusion** — `fusion.py`, combining ranked lists. → the RRF paper (Cormack 2009) — it's 2 pages
- [ ] **Cross-encoder reranking** — `rerank.py`, why it beats bi-encoders. → Sentence-Transformers: cross-encoder docs
- [ ] **Metadata filtering** — `filters.py`, the cheapest +52% you'll ever get. → Qdrant filtering docs
- **Do:** index your Phase-1 pages, retrieve, watch Recall@5 appear. This is the spine.
- **IBM:** later transformers/embeddings courses touch embeddings; retrieval systems mostly not · **Est: 2–3 weeks**

## Phase 3 — Generation: turning pages into answers
Files: `answerer.py`, `verify.py`, `llm.py`, `prompts.py`

- [ ] **RAG (retrieval-augmented generation)** — the whole pattern. → original **RAG paper** (Lewis 2020) · Pinecone/LlamaIndex RAG guides
- [ ] **LLM APIs + prompting** — `llm.py`, `prompts.py`, structured output, the ABSTAIN token. → **Anthropic prompt-engineering docs**
- [ ] **Grounding, citations, abstention** — `answerer.py`, `verify.py`; why "I don't know" is a feature. → your own `verify.py` (small, readable — reimplement it)
- [ ] **Prompt versioning** — `prompts.py` registry, why v2 replaced v1. → concept: "prompt management"
- **IBM:** the later Generative AI / LLM application courses · **Est: 1–2 weeks**

## Phase 4 — Evaluation: the part that makes this special
Files: `eval/metrics.py`, `eval/run.py`, `eval/judge.py`, `eval/gate.py`, `eval/calibration.py`

- [ ] **Retrieval metrics** — Recall@k, nDCG, MRR. → *Intro to IR* ch. 8 · **reimplement `metrics.py` from scratch — it's tiny, do this first, it's the highest-confidence win**
- [ ] **LLM-as-judge** — `judge.py`, grading free text at scale. → Ragas docs · search "LLM as a judge"
- [ ] **Cohen's κ** — `calibration.py`, judge–human agreement. → the formula (Wikipedia) — you'll understand why n=3 was too small
- [ ] **CI eval gate** — `gate.py`, "CI for model behavior". → concept: regression testing applied to metrics
- **Why it matters:** this discipline (measure → bench failures → prove lift) is the #1 thing that separates you from "API wrapper" portfolios. Learn it cold.
- **IBM:** not covered — courses rarely teach evaluation rigor · **Est: 1 week**

## Phase 5 — The advanced / differentiator bits
Files: `retrieval/visual.py`, `retrieval/routed.py`, `agents/router.py`, `cost.py`, `observability.py`

- [ ] **Late-interaction visual retrieval** — `visual.py` (ColModernVBERT). → **ColPali paper** (arXiv 2407.01449) + blog posts
- [ ] **Agentic routing / decomposition / verifier** — `routed.py`, `router.py`; agents *where they earn their cost*. → LangGraph docs (concept only)
- [ ] **LLMOps** — caching, cost accounting, tracing. → **Chip Huyen, "AI Engineering" (2025)** — this whole book is your project's back half
- **IBM:** the later AI-agents / RAG courses touch agents · **Est: 2–3 weeks**

## Phase 6 — Serving & frontend
Files: `api/main.py`, `api/landing.html`, `api/app.html`, `Dockerfile`

- [ ] **FastAPI (async, threading)** — `main.py`; the singleton + lock bugs you saw are here. → **FastAPI docs** (excellent) · "GIL / threads" basics
- [ ] **Caching & warmup** — response cache, model-load-once. → concept: memoization
- [ ] **Frontend** — HTML/CSS/JS, the highlight-box overlay math. → MDN
- [ ] **Docker** — `Dockerfile`, deploy. → Docker "get started"
- **IBM:** not covered · **Est: 1–2 weeks**

---

## Suggested first two weeks (the "understand" track)
1. Reimplement `eval/metrics.py` from scratch; test it against the original.
2. Read `edgar.py` + `rasterize.py`; rebuild ingest for one filing by hand.
3. Read `text_baseline.py`; learn what an embedding is (Alammar); index + retrieve; watch the number.
4. Read `answerer.py` + `verify.py`; trace one question end-to-end on paper.

By day 14 you'll understand ingestion → retrieval → answer → verify — the spine of the whole
thing, and enough to defend the project in an interview.

## The single most important resources
- 📕 **"Hands-On Large Language Models"** — Alammar & Grootendorst (retrieval, embeddings, RAG)
- 📕 **"AI Engineering"** — Chip Huyen, 2025 (evaluation, serving, LLMOps — the systems half)
- 🎥 **Karpathy, "Neural Networks: Zero to Hero"** (what embeddings/transformers actually are)
- 📖 **FastAPI docs**, **Qdrant docs**, **Ragas docs**, **Anthropic prompt docs** (the exact tools here)
- 📖 **"Introduction to Information Retrieval"** — Manning et al., free online (metrics, BM25)
