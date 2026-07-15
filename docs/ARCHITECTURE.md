# Architecture

Sightline answers analyst-grade questions about SEC filings by retrieving and reasoning over
**page images** rather than OCR'd text, and returns answers with page-level citations — or
abstains when the corpus can't support one. This document covers the system design and the
reasoning behind the main choices. Measured results live in [`RESULTS.md`](RESULTS.md).

## Problem

Financial filings (10-Ks, 10-Qs) are visually dense: nested tables, footnotes, multi-column
layouts, charts. OCR-and-chunk pipelines mangle exactly those structures. Sightline keeps each
page as an image so layout survives, and treats retrieval quality as a measured quantity rather
than an assumption.

## Pipeline

```
question
  → router        classify shape: simple / comparison / multi-hop
  → retrieval     metadata filter → chunked dense search → (rerank | decompose per config)
  → answerer      grounded on the retrieved pages only, every claim cited
  → verifier      strip uncited/fabricated citations; abstain if nothing valid remains
  → response      answer + cited page images, with the answer's figures boxed in place
```

A single trace spans every stage (router decision → candidates → rerank → answer → verify),
returned in the API response for inspection.

## Retrieval

Retrieval is the largest lever, so it is the most heavily measured part of the system. The
components:

- **Dense** — page text embedded with a BGE model (via `fastembed`/ONNX, CPU-friendly),
  cosine similarity in Qdrant.
- **Chunking** — pages are embedded as overlapping windows so facts below the embedder's token
  limit remain findable; results collapse back to page level, so the page stays the unit of
  retrieval and citation.
- **Metadata filtering** — company and form type are parsed from the question and used as
  payload filters, so a question about one company's 10-K isn't answered from a near-identical
  10-Q of another.
- **Cross-encoder reranking** — the fused candidate set (~20) is re-ordered by a model that
  reads the (question, page) pair together. Expensive, so it runs only on candidates.
- **Sparse (BM25)** and **visual (late-interaction page-image) retrieval** are implemented and
  measured; both underperformed the tuned text pipeline on this corpus and are kept out of the
  champion configuration (see `RESULTS.md` for the numbers).

No single configuration wins every question type, so a **router selects the retrieval strategy
per question**: comparison questions fan out into one search per company (preserving each
company's representation), everything else uses the reranked path. The router therefore doubles
as a quality optimizer, and each branch is backed by a measured per-slice result.

## Generation & grounding

The answerer receives only the retrieved pages and must tag every claim with the page it came
from. A deterministic **verifier** then removes citations to pages that were never retrieved and
converts unsupported answers into abstentions — so no uncited claim reaches the user.
**Abstention is a feature:** unanswerable questions are part of the benchmark on purpose.

## Evaluation

Every change is judged against a hand-verified benchmark of `(question → page)` and
`(question → answer)` pairs, including deliberately unanswerable questions.

- **Retrieval** — Recall@k, nDCG@10, MRR, reported as an ablation table across configurations.
- **Generation** — answer correctness (exact numeric match where the gold answer is a figure,
  LLM-as-judge for prose), citation accuracy, and abstention recall.
- **Regression gate** — a committed baseline plus a comparator that fails CI if a tracked metric
  drops past tolerance.

## Serving

FastAPI serves the console, the landing page, cited page images, and the `/query` and `/upload`
endpoints. The retrieval models load once (a warmed singleton) and requests are serialized,
because the embedded vector store is single-writer. A response cache makes repeated questions
free. Users can upload their own PDF, which flows through the same rasterize → index → retrieve
path and becomes immediately queryable.

## Tech choices

| Layer | Choice | Why |
|---|---|---|
| PDF → image + text | PyMuPDF | one dependency for raster + page-aligned text, no system libs |
| HTML → PDF | Playwright (headless Chromium) | faithful rendering of filing tables; honors page breaks |
| Text embeddings | BGE via fastembed (ONNX) | strong dense retriever, CPU-friendly, no torch |
| Vector store | Qdrant (embedded) | native multi-vector support; same API as the server |
| Reranker | BGE cross-encoder | large accuracy lever on a small candidate set |
| Visual retriever | ColModernVBERT | late-interaction, runs on CPU |
| Serving | FastAPI | async, simple, production-shaped |

## Scaling notes

Late-interaction visual embeddings produce many vectors per page, so index size is the real
cost at scale; the mitigations are token pooling, quantized multi-vectors, and the two-stage
retrieve (cheap prefilter → expensive model on the top-N only). Corpus size is capped
deliberately so iteration stays fast and the index stays on a single machine.
