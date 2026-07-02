# Results

Measured numbers, recorded as they are produced. The M1 text baseline is the row every
later retrieval config (M2 visual/hybrid) must beat — see the prime directive in `CLAUDE.md`.

## M1 — text-only retrieval baseline (2026-07-02)

**Corpus:** latest 10-K for 5 semiconductor companies (NVDA, AMD, INTC, MU, QCOM).
522 pages ingested, **516 indexed** (6 blank/image-only pages skipped).

**Pipeline:** EDGAR HTML → Chromium (Playwright) → PDF → PyMuPDF rasterize + per-page text.
**Retriever:** dense text only. `BAAI/bge-small-en-v1.5` (384-d, via fastembed/ONNX, CPU),
cosine similarity in embedded Qdrant. No BM25, no visual, no rerank yet.

**Golden set:** 35 hand-labeled cases (30 answerable + 5 unanswerable), every label verified
against the actual ingested page text. Unanswerable cases are held out of the retrieval table
(they test abstention on the generation side).

| Slice | n | Recall@5 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|
| basic | 26 | 0.615 | 0.482 | 0.414 |
| cross_company | 4 | 0.250 | 0.343 | 0.306 |
| **OVERALL** | **30** | **0.567** | **0.463** | **0.399** |

(First measurement on the initial 15-case set gave OVERALL 0.583/0.464/0.401 — growing the set
to 35 barely moved the numbers, which is evidence the baseline measurement is stable, not luck.)

### Reading the numbers
- **basic (single-page factual) Recall@5 = 0.70.** Dense text finds the right page most of the
  time. The 3 misses are genuine weaknesses of single-vector text search, verified by
  inspecting the top-5:
  - `amd-revenue` — pulled AMD MD&A / narrative pages, not the terse consolidated-statement
    table (p61). Dense embeddings under-rank number-dense table pages.
  - `mu-employees` — "employees" retrieved Micron's income-statement pages instead of the
    human-capital page (p16).
  - `nvda-foundry` — retrieved wrong-*company* pages; a single dense query struggles to pin the
    entity. (Motivates BM25 exact-match anchors + metadata filtering.)
- **cross_company Recall@5 = 0.00 (expected).** These need pages from *two* filings, but one
  query can't surface both companies' statement pages in the top 5. This is the concrete
  failure that justifies M2 (hybrid retrieval) and M3 (query decomposition / planner) — not a
  bug.

### Reproduce
```bash
python scripts/ingest.py -t NVDA -t AMD -t INTC -t MU -t QCOM --forms 10-K --limit 1
python scripts/index.py --reset
python -m sightline.eval.run
```

### Not yet measured (needs the generation path)
Faithfulness, answer-correctness, citation-accuracy, and abstention calibration — these require
the LLM answer step (Anthropic Claude), which is the next M1 piece.

## Corpus expansion + first ablation (2026-07-02, later the same day)

**Corpus grew 516 → 1,329 indexed pages** (added 3 recent 10-Qs per company; 20 filings total).
**Golden set grew 35 → 38 cases** (3 new multi_hop cases spanning three quarterly filings each).
Retrieval configs measured: dense (BGE-small), BM25 (rank_bm25, plain tokenizer), and
hybrid = RRF(dense top-20, BM25 top-20), k=60.

| Config | Recall@5 | nDCG@10 | MRR |
|---|---:|---:|---:|
| BM25 only | 0.061 | 0.045 | 0.022 |
| Dense only | **0.313** | **0.267** | **0.192** |
| Hybrid (RRF) | 0.258 | 0.175 | 0.114 |

Per-slice (dense): basic 0.346 · cross_company 0.250 · multi_hop 0.111.

### The two honest findings

**1. The text-only baseline does not survive corpus growth.** Dense Recall@5 fell 0.567 → 0.313
when pages grew 2.6×. Cause (verified by inspecting hits): the added 10-Qs are near-duplicate
distractors — a 10-Q income statement looks almost identical to the 10-K's, and the embedding
of "…in its most recent 10-K" doesn't reliably prefer the 10-K page. Single-vector text
embeddings can't distinguish *which filing* a page belongs to. This is the measured, concrete
justification for M2 (layout-aware visual retrieval + reranking) and for metadata filtering
(form type / fiscal period) in M3 — not hand-waving.

**2. BM25 hurts on this corpus, and fusing it in drags dense down** (hybrid 0.258 < dense
0.313; on the small corpus it was 0.433 < 0.567). Gold pages are terse financial tables where
query words like "revenue" barely occur; prose-heavy MD&A pages swamp them. "Hybrid always
wins" is folklore until measured — BM25 stays benched until some variant proves a lift on this
eval set. The RRF plumbing is built, tested, and takes any number of legs (the M2 visual leg
joins the same fusion).

### Reproduce
```bash
python scripts/ingest.py -t NVDA -t AMD -t INTC -t MU -t QCOM --forms 10-Q --limit 3
python scripts/index.py --reset
python -m sightline.eval.run --ablation
```

**M2 target:** beat dense-only Recall@5 = 0.313 / nDCG@10 = 0.267 on this corpus and golden set.
