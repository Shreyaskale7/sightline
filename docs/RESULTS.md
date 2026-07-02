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
