# Results

Measured numbers, recorded as they are produced. The M1 text baseline is the row every
later retrieval config (M2 visual/hybrid) must beat — see the prime directive in `CLAUDE.md`.

## M1 — text-only retrieval baseline (2026-07-02)

**Corpus:** latest 10-K for 5 semiconductor companies (NVDA, AMD, INTC, MU, QCOM).
522 pages ingested, **516 indexed** (6 blank/image-only pages skipped).

**Pipeline:** EDGAR HTML → Chromium (Playwright) → PDF → PyMuPDF rasterize + per-page text.
**Retriever:** dense text only. `BAAI/bge-small-en-v1.5` (384-d, via fastembed/ONNX, CPU),
cosine similarity in embedded Qdrant. No BM25, no visual, no rerank yet.

**Golden set:** 15 hand-labeled cases (12 answerable + 3 unanswerable), labels grounded in the
actual ingested pages. Unanswerable cases are held out of the retrieval table (they test
abstention on the generation side, which doesn't exist yet).

| Slice | n | Recall@5 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|
| basic | 10 | 0.700 | 0.519 | 0.458 |
| cross_company | 2 | 0.000 | 0.185 | 0.113 |
| **OVERALL** | **12** | **0.583** | **0.464** | **0.401** |

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
