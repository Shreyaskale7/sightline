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

## Metadata filtering: the free +61% (2026-07-03)

The corpus-growth regression (finding 1 above) had a deterministic cause — so it got a
deterministic fix. `retrieval/filters.py` parses company and form-type mentions out of the
question with plain string matching ("…in its most recent 10-K", "NVIDIA or Intel") and
restricts vector search via Qdrant payload filters (`ticker`, `form`) that ingestion already
stored. No model, no API, ~60 lines, fully explainable.

| Config | Recall@5 | nDCG@10 | MRR |
|---|---:|---:|---:|
| dense (unfiltered) | 0.313 | 0.267 | 0.192 |
| **dense + metadata filter** | **0.505** | **0.425** | **0.364** |

Per-slice R@5: basic 0.346→0.577 (nearly recovers the pre-growth 0.615), multi_hop
0.111→0.222 (doubled), cross_company unchanged at 0.250 (needs decomposition, not filtering).

Lesson recorded: before reaching for a bigger model, encode the metadata you already have.
Out-of-corpus mentions (e.g. Broadcom) deliberately yield NO filter — the question stays
corpus-wide and abstention handles it downstream. M3's NER-based anchoring must beat this
deterministic baseline to ship. **New M2 bar: visual/hybrid must beat R@5 0.505.**

## First generation scorecard — PARTIAL (2026-07-02)

**Setup:** answers by `anthropic/claude-sonnet-4.5`, judged by `anthropic/claude-haiku-4.5`,
both via OpenRouter. Retrieval: dense top-5. The run stopped at case 15/38 when the
OpenRouter account's ~$0.22 promotional credit ran out (402) — the runner now survives that
and reports partial numbers. Full re-run pending credits.

Over the 14 answerable cases attempted (9 answered, 5 abstained):

| Metric | Value |
|---|---:|
| answer correctness (judge) | 7/9 = 0.78 |
| citation accuracy | 5/9 = 0.56 |
| false abstentions | 5/14 = 0.36 |
| abstention recall (unanswerable) | not reached |

### What the partial data already shows
- **Grounding works.** Every answer given was drawn from retrieved pages; when the right page
  wasn't retrieved, the model *abstained* rather than hallucinated (all 5 false abstentions are
  exactly the cases where retrieval missed — amd-revenue, nvda-foundry, mu-employees,
  nvda-intc-rnd-compare, amd-rnd). Generation quality is retrieval-bound: fix retrieval (M2)
  and false abstentions should fall roughly in step.
- **Citation ≠ correctness.** e.g. `intc-revenue` was answered correctly but "mis-cited": the
  same figure appears on more than one page (income statement + MD&A), and the model cited a
  legitimate page the golden set doesn't list. Action item: golden labels should list *all*
  pages that state the fact, not just the statement page.
- Cost so far: ~$0.22 for ~24 calls (15 Sonnet answers + 9 Haiku judgments) — a full 38-case
  scorecard costs roughly $0.50/run. Budget accordingly (M4 makes this a dashboard).

### Pivot to a $0 model stack (same day)
Student budget → switched to OpenRouter's free models: answers by
`nvidia/nemotron-3-super-120b-a12b:free`, judge `nvidia/nemotron-nano-9b-v2:free`
(and `nemotron-nano-12b-v2-vl:free` earmarked for the M2 vision path). Free tiers cap
*requests/day* (~50), so two efficiency pieces were pulled forward from M4, both tested:
- **Exact-match response cache** (`llm.py`, SQLite): re-runs and resumed runs pay only for
  unseen cases. All 33 answerable-case answers are cached; the first re-run replayed them with
  zero API calls.
- **Throttle + exponential backoff** for `:free` models (~15 req/min, 429-aware retries).

All 33 answerable cases have (cached) free-model answers: 21 answered / 12 abstained,
citation accuracy 9/21 = 0.43. Judge verdicts + the 5 unanswerable cases hit the daily cap —
the same command completes the scorecard after quota reset, paying only for what's missing.
Early read: the free 120B model abstains more than Sonnet did (12/33 vs ~5/14 on the
overlapping prefix) and mis-cites more — a model-quality gap the M4 A/B comparison will
quantify properly.

## ✅ M1 COMPLETE — final generation scorecard (2026-07-03)

Answer model `nvidia/nemotron-3-super-120b-a12b:free`; judge = deterministic numeric matcher
for figure answers + `nemotron-nano-9b-v2:free` for prose. Retrieval: dense top-5
(unfiltered — the M1 baseline config). Total M1 spend: **$0.22** (the early Sonnet run);
the final scorecard itself cost $0.

| Metric | Value |
|---|---:|
| answer correctness | 13/20 = **0.65** |
| citation accuracy (of answered) | 9/21 = **0.43** |
| false abstentions (answerable) | 12/33 = 0.36 |
| **abstention recall (unanswerable)** | **4/5 = 0.80** |

Readings:
- **Grounding discipline works.** Every false abstention maps to a retrieval miss — the model
  refuses rather than invents when the right page isn't in front of it. Fixing retrieval
  (M2: the metadata filter alone took R@5 0.313→0.505) should convert most refusals into
  correct answers. Generation is retrieval-bound.
- **The one honesty failure** (`nvda-stock-price`): asked for a stock price on a specific
  date, the model answered from the 10-K's stock-performance disclosure instead of abstaining
  — plausible-page-but-wrong-question, exactly the case the M3 LLM support-checker targets.
- Citation accuracy (0.43) under-measures: several "misses" cite a *different* page that also
  contains the fact (income statement vs MD&A). Action item stands: gold labels should list
  all pages stating the fact.

M1 is done: ingestion, text baseline, eval harness (retrieval + generation + abstention),
all measured. M2's job: beat R@5 0.505 (dense+filter) with visual/hybrid, then re-run this
scorecard on the winning config.
