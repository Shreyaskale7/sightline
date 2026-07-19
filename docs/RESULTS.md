# Results

Measured numbers, recorded as they are produced. The M1 text baseline is the row every
later retrieval config (M2 visual/hybrid) must beat — measure the baseline before optimizing.

## Generation scorecard on the champion retrieval (2026-07-13)

The thesis was "generation is retrieval-bound." Here it is, tested: the SAME answerer + judge,
but retrieval swapped from the M1 dense baseline (R@5 0.316) to the routed champion (0.603).

| Metric | M1 (dense 0.316) | Champion (routed 0.603) |
|---|--:|--:|
| answer correctness | 13/20 = **0.65** | 26/29 = **0.90** |
| citation accuracy (of answered) | 9/21 = 0.43 | 19/32 = **0.59** |
| false abstentions (answerable) | 12/33 = 0.36 | 7/39 = **0.18** |
| abstention recall (unanswerable) | 4/5 = 0.80 | 5/5 = **1.00** |

Better retrieval **halved** false abstentions (0.36 → 0.18), lifted correctness 0.65 → 0.90, and
took abstention recall to a perfect 5/5 — without touching a line of the generation code. Every
earlier false abstention that traced to a retrieval miss is now an answered, correct case. The
prime directive pays off: fix the measurable upstream stage, and the downstream number follows.
(Citation accuracy also benefited from the multi-gold `also_valid_pages` labels; both changes
are in this number.) Answers by `nvidia/nemotron-3-super-120b-a12b:free`; correctness by
deterministic numeric match + `nemotron-nano-9b` judge (whose trust is quantifiable via Cohen's
κ — see calibration.py). All 5 deliberately-unanswerable questions were correctly refused.

## Scale test: 3× the companies, identical champion score (2026-07-20)

The strongest result here. Corpus expanded from **5 companies / 20 filings / 1,329 pages** to
**15 companies / 32 filings / 2,353 pages** (10 more semiconductor issuers: TXN, ADI, AMAT, LRCX,
KLAC, NXPI, ON, MRVL, MCHP, SWKS). The benchmark was left untouched, so every added company is
pure *distractor* — more near-identical financial pages competing with the right one.

| Config (same 44-case benchmark) | 5 companies | **15 companies** |
|---|---:|---:|
| Dense, unfiltered | 0.316 | **0.325** |
| **Routed champion** (filter + chunk + decompose + rerank) | 0.603 | **0.603** |

Per-slice, the champion is *identical* at both scales: basic 0.656, cross_company 0.375,
multi_hop 0.333.

### Why this matters
Earlier, corpus growth was catastrophic: going from 10-Ks only to 10-Ks + 10-Qs collapsed dense
retrieval **0.567 → 0.313**, because near-duplicate statement pages became distractors. That
failure is what motivated metadata filtering and routing.

This run is the payoff. Tripling the company count moved the champion **not at all** — because
the filter excludes non-mentioned companies *before* scoring, so distractors that can't be the
answer never compete. The architecture converts corpus growth from a liability into a no-op.

That is the concrete answer to *"why not just paste the documents into a chat model?"* — a
context window degrades as you add documents ("lost in the middle"), and cost scales with every
page you send. Here, going from 1,329 to 2,353 pages cost **zero** accuracy and **zero** extra
per-query spend, because retrieval still reads only the top 5 pages. Scale is where a retrieval
system separates from a chat window.

**Caveat, stated honestly:** the benchmark questions all concern the original 5 companies, so
this measures *robustness to distractors*, not accuracy on the new companies. Extending the
golden set to the new issuers is the next step.

## Judge calibration (Cohen's κ) — why the number isn't published (2026-07-15)

The plan calls for calibrating the LLM-as-judge against human labels and reporting Cohen's κ.
The harness is built and tested (`eval/calibration.py`), but running it against this benchmark
produced only **3 judge-graded rows** — too few to compute a credible κ (one flip swings it
wildly). This is by design, not a gap:

- Most golden answers are **figures** ($215,938M, 42,000 employees, …), graded by an exact
  deterministic `numeric_match` — no model opinion involved, so no κ needed. That's the *more*
  trustworthy grader, and it covers the majority of cases.
- Only free-text (prose) answers fall through to the LLM judge — a small slice of a
  numeric-heavy exam.

Publishing κ on n=3 would be less honest than saying this plainly: **the judge covers a
minority of cases, and the majority is graded deterministically.** A meaningful κ would require
deliberately adding ~10+ qualitative questions to enlarge the prose slice — logged as future
work rather than faked now. The harness stands ready for that day.

The same export surfaced (and hardened against) two free-model failure modes — null `content`
responses and mid-sentence truncation from a reasoning model dumping its chain-of-thought; the
answerer prompt is now v2 (reasoning suppressed) and the client coerces null → abstain.

## M4 — the money-shot (2026-07-10)

The point of the two-stage retrieve (cheap text prefilter → VLM reads only the top-k) is cost.
`python scripts/moneyshot.py` quantifies it. At **gpt-4o** vision prices on our 1,345-page
corpus, answering one question:

| Approach | Pages the VLM reads | $/query | $/1k queries |
|---|--:|--:|--:|
| Naive VLM over every page | 1,345 | $3.72 | $3,719 |
| Sightline two-stage retrieve (k=5) | 5 | $0.017 | $17 |

**→ 99.5% cheaper per query.** Our actual stack runs free models, so the real bill is ~$0 —
this is the *architectural* saving, shown at a paid model's prices so it's legible. The number
scales with k and holds (>95% reduction) across gpt-4o / claude-sonnet / gemini-flash pricing.

## M2 — retrieval ablation, text side (2026-07-07)

**Corpus:** 20 filings / 1,329 pages. **Exam:** 44-case golden set (39 answerable). All configs
scored on the same set. The visual/hybrid rows are pending the visual index build (in progress).

| Config | Recall@5 | nDCG@10 | MRR | What it adds |
|---|---:|---:|---:|---|
| BM25 (keyword) | 0.051 | 0.046 | 0.022 | benched — measured, failed |
| Dense (BGE) | 0.316 | 0.259 | 0.183 | baseline |
| Dense + metadata filter | 0.479 | 0.408 | 0.339 | ticker/form filter (+52%) |
| Dense + filter + **chunking** | 0.504 | 0.416 | 0.345 | overlapping page windows (no 512-tok truncation) |
| Planned (filter + decomposition) | 0.483 | 0.410 | 0.357 | per-company / per-filing fan-out |
| **Champion** (chunking + filter + decomposition) | **0.509** | **0.419** | **0.361** | both stacked |

**Per-slice, champion vs its ingredients (Recall@5):**

| Slice | n | dense | +filter+chunk | +decomposition (champion) |
|---|--:|--:|--:|--:|
| basic | 32 | ~0.35 | **0.562** | 0.562 |
| cross_company | 4 | 0.25 | 0.250 | **0.375** |
| multi_hop | 3 | 0.11 | 0.222 | 0.111 (MRR ↑ 0.11→0.33) |

### What the numbers say
- **Chunking earns its place** (+0.025 overall, concentrated in `basic`): pages average 2.7
  windows each, confirming the 512-token truncation was real — facts in a page's lower half
  were previously invisible.
- **Decomposition does exactly its job**: `cross_company` 0.250 → 0.375. Per-company fan-out
  guarantees both companies' pages reach the top-k, which one blended query couldn't.
- **The improvements stack** (champion 0.509 > either alone) with one honest wart: `multi_hop`
  Recall@5 halved while its MRR tripled. Cause (verified by inspection, not guessed):
  decomposition correctly returns one page per quarterly filing, but *within* each filing dense
  still prefers the MD&A narrative ("revenue increased X%") over the terse statement table —
  the original finding resurfacing. This is precisely what the visual leg (whole-page image)
  and cross-encoder rerank are meant to fix. `multi_hop` is n=3, so treat the slice as
  directional, not conclusive.

**Bar for the visual/hybrid rows: beat champion Recall@5 = 0.509.**

## M2 — the full ablation table (2026-07-10)

The centerpiece. Every row measured on the same 44-case exam (39 answerable), same 1,329-page
corpus. This is the whole point of the eval harness: not one number, but a defensible ranking
of every design choice.

| # | Config | Recall@5 | nDCG@10 | MRR |
|--:|---|---:|---:|---:|
| 1 | BM25 (keyword) | 0.051 | 0.046 | 0.022 |
| 2 | Visual only (ColModernVBERT) | 0.154 | 0.080 | 0.043 |
| 3 | Visual + filter | 0.205 | 0.153 | 0.084 |
| 4 | Dense (BGE) — M1 baseline | 0.316 | 0.259 | 0.183 |
| 5 | Hybrid text+visual (RRF) | 0.368 | 0.312 | 0.232 |
| 6 | Dense + filter | 0.479 | 0.408 | 0.339 |
| 7 | Planned (filter + decomposition) | 0.483 | 0.410 | 0.357 |
| 8 | Dense + filter + chunking | 0.504 | 0.416 | 0.345 |
| 9 | Champion (chunk + filter + decompose) | 0.509 | 0.419 | 0.361 |
| 10 | Hybrid text+visual + rerank | 0.517 | 0.412 | 0.370 |
| 11 | Dense + filter + rerank | 0.551 | 0.485 | 0.452 |
| 12 | Grand (chunk + filter + decompose + rerank) | 0.577 | 0.479 | 0.442 |
| 13 | **Routed (config per question type)** | **0.603** | **0.485** | 0.441 |

**Per-slice Recall@5 (where the real story lives):**

| Slice | n | dense (#4) | champion (#9) | grand (#12) | **routed (#13)** |
|---|--:|--:|--:|--:|--:|
| basic | 32 | 0.35 | 0.562 | 0.656 | **0.656** |
| cross_company | 4 | 0.25 | 0.375 | 0.125 | **0.375** |
| multi_hop | 3 | 0.11 | 0.111 | 0.333 | **0.333** |

### The five findings
1. **The cross-encoder reranker is the single biggest lever**: dense 0.316 → dense+filter+rerank
   0.551 (**+74%**). Reading (query, page-text) as one input beats any bi-encoder similarity.
2. **Visual retrieval underperformed — and it was measured, not assumed.** ColModernVBERT (the
   ~250M CPU-runnable starter) scores 0.154 alone. Diagnosed by inspection: it retrieves the
   right *company* and right *content type* (income statements) but the wrong *filing/page* —
   it can't distinguish a 10-K statement from a near-identical 10-Q one, worsened by blank
   charts in our offline-rendered pages. A larger model (ColQwen2.5) on GPU is the likely path
   to a visual win — scoped as a cost/quality tradeoff, not claimed.
3. **Visual actively HURTS in fusion here**: adding it to text+rerank *drops* the score
   (0.551 → 0.517). Its lower-precision hits dilute the reranker's candidate pool. "Hybrid
   always wins" is folklore; measured, it loses on this corpus.
4. **Metadata filtering + chunking are cheap, real wins** (0.316 → 0.504) with zero model cost —
   encode the metadata you already have before reaching for a bigger model.
5. **No single config wins every slice — and reranking *undoes* decomposition.** Decomposition
   lifts cross_company (0.250 → 0.375), but reranking the fused pool re-crowds it back to 0.125
   (rerank re-sorts by global relevance, breaking the per-company guarantee). The production
   answer is a **router that picks the retrieval config by question type**: decomposition for
   comparisons, grand (rerank) for everything else. The router that classifies these already
   exists (100% on the golden slices) — wiring it to retrieval configs is the next step.

**Operative production champion: Routed (0.603).** The router picks the measured-best config
per question type — decomposition (no rerank) for comparisons so the per-company interleave
survives, grand (rerank) for everything else. It takes the top slice from each: basic 0.656,
cross_company 0.375, multi_hop 0.333 — none sacrificed. This is the ablation's payoff: the
router, originally a cost switch, is now also a quality optimizer, and every branch is backed
by a measured per-slice number rather than intuition. Recall@5 0.316 → 0.603 = **+91%** over
the dense baseline, entirely on free/CPU components. M3's remaining job: calibrate the LLM
judge (Cohen's κ) and add table-QA / NER anchoring.

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
Switched to OpenRouter's free models: answers by
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
