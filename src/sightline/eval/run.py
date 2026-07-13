"""Eval runner.

    python -m sightline.eval.run                 # retrieval metrics (free, no API key)
    python -m sightline.eval.run --generation    # + answer quality (calls Claude, costs cents)

Retrieval metrics grade "did we find the right page"; generation metrics grade "did the answer
state the right fact, cite a right page, and abstain when it should". In M4 this same runner
is what the CI gate calls to block regressions.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .dataset import load_golden_set
from .metrics import mrr, ndcg_at_k, recall_at_k

console = Console()
GOLDEN = Path(__file__).parent / "golden_set.yaml"


def _make_retrieve_fn(name: str):
    """Build the retrieval function for one ablation config. Returns (fn, cleanup).

    dense            — BGE text embeddings in Qdrant (the M1 baseline)
    dense_filtered   — dense + deterministic ticker/form metadata filter
    bm25             — keyword/exact-match ranking (rank_bm25), built in memory from the store
    hybrid           — dense + bm25 fused with Reciprocal Rank Fusion
    visual           — ColModernVBERT page-image multivectors, MaxSim (M2)
    visual_filtered  — visual + the same metadata filter
    hybrid_tv        — filtered dense + filtered visual fused with RRF (M2)
    hybrid_tv_rerank — hybrid_tv top-20 re-ordered by a BGE cross-encoder (full M2 pipeline)
    """
    from sightline.config import settings
    from sightline.ingest.store import MetadataStore
    from sightline.retrieval.text_baseline import TextRetriever

    if name == "dense":
        r = TextRetriever()
        if r.count() == 0:
            raise SystemExit("Index is empty — run scripts/ingest.py then scripts/index.py first.")
        return r.retrieve, r.close

    if name in ("dense_filtered", "dense_chunked", "dense_chunked_filtered"):
        from sightline.retrieval.filters import parse_query_filters, to_qdrant_filter

        rf = TextRetriever(chunked="chunked" in name)
        if rf.count() == 0:
            raise SystemExit(
                f"Collection '{rf.collection}' is empty — build it first "
                f"(scripts/index.py{' --chunked' if rf.chunked else ''})."
            )
        if name == "dense_chunked":
            return rf.retrieve, rf.close

        def fn_filtered(query: str, k: int = 5):
            # Deterministic ticker/form filter parsed from the question itself.
            return rf.retrieve(query, k=k, query_filter=to_qdrant_filter(parse_query_filters(query)))

        return fn_filtered, rf.close

    if name == "dense_rerank":
        # Isolates the reranker's contribution: dense+filter top-20 -> cross-encoder, NO visual.
        # Compared against hybrid_tv_rerank, this reveals whether the visual leg adds anything.
        from sightline.retrieval.filters import parse_query_filters, to_qdrant_filter
        from sightline.retrieval.rerank import Reranker

        rr = TextRetriever()
        if rr.count() == 0:
            raise SystemExit("Index is empty — build scripts/index.py first.")
        store_r = MetadataStore(settings.data_dir / "sightline.db")
        reranker_r = Reranker()

        def fn_dr(query: str, k: int = 5):
            hits = rr.retrieve(query, k=20, query_filter=to_qdrant_filter(parse_query_filters(query)))
            pairs = [(h, (p.text if (p := store_r.get_page(h.accession, h.page_no)) else "")) for h in hits]
            return reranker_r.rerank(query, pairs, k=k)

        def cleanup_dr():
            rr.close()
            store_r.close()

        return fn_dr, cleanup_dr

    if name == "routed":
        # Config-routing: comparison -> decomposition (no rerank), else grand (rerank).
        from sightline.retrieval.routed import RoutedRetriever

        rr = RoutedRetriever()
        return rr.retrieve, rr.close

    if name == "grand":
        # All winning levers stacked: chunked embeddings + metadata filter + router-driven
        # decomposition (per-company/filing fan-out) -> candidate pool -> cross-encoder rerank.
        # No visual leg — the ablation showed it dilutes precision on this corpus/model.
        from sightline.retrieval.decompose import decomposed_retrieve
        from sightline.retrieval.rerank import Reranker

        rg = TextRetriever(chunked=True)
        if rg.count() == 0:
            raise SystemExit("Chunked index empty — build scripts/index.py --chunked first.")
        store_g = MetadataStore(settings.data_dir / "sightline.db")
        reranker_g = Reranker()

        def fn_grand(query: str, k: int = 5):
            fused = decomposed_retrieve(query, 20, rg.retrieve, store_g.list_accessions)
            pairs = [(h, (p.text if (p := store_g.get_page(h.accession, h.page_no)) else "")) for h in fused]
            return reranker_g.rerank(query, pairs, k=k)

        def cleanup_grand():
            rg.close()
            store_g.close()

        return fn_grand, cleanup_grand

    if name in ("planned", "champion"):
        # Deterministic decomposition: per-company / per-filing fan-out + interleave.
        # champion = decomposition on top of CHUNKED embeddings (both improvements stacked).
        from sightline.retrieval.decompose import decomposed_retrieve

        rp = TextRetriever(chunked=(name == "champion"))
        if rp.count() == 0:
            raise SystemExit(
                f"Collection '{rp.collection}' is empty — build it first "
                f"(scripts/index.py{' --chunked' if rp.chunked else ''})."
            )
        store_p = MetadataStore(settings.data_dir / "sightline.db")

        def fn_planned(query: str, k: int = 5):
            return decomposed_retrieve(query, k, rp.retrieve, store_p.list_accessions)

        def cleanup_planned():
            rp.close()
            store_p.close()

        return fn_planned, cleanup_planned

    if name in ("visual", "visual_filtered"):
        from sightline.retrieval.filters import parse_query_filters, to_qdrant_filter
        from sightline.retrieval.visual import VisualRetriever

        v = VisualRetriever()
        if v.count() == 0:
            raise SystemExit("Visual index is empty — run scripts/index_visual.py first.")
        if name == "visual":
            return v.retrieve, v.close

        def fn_vf(query: str, k: int = 5):
            return v.retrieve(query, k=k, query_filter=to_qdrant_filter(parse_query_filters(query)))

        return fn_vf, v.close

    if name in ("hybrid_tv", "hybrid_tv_rerank"):
        # Both legs get the metadata filter — it's the champion config's standard equipment.
        from qdrant_client import QdrantClient

        from sightline.retrieval.filters import parse_query_filters, to_qdrant_filter
        from sightline.retrieval.fusion import rrf
        from sightline.retrieval.visual import VisualRetriever

        # ONE shared client: embedded Qdrant is single-client-per-path, so the two legs must
        # not each open their own (that was the hybrid rows' silent failure).
        shared = QdrantClient(path=str(Path(settings.data_dir) / "qdrant"))
        dense = TextRetriever(client=shared)
        v = VisualRetriever(client=shared)
        if dense.count() == 0 or v.count() == 0:
            raise SystemExit("Need both text and visual indexes (scripts/index.py + index_visual.py).")

        def _fused(query: str, top_n: int):
            qf = to_qdrant_filter(parse_query_filters(query))
            return rrf(
                dense.retrieve(query, k=20, query_filter=qf),
                v.retrieve(query, k=20, query_filter=qf),
                top_n=top_n,
            )

        if name == "hybrid_tv":
            def fn_tv(query: str, k: int = 5):
                return _fused(query, top_n=k)

            def cleanup_tv():
                shared.close()

            return fn_tv, cleanup_tv

        from sightline.retrieval.rerank import Reranker

        store = MetadataStore(settings.data_dir / "sightline.db")
        reranker = Reranker()

        def fn_rerank(query: str, k: int = 5):
            fused = _fused(query, top_n=20)
            pairs = []
            for h in fused:
                page = store.get_page(h.accession, h.page_no)
                pairs.append((h, page.text if page else ""))
            return reranker.rerank(query, pairs, k=k)

        def cleanup_rerank():
            shared.close()
            store.close()

        return fn_rerank, cleanup_rerank

    from sightline.retrieval.bm25 import BM25Retriever

    with MetadataStore(settings.data_dir / "sightline.db") as store:
        bm25 = BM25Retriever(store.iter_pages())
    if name == "bm25":
        return bm25.retrieve, (lambda: None)

    if name == "hybrid":
        from sightline.retrieval.fusion import rrf

        dense = TextRetriever()
        if dense.count() == 0:
            raise SystemExit("Index is empty — run scripts/ingest.py then scripts/index.py first.")

        def fn(query: str, k: int = 5):
            # Each leg over-fetches (top-20) so fusion has real overlap to work with.
            return rrf(dense.retrieve(query, k=20), bm25.retrieve(query, k=20), top_n=k)

        return fn, dense.close

    raise SystemExit(f"Unknown retriever '{name}' "
                     f"(expected dense | bm25 | hybrid | visual | hybrid_tv)")


def run(golden_path: Path = GOLDEN, retriever_name: str = "dense",
        json_out: Path | None = None) -> None:
    cases = load_golden_set(golden_path)
    console.print(f"[bold]Loaded {len(cases)} eval cases[/bold] — "
                  f"retriever: [bold]{retriever_name}[/bold]")

    retrieve_fn, cleanup = _make_retrieve_fn(retriever_name)
    try:
        # Retrieval metrics score only answerable cases that have labeled gold pages.
        # Unanswerable cases exist to test *abstention* on the generation side, so including
        # them here (empty relevant-set -> guaranteed 0) would understate retrieval quality.
        retrieval_cases = [c for c in cases if c.answerable and c.relevant_pages]
        n_unanswerable = sum(1 for c in cases if not c.answerable)

        k = 10  # retrieve 10 so nDCG@10 is meaningful; Recall@5 slices the top 5
        per_slice: dict[str, list[tuple[float, float, float]]] = {}
        for c in retrieval_cases:
            relevant = {f"{p.accession}#{p.page_no}" for p in c.relevant_pages}
            hits = retrieve_fn(c.question, k=k)  # -> list[Hit]
            retrieved = [f"{h.accession}#{h.page_no}" for h in hits]
            scores = (
                recall_at_k(retrieved, relevant, 5),
                ndcg_at_k(retrieved, relevant, 10),
                mrr(retrieved, relevant),
            )
            per_slice.setdefault(c.slice, []).append(scores)

        _print_metrics(per_slice, n_cases=len(retrieval_cases), n_unanswerable=n_unanswerable,
                       retriever_name=retriever_name)

        if json_out is not None:
            import json

            all_rows = [s for rows in per_slice.values() for s in rows]
            metrics = {
                "retriever": retriever_name,
                "n": len(all_rows),
                "recall@5": round(_mean(all_rows, 0), 4),
                "ndcg@10": round(_mean(all_rows, 1), 4),
                "mrr": round(_mean(all_rows, 2), 4),
            }
            Path(json_out).write_text(json.dumps(metrics, indent=2) + "\n")
            console.print(f"[dim]wrote metrics -> {json_out}[/dim]")
    finally:
        cleanup()


def _mean(rows: list[tuple[float, float, float]], i: int) -> float:
    return sum(r[i] for r in rows) / len(rows) if rows else 0.0


def run_generation(golden_path: Path = GOLDEN, k: int = 5) -> None:
    """End-to-end answer quality over the golden set, on the CHAMPION retrieval config
    (routed, R@5 0.603) — generation is retrieval-bound, so it gets the best retrieval.
    Costs a few cents (one LLM call per case + a judge call per answered prose case)."""
    from sightline.answerer import Answerer
    from sightline.config import settings
    from sightline.ingest.store import MetadataStore
    from sightline.retrieval.routed import RoutedRetriever

    from .judge import Judge, numeric_match

    cases = load_golden_set(golden_path)
    retriever = RoutedRetriever()
    store = MetadataStore(settings.data_dir / "sightline.db")
    answerer = Answerer()
    judge = Judge()

    # Counters. "Citation accuracy" = of the answers given, how many cite a gold page.
    # "Abstention recall" = of the unanswerable questions, how many were refused.
    n_ans = n_false_abstain = n_cite_hit = n_correct = n_judged = 0
    n_unans = n_abstained_ok = n_attempted = 0
    interrupted: str | None = None
    judge_dead = False
    try:
        for c in cases:
            hits = retriever.retrieve(c.question, k=k)
            pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
            try:
                result = answerer.answer(c.question, pages)
            except Exception as e:  # quota/network mid-run -> keep the partial scorecard
                interrupted = f"stopped at '{c.id}': {e}"
                break
            n_attempted += 1

            if not c.answerable:
                n_unans += 1
                n_abstained_ok += result.abstained
                status = "[green]abstained ✓[/green]" if result.abstained else "[red]answered ✗[/red]"
                console.print(f"  {c.id:32s} {status}")
                continue

            n_ans += 1
            if result.abstained:
                n_false_abstain += 1
                console.print(f"  {c.id:32s} [yellow]abstained (should answer)[/yellow]")
                continue
            # A citation is correct if it points at ANY page that states the fact —
            # the canonical gold pages or the mined alternates (also_valid_pages).
            gold = {(p.accession, p.page_no) for p in c.relevant_pages} | {
                (p.accession, p.page_no) for p in c.also_valid_pages
            }
            cite_hit = any((ct.accession, ct.page_no) in gold for ct in result.citations)
            n_cite_hit += cite_hit
            if judge_dead:
                # LLM judge is quota-dead, but numeric gold answers still grade for free.
                deterministic = numeric_match(c.gold_answer or "", result.answer)
                if deterministic is None:
                    verdict = "?"
                else:
                    n_judged += 1
                    n_correct += deterministic
                    verdict = "✓" if deterministic else "✗"
            else:
                try:
                    correct = judge.is_correct(c.question, c.gold_answer or "", result.answer)
                    n_judged += 1
                    n_correct += correct
                    verdict = "✓" if correct else "✗"
                except Exception:
                    # Circuit breaker: one quota failure means the rest would fail too —
                    # stop burning retries on every remaining case.
                    judge_dead = True
                    verdict = "?"
            console.print(f"  {c.id:32s} cite={'✓' if cite_hit else '✗'} correct={verdict}")
    finally:
        retriever.close()
        store.close()

    title = "Text baseline — generation metrics"
    if interrupted:
        title += f" (PARTIAL: {n_attempted}/{len(cases)} cases)"
        console.print(f"[red]{interrupted}[/red]")
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    answered = n_ans - n_false_abstain
    table.add_row("answerable cases", str(n_ans))
    table.add_row("false abstentions", f"{n_false_abstain}/{n_ans}")
    table.add_row("citation accuracy (of answered)", f"{n_cite_hit}/{answered or 1} = {n_cite_hit/(answered or 1):.2f}")
    table.add_row("answer correctness (judge)", f"{n_correct}/{n_judged or 1} = {n_correct/(n_judged or 1):.2f}")
    table.add_row("abstention recall (unanswerable)", f"{n_abstained_ok}/{n_unans or 1} = {n_abstained_ok/(n_unans or 1):.2f}")
    console.print(table)
    console.print("[dim]Numeric gold answers are graded deterministically (free, exact); prose "
                  "answers by an uncalibrated LLM judge — treat those as approximate until M3 "
                  "calibration (Cohen's κ vs human labels).[/dim]")


def _print_metrics(per_slice: dict[str, list], n_cases: int, n_unanswerable: int,
                   retriever_name: str = "dense") -> None:
    table = Table(title=f"Retrieval metrics — {retriever_name} ({n_cases} answerable cases)")
    table.add_column("slice")
    table.add_column("n", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("MRR", justify="right")

    all_rows: list[tuple[float, float, float]] = []
    for slice_name in sorted(per_slice):
        rows = per_slice[slice_name]
        all_rows.extend(rows)
        table.add_row(slice_name, str(len(rows)),
                      f"{_mean(rows,0):.3f}", f"{_mean(rows,1):.3f}", f"{_mean(rows,2):.3f}")
    table.add_section()
    table.add_row("[bold]OVERALL[/bold]", str(len(all_rows)),
                  f"[bold]{_mean(all_rows,0):.3f}[/bold]",
                  f"[bold]{_mean(all_rows,1):.3f}[/bold]",
                  f"[bold]{_mean(all_rows,2):.3f}[/bold]")
    console.print(table)
    if n_unanswerable:
        console.print(f"[dim]({n_unanswerable} unanswerable cases held out — for abstention eval "
                      f"once the generation path exists.)[/dim]")


def _preview_slices(cases) -> None:
    table = Table(title="Golden set by slice")
    table.add_column("slice")
    table.add_column("count", justify="right")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.slice] = counts.get(c.slice, 0) + 1
    for slice_name, n in sorted(counts.items()):
        table.add_row(slice_name, str(n))
    console.print(table)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    name = args[args.index("--retriever") + 1] if "--retriever" in args else "dense"
    json_out = Path(args[args.index("--json") + 1]) if "--json" in args else None
    if "--ablation" in args:
        for n in ("bm25", "dense", "dense_filtered", "dense_chunked_filtered", "planned",
                  "champion", "routed", "grand", "hybrid", "visual", "visual_filtered",
                  "hybrid_tv", "hybrid_tv_rerank"):
            try:
                run(retriever_name=n)
            except SystemExit as e:  # e.g. visual index not built yet — skip, keep the rest
                console.print(f"[yellow]skipping {n}: {e}[/yellow]")
    else:
        run(retriever_name=name, json_out=json_out)
    if "--generation" in args:
        run_generation()  # the paid pass (needs LLM_API_KEY)
