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


def run(golden_path: Path = GOLDEN) -> None:
    from sightline.retrieval.text_baseline import TextRetriever

    cases = load_golden_set(golden_path)
    console.print(f"[bold]Loaded {len(cases)} eval cases[/bold]")

    retriever = TextRetriever()
    try:
        indexed = retriever.count()
        if indexed == 0:
            console.print(
                "[yellow]Index is empty.[/yellow] Ingest filings (scripts/ingest.py) and build "
                "the index (scripts/index.py) first."
            )
            _preview_slices(cases)
            return
        console.print(f"[dim]retrieving against {indexed} indexed pages[/dim]")

        # Retrieval metrics score only answerable cases that have labeled gold pages.
        # Unanswerable cases exist to test *abstention* on the generation side, so including
        # them here (empty relevant-set -> guaranteed 0) would understate retrieval quality.
        retrieval_cases = [c for c in cases if c.answerable and c.relevant_pages]
        n_unanswerable = sum(1 for c in cases if not c.answerable)

        k = 10  # retrieve 10 so nDCG@10 is meaningful; Recall@5 slices the top 5
        per_slice: dict[str, list[tuple[float, float, float]]] = {}
        for c in retrieval_cases:
            relevant = {f"{p.accession}#{p.page_no}" for p in c.relevant_pages}
            hits = retriever.retrieve(c.question, k=k)  # -> list[Hit]
            retrieved = [f"{h.accession}#{h.page_no}" for h in hits]
            scores = (
                recall_at_k(retrieved, relevant, 5),
                ndcg_at_k(retrieved, relevant, 10),
                mrr(retrieved, relevant),
            )
            per_slice.setdefault(c.slice, []).append(scores)

        _print_metrics(per_slice, n_cases=len(retrieval_cases), n_unanswerable=n_unanswerable)
    finally:
        retriever.close()


def _mean(rows: list[tuple[float, float, float]], i: int) -> float:
    return sum(r[i] for r in rows) / len(rows) if rows else 0.0


def run_generation(golden_path: Path = GOLDEN, k: int = 5) -> None:
    """End-to-end answer quality over the golden set. Costs a few cents (one Claude call per
    case + one cheap judge call per answered case)."""
    from sightline.answerer import Answerer
    from sightline.config import settings
    from sightline.ingest.store import MetadataStore
    from sightline.retrieval.text_baseline import TextRetriever

    from .judge import Judge

    cases = load_golden_set(golden_path)
    retriever = TextRetriever()
    store = MetadataStore(settings.data_dir / "sightline.db")
    answerer = Answerer()
    judge = Judge()

    # Counters. "Citation accuracy" = of the answers given, how many cite a gold page.
    # "Abstention recall" = of the unanswerable questions, how many were refused.
    n_ans = n_false_abstain = n_cite_hit = n_correct = n_judged = 0
    n_unans = n_abstained_ok = 0
    try:
        for c in cases:
            hits = retriever.retrieve(c.question, k=k)
            pages = [p for h in hits if (p := store.get_page(h.accession, h.page_no))]
            result = answerer.answer(c.question, pages)

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
            gold = {(p.accession, p.page_no) for p in c.relevant_pages}
            cite_hit = any((ct.accession, ct.page_no) in gold for ct in result.citations)
            n_cite_hit += cite_hit
            correct = judge.is_correct(c.question, c.gold_answer or "", result.answer)
            n_judged += 1
            n_correct += correct
            console.print(f"  {c.id:32s} cite={'✓' if cite_hit else '✗'} "
                          f"correct={'✓' if correct else '✗'}")
    finally:
        retriever.close()
        store.close()

    table = Table(title="Text baseline — generation metrics")
    table.add_column("metric")
    table.add_column("value", justify="right")
    answered = n_ans - n_false_abstain
    table.add_row("answerable cases", str(n_ans))
    table.add_row("false abstentions", f"{n_false_abstain}/{n_ans}")
    table.add_row("citation accuracy (of answered)", f"{n_cite_hit}/{answered or 1} = {n_cite_hit/(answered or 1):.2f}")
    table.add_row("answer correctness (judge)", f"{n_correct}/{n_judged or 1} = {n_correct/(n_judged or 1):.2f}")
    table.add_row("abstention recall (unanswerable)", f"{n_abstained_ok}/{n_unans or 1} = {n_abstained_ok/(n_unans or 1):.2f}")
    console.print(table)
    console.print("[dim]Correctness is graded by an uncalibrated LLM judge (Haiku) — treat as "
                  "approximate until M3 calibration (Cohen's κ vs human labels).[/dim]")


def _print_metrics(per_slice: dict[str, list], n_cases: int, n_unanswerable: int) -> None:
    table = Table(title=f"Text baseline — retrieval metrics ({n_cases} answerable cases)")
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

    if "--generation" in sys.argv:
        run()            # retrieval table first (free) ...
        run_generation()  # ... then the paid generation pass
    else:
        run()
