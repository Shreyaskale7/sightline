"""Eval runner: `python -m sightline.eval.run`.

Right now it loads the golden set and prints structure so you can see the loop. As you build
M1, wire `retrieve_fn` to your TextRetriever and print the metrics table. In M4 this same
runner is what the CI gate calls to block regressions.
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
    run()
