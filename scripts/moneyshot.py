"""Print the money-shot metric: what the two-stage retrieve saves vs naive VLM-over-everything.

    python scripts/moneyshot.py            # default reference model (gpt-4o)
    python scripts/moneyshot.py --model gemini-2.5-flash

Uses the real corpus size from the store when available, else the default.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from sightline.cost import PRICING, money_shot

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: str = typer.Option("gpt-4o", "--model", help=f"one of: {', '.join(PRICING)}"),
    k: int = typer.Option(5, "--k", help="pages the retriever feeds the VLM"),
) -> None:
    corpus = 1329
    try:  # real number if the store exists
        from sightline.config import settings
        from sightline.ingest.store import MetadataStore

        with MetadataStore(settings.data_dir / "sightline.db") as store:
            corpus = store.count_pages() or corpus
    except Exception:
        pass

    ms = money_shot(model=model, corpus_pages=corpus, k=k)
    t = Table(title=f"Money-shot — cost per query ({model}, {corpus}-page corpus)")
    t.add_column("approach")
    t.add_column("pages read", justify="right")
    t.add_column("$/query", justify="right")
    t.add_column("$/1k queries", justify="right")
    t.add_row("Naive VLM over every page", str(ms.pages_read_naive),
              f"${ms.naive_usd:.4f}", f"${ms.naive_usd*1000:.2f}")
    t.add_row("Sightline two-stage retrieve", str(ms.pages_read_sightline),
              f"${ms.sightline_usd:.4f}", f"${ms.sightline_usd*1000:.2f}")
    console.print(t)
    console.print(f"[bold green]→ {ms.reduction_pct:.1f}% cheaper per query[/bold green] "
                  f"(reads {ms.pages_read_sightline} pages, not {ms.pages_read_naive}). "
                  f"Our actual stack is free models, so the real bill is ~$0 — this is the "
                  f"architectural saving at a paid model's prices.")


if __name__ == "__main__":
    app()
