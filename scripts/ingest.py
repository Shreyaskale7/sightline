"""Ingestion CLI.

Examples:
    python scripts/ingest.py --tickers NVDA AMD --forms 10-K --limit 1
    python scripts/ingest.py --tickers NVDA --forms 10-K 10-Q --since 2022-01-01
"""
from __future__ import annotations

import typer
from rich.console import Console

from sightline.config import settings
from sightline.ingest.edgar import EdgarClient
from sightline.ingest.pipeline import ingest_filing
from sightline.ingest.store import MetadataStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    tickers: list[str] = typer.Option(..., "--tickers", "-t", help="Ticker symbols, e.g. NVDA AMD"),
    forms: list[str] = typer.Option(["10-K", "10-Q"], "--forms", "-f"),
    limit: int = typer.Option(1, "--limit", "-l", help="Max filings per company per form set"),
    since: str = typer.Option(None, "--since", help="ISO date lower bound, e.g. 2022-01-01"),
    list_only: bool = typer.Option(
        False, "--list-only", help="List matching filings without downloading/rasterizing."
    ),
) -> None:
    client = EdgarClient()
    store = MetadataStore(settings.data_dir / "sightline.db")
    try:
        cik_map = client.ticker_to_cik()
        for ticker in tickers:
            t = ticker.upper()
            cik = cik_map.get(t)
            if cik is None:
                console.print(f"[red]Unknown ticker: {t}[/red]")
                continue
            filings = client.get_filings(cik, t, forms=tuple(forms), limit=limit, since=since)
            console.print(f"[bold]{t}[/bold] (CIK {cik}): {len(filings)} filing(s)")
            for f in filings:
                console.print(f"  {f.form} {f.filing_date}  {f.accession}")
                if list_only:
                    console.print(f"    {f.primary_url}")
                    continue
                res = ingest_filing(client, store, f, settings.data_dir)
                if res.skipped:
                    console.print("    [dim]already ingested — skipped[/dim]")
                else:
                    console.print(f"    [green]ingested {res.n_pages} pages[/green]")
        console.print(
            f"\n[bold]Store:[/bold] {store.count_filings()} filings, {store.count_pages()} pages "
            f"in {settings.data_dir / 'sightline.db'}"
        )
    finally:
        store.close()
        client.close()


if __name__ == "__main__":
    app()
