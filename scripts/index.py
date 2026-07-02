"""Build the text retrieval index from the ingested pages.

    python scripts/index.py            # index every page in the store
    python scripts/index.py --reset    # drop and rebuild the collection first

Kept separate from ingestion so you can re-embed (e.g. after changing the model) without
re-downloading filings. Reads the SQLite store, writes vectors into Qdrant.
"""
from __future__ import annotations

import typer
from rich.console import Console

from sightline.config import settings
from sightline.ingest.store import MetadataStore
from sightline.retrieval.text_baseline import TextRetriever

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(reset: bool = typer.Option(False, "--reset", help="Drop the collection before indexing.")) -> None:
    store = MetadataStore(settings.data_dir / "sightline.db")
    retriever = TextRetriever()
    try:
        if reset:
            retriever._ensure()  # noqa: SLF001 - CLI convenience
            if retriever._client.collection_exists(retriever.collection):
                retriever._client.delete_collection(retriever.collection)
                console.print(f"[yellow]dropped collection {retriever.collection}[/yellow]")

        n = retriever.index(store.iter_pages())
        console.print(f"[green]indexed {n} pages[/green] into '{retriever.collection}' "
                      f"(total in collection: {retriever.count()})")
    finally:
        retriever.close()
        store.close()


if __name__ == "__main__":
    app()
