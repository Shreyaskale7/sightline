"""Build the VISUAL (page-image) retrieval index from ingested pages.

    python scripts/index_visual.py                 # embed every page image (CPU, slow-ish)
    python scripts/index_visual.py --limit 100     # first N pages (for quick experiments)
    python scripts/index_visual.py --reset         # drop and rebuild

Separate from the text index because this one is expensive (ColModernVBERT forward pass per
page). Upserts happen per batch with deterministic ids, so an interrupted run is safe to
re-run — it just re-embeds; nothing duplicates.
"""
from __future__ import annotations

import typer
from rich.console import Console

from sightline.config import settings
from sightline.ingest.store import MetadataStore
from sightline.retrieval.visual import VisualRetriever

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    reset: bool = typer.Option(False, "--reset", help="Drop the collection before indexing."),
    limit: int = typer.Option(0, "--limit", help="Only index the first N pages (0 = all)."),
    batch_size: int = typer.Option(4, "--batch-size", help="Pages per forward pass."),
) -> None:
    store = MetadataStore(settings.data_dir / "sightline.db")
    retriever = VisualRetriever(batch_size=batch_size)
    try:
        if reset:
            retriever._ensure_client()  # noqa: SLF001 - CLI convenience
            if retriever._client.collection_exists(retriever.collection):
                retriever._client.delete_collection(retriever.collection)
                console.print(f"[yellow]dropped collection {retriever.collection}[/yellow]")

        pages = list(store.iter_pages())
        if limit:
            pages = pages[:limit]
        console.print(f"embedding {len(pages)} page images with {retriever.model_name} (CPU)…")
        n = retriever.index(pages)
        console.print(f"[green]indexed {n} pages[/green] into '{retriever.collection}' "
                      f"(total in collection: {retriever.count()})")
    finally:
        retriever.close()
        store.close()


if __name__ == "__main__":
    app()
