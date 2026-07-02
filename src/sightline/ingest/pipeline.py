"""Ingest one filing end-to-end: download -> PDF -> page images + text -> metadata store.

This is the side-effect-heavy edge of ingestion (network, headless browser, disk). Keeping it
in one place lets the pure pieces (rasterize, store) stay easy to test. Every stage runs inside
a `span(...)` so a whole ingest is one trace once Langfuse is wired.

Idempotency (hard constraint): we skip a filing whose accession is already fully stored, so
`make ingest` is safe to re-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..observability import span
from .edgar import EdgarClient, Filing
from .rasterize import build_pages, html_to_pdf
from .store import MetadataStore


@dataclass
class IngestResult:
    accession: str
    ticker: str
    form: str
    n_pages: int
    skipped: bool  # True if already ingested (idempotent no-op)


def ingest_filing(
    client: EdgarClient,
    store: MetadataStore,
    filing: Filing,
    data_dir: Path,
) -> IngestResult:
    """Download, rasterize, and persist a single filing. Idempotent on accession number."""
    if store.is_ingested(filing.accession):
        return IngestResult(filing.accession, filing.ticker, filing.form, 0, skipped=True)

    # One directory per filing, named by the dash-free accession (stable, collision-free).
    work_dir = Path(data_dir) / "pages" / filing.accession_nodash
    work_dir.mkdir(parents=True, exist_ok=True)

    with span("ingest.download", accession=filing.accession, url=filing.primary_url):
        raw = client.download(filing.primary_url)

    pdf_path = work_dir / "filing.pdf"
    with span("ingest.to_pdf", accession=filing.accession):
        # 10-K/10-Q primary docs are HTML; guard the rare case of an already-PDF primary doc.
        if filing.primary_document.lower().endswith(".pdf"):
            pdf_path.write_bytes(raw)
        else:
            html_to_pdf(raw, pdf_path)

    with span("ingest.rasterize", accession=filing.accession) as s:
        pages = build_pages(pdf_path, work_dir, filing.accession)
        s["n_pages"] = len(pages)

    with span("ingest.store", accession=filing.accession):
        store.save_filing_with_pages(filing, pages)

    return IngestResult(filing.accession, filing.ticker, filing.form, len(pages), skipped=False)
