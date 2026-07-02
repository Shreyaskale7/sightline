"""Tests for the ingestion metadata store — idempotency is a hard constraint, so prove it."""
from __future__ import annotations

from pathlib import Path

from sightline.ingest.edgar import Filing
from sightline.ingest.rasterize import Page
from sightline.ingest.store import MetadataStore


def _filing(accession: str = "0000320193-24-000123") -> Filing:
    return Filing(
        cik=320193, ticker="AAPL", form="10-K", filing_date="2024-11-01",
        accession=accession, primary_document="aapl-20240928.htm",
    )


def _pages(accession: str, n: int) -> list[Page]:
    return [
        Page(accession=accession, page_no=i, image_path=Path(f"/tmp/p{i:04d}.png"), text=f"page {i}")
        for i in range(1, n + 1)
    ]


def test_save_and_counts(tmp_path):
    f = _filing()
    with MetadataStore(tmp_path / "db.sqlite") as store:
        assert not store.is_ingested(f.accession)
        store.save_filing_with_pages(f, _pages(f.accession, 3))
        assert store.is_ingested(f.accession)
        assert store.count_filings() == 1
        assert store.count_pages() == 3


def test_reingest_is_idempotent(tmp_path):
    """Re-ingesting the same accession must not duplicate pages — it replaces them."""
    f = _filing()
    with MetadataStore(tmp_path / "db.sqlite") as store:
        store.save_filing_with_pages(f, _pages(f.accession, 5))
        store.save_filing_with_pages(f, _pages(f.accession, 5))  # re-run
        assert store.count_filings() == 1
        assert store.count_pages() == 5  # not 10


def test_reingest_replaces_page_set(tmp_path):
    """A re-render with a different page count fully replaces the old pages (no stragglers)."""
    f = _filing()
    with MetadataStore(tmp_path / "db.sqlite") as store:
        store.save_filing_with_pages(f, _pages(f.accession, 5))
        store.save_filing_with_pages(f, _pages(f.accession, 2))
        assert store.count_pages() == 2


def test_iter_pages_carries_provenance(tmp_path):
    f = _filing()
    with MetadataStore(tmp_path / "db.sqlite") as store:
        store.save_filing_with_pages(f, _pages(f.accession, 2))
        rows = list(store.iter_pages())
        assert [r.page_no for r in rows] == [1, 2]
        assert all(r.ticker == "AAPL" and r.form == "10-K" for r in rows)
        assert rows[0].text == "page 1"
