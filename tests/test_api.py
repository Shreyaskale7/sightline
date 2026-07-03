"""API tests: the demo page and the cited-page-image endpoint (no LLM, no Qdrant needed)."""
from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from sightline.api.main import app
from sightline.config import settings
from sightline.ingest.edgar import Filing
from sightline.ingest.rasterize import Page
from sightline.ingest.store import MetadataStore

client = TestClient(app)


def test_home_serves_demo_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Sightline" in r.text and "/query" in r.text


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.fixture()
def tmp_corpus(tmp_path, monkeypatch):
    """A one-page corpus in a temp data dir, with a real PNG on disk."""
    from PIL import Image

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    img = tmp_path / "p0001.png"
    Image.new("RGB", (10, 10), "white").save(img)
    filing = Filing(cik=1, ticker="TEST", form="10-K", filing_date="2026-01-01",
                    accession="0000000001-26-000001", primary_document="x.htm")
    with MetadataStore(tmp_path / "sightline.db") as store:
        store.save_filing_with_pages(
            filing, [Page(filing.accession, 1, img, "hello")]
        )
    return filing.accession


def test_page_image_served(tmp_corpus):
    r = client.get(f"/pages/{tmp_corpus}/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes


def test_page_image_404_for_unknown_page(tmp_corpus):
    assert client.get(f"/pages/{tmp_corpus}/999").status_code == 404
