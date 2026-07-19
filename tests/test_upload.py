"""Upload tests — identity must be citation-safe, guardrails must guard, endpoint must flow."""
from __future__ import annotations

import re

import pytest

from sightline.answerer import _CITE_RE
from sightline.ingest.upload import (
    MAX_BYTES,
    UploadError,
    ingest_upload,
    label_from_filename,
    synthetic_accession,
)


def _tiny_pdf(text: str = "Total revenue $9,876 million") -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 100), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_synthetic_accession_is_citation_parseable():
    acc = synthetic_accession(b"some pdf bytes")
    assert re.fullmatch(r"\d{10}-99-\d{6}", acc)
    # The whole point: a citation tag built from it must parse.
    assert _CITE_RE.findall(f"Revenue was $5 [p:{acc}#3].") == [(acc, "3")]


def test_synthetic_accession_deterministic_and_distinct():
    assert synthetic_accession(b"aaa") == synthetic_accession(b"aaa")
    assert synthetic_accession(b"aaa") != synthetic_accession(b"bbb")


def test_label_from_filename():
    assert label_from_filename("Annual-Report_2026 final.pdf") == "ANNUAL"
    assert label_from_filename("???.pdf") == "DOC"


def test_guardrails():

    with pytest.raises(UploadError, match="empty"):
        ingest_upload(b"", "x.pdf", None, None)
    with pytest.raises(UploadError, match="look like a PDF"):
        ingest_upload(b"MZplainly-not-pdf", "x.pdf", None, None)
    with pytest.raises(UploadError, match="limit is 100 MB"):
        ingest_upload(b"%PDF" + b"0" * (MAX_BYTES + 1), "x.pdf", None, None)


def test_ingest_upload_end_to_end(tmp_path):
    from sightline.ingest.store import MetadataStore

    with MetadataStore(tmp_path / "db.sqlite") as store:
        filing, pages = ingest_upload(_tiny_pdf(), "MyCo-Annual.pdf", store, tmp_path)
        assert filing.form == "UPLOAD" and filing.ticker == "MYCOAN"
        assert len(pages) == 1 and pages[0].image_path.exists()
        assert "9,876" in pages[0].text
        assert store.is_ingested(filing.accession)
        # provenance flows to the indexer's input
        rows = list(store.iter_pages_for(filing.accession))
        assert rows[0].ticker == "MYCOAN" and rows[0].form == "UPLOAD"


def test_upload_endpoint_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sightline.api import main as api_main
    from sightline.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    class _FakeRetriever:
        def index_pages(self, pages):
            return len(list(pages))

    monkeypatch.setattr(api_main, "_get_retriever", lambda: _FakeRetriever())
    client = TestClient(api_main.app)

    pdf = _tiny_pdf()
    r = client.post("/upload", files=[("files", ("report.pdf", pdf, "application/pdf"))])
    assert r.status_code == 200, r.text
    d = r.json()
    doc = d["documents"][0]
    assert doc["pages"] == 1 and doc["indexed"] == 1 and doc["label"] == "REPORT"
    assert d["total_pages"] == 1 and d["total_indexed"] == 1

    # same bytes again -> idempotent no-op
    r2 = client.post("/upload", files=[("files", ("report.pdf", pdf, "application/pdf"))])
    assert r2.json()["documents"][0]["already_known"] is True

    # the uploaded page image is servable for the exhibit view
    img = client.get(f"/pages/{doc['accession']}/1")
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"

    # a non-PDF is refused with a clear message
    bad = client.post("/upload", files=[("files", ("x.pdf", b"not a pdf", "application/pdf"))])
    assert bad.status_code == 400 and "PDF" in bad.json()["detail"]


def test_upload_batch_partial_failure(tmp_path, monkeypatch):
    """A batch with one bad file still indexes the good ones — per-file errors, not all-or-nothing."""
    from fastapi.testclient import TestClient

    from sightline.api import main as api_main
    from sightline.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    class _FakeRetriever:
        def index_pages(self, pages):
            return len(list(pages))

    monkeypatch.setattr(api_main, "_get_retriever", lambda: _FakeRetriever())
    client = TestClient(api_main.app)

    r = client.post("/upload", files=[
        ("files", ("good_a.pdf", _tiny_pdf("Revenue $1,111"), "application/pdf")),
        ("files", ("broken.pdf", b"definitely not a pdf", "application/pdf")),
        ("files", ("good_b.pdf", _tiny_pdf("Revenue $2,222"), "application/pdf")),
    ])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_pages"] == 2 and d["total_indexed"] == 2   # both good files landed
    errs = [x for x in d["documents"] if x["error"]]
    assert len(errs) == 1 and errs[0]["filename"] == "broken.pdf"
