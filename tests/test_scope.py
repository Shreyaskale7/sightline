"""Search-scope resolution: once a user has uploaded documents, their questions search THEIR
documents by default — not the sample SEC corpus — unless they name a sample company."""
from __future__ import annotations

from sightline.api.main import _resolve_scope
from sightline.ingest.store import MetadataStore
from sightline.ingest.upload import ingest_upload


def _tiny_pdf(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 100), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_has_uploads(tmp_path):
    with MetadataStore(tmp_path / "sightline.db") as store:
        assert store.has_uploads() is False
        ingest_upload(_tiny_pdf("Total contract value $12,000"), "deal.pdf", store, tmp_path)
        assert store.has_uploads() is True


def test_auto_scope_prefers_uploads_when_present(tmp_path):
    with MetadataStore(tmp_path / "sightline.db") as store:
        ingest_upload(_tiny_pdf("Monthly rent $4,500"), "lease.pdf", store, tmp_path)

    # A natural question naming no sample company -> scoped to the user's uploads.
    f = _resolve_scope("auto", "What is the monthly rent?", tmp_path)
    assert f is not None and f.form == "UPLOAD"


def test_auto_scope_defers_to_corpus_when_company_named(tmp_path):
    with MetadataStore(tmp_path / "sightline.db") as store:
        ingest_upload(_tiny_pdf("irrelevant"), "x.pdf", store, tmp_path)

    # Naming a sample-corpus company means the user wants the sample data, not their upload.
    assert _resolve_scope("auto", "What was NVIDIA's total revenue?", tmp_path) is None


def test_auto_scope_no_uploads_is_corpus(tmp_path):
    MetadataStore(tmp_path / "sightline.db").close()  # empty store
    assert _resolve_scope("auto", "What is the monthly rent?", tmp_path) is None


def test_explicit_scopes(tmp_path):
    MetadataStore(tmp_path / "sightline.db").close()
    # "uploads" forces the upload filter even with no uploads present (empty result is honest);
    # "corpus" forces the sample set even when uploads exist.
    assert _resolve_scope("uploads", "anything", tmp_path).form == "UPLOAD"
    assert _resolve_scope("corpus", "What is the monthly rent?", tmp_path) is None
