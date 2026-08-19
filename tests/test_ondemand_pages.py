"""On-demand page rendering: the deployable artifact ships source PDFs, not rendered PNGs.

Page images are derived data (2,353 PNGs ~738 MB vs 32 source PDFs ~21 MB). Shipping only the
PDFs is what makes the deployment fit a free tier, so the guarantee under test is: a page image
that is absent from disk can be reproduced from its filing PDF, and reproduced *faithfully*.
"""
from __future__ import annotations

import fitz

from sightline.ingest.rasterize import build_pages, ensure_page_image


def _two_page_pdf(path) -> None:
    doc = fitz.open()
    for text in ("Consolidated Statements of Income", "Human Capital"):
        doc.new_page(width=612, height=792).insert_text((72, 100), text, fontsize=14)
    doc.save(str(path))
    doc.close()


def test_renders_missing_page_from_filing_pdf(tmp_path):
    work = tmp_path / "0000000000-99-000001"
    work.mkdir()
    _two_page_pdf(work / "filing.pdf")

    pages = build_pages(work / "filing.pdf", work, accession="0000000000-99-000001")
    target = pages[1].image_path
    original = target.read_bytes()

    target.unlink()                       # simulate a deployed image that ships PDFs only
    assert not target.exists()

    out = ensure_page_image(target, page_no=2)
    assert out is not None and out.exists()
    # Faithful, not merely present: same renderer, same DPI -> same bytes.
    assert out.read_bytes() == original


def test_existing_image_is_returned_untouched(tmp_path):
    work = tmp_path / "0000000000-99-000002"
    work.mkdir()
    _two_page_pdf(work / "filing.pdf")
    pages = build_pages(work / "filing.pdf", work, accession="0000000000-99-000002")

    before = pages[0].image_path.read_bytes()
    out = ensure_page_image(pages[0].image_path, page_no=1)
    assert out == pages[0].image_path
    assert out.read_bytes() == before      # cached: not re-rendered


def test_returns_none_when_unrenderable(tmp_path):
    # No filing.pdf to render from -> None, so callers 404 instead of crashing.
    missing = tmp_path / "nope" / "p0001.png"
    assert ensure_page_image(missing, page_no=1) is None

    # Page number out of range -> None.
    work = tmp_path / "0000000000-99-000003"
    work.mkdir()
    _two_page_pdf(work / "filing.pdf")
    assert ensure_page_image(work / "p0099.png", page_no=99) is None


def test_upload_does_not_rasterize_eagerly(tmp_path):
    """Serving must not write every page image up front.

    A container's writable filesystem is often memory-backed, so rasterizing a few hundred pages
    during an upload means hundreds of MB of RAM on top of the loaded models — which is what
    OOM-killed the deployed container. The images are derived data we render on demand anyway.
    """
    from sightline.ingest.store import MetadataStore
    from sightline.ingest.upload import ingest_upload

    doc = fitz.open()
    for i in range(12):
        doc.new_page(width=612, height=792).insert_text((72, 100), f"Page {i+1} revenue 60,922")
    data = doc.tobytes()
    doc.close()

    store = MetadataStore(tmp_path / "sightline.db")
    filing, pages = ingest_upload(data, "report.pdf", store, tmp_path)
    store.close()

    assert len(pages) == 12
    assert all(p.text for p in pages)                    # text still extracted for indexing
    written = list((tmp_path / "pages").rglob("*.png"))
    assert written == []                                 # nothing rasterized up front
    assert (tmp_path / "pages" / filing.accession_nodash / "filing.pdf").exists()

    # ...and the images are still obtainable, rendered from the stored PDF on first view.
    got = ensure_page_image(pages[3].image_path, page_no=4)
    assert got is not None and got.exists()
