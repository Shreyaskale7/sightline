"""OCR fallback: scanned/image-only pages must become indexable text, and the fallback must
degrade gracefully when no OCR backend is present."""
from __future__ import annotations

import fitz

from sightline.ingest import ocr
from sightline.ingest.rasterize import build_pages


def _scanned_one_page_pdf(lines: list[str], path) -> None:
    """Write an image-only PDF (text rasterized to a picture, no text layer) to `path`."""
    tmp = fitz.open()
    tp = tmp.new_page(width=612, height=792)
    y = 100
    for ln in lines:
        tp.insert_text((72, y), ln, fontsize=16)
        y += 32
    pix = tp.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
    tmp.close()
    out = fitz.open()
    out.new_page(width=612, height=792).insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pix)
    out.save(str(path))
    out.close()


def test_scanned_pdf_has_no_text_layer(tmp_path):
    # Guards the fixture itself: the "scanned" page must truly have no extractable text,
    # otherwise the OCR test below would pass for the wrong reason.
    pdf = tmp_path / "scan.pdf"
    _scanned_one_page_pdf(["Amount due: $12,800.00"], pdf)
    doc = fitz.open(pdf)
    assert doc[0].get_text().strip() == ""
    doc.close()


def test_build_pages_ocrs_scanned_page(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _scanned_one_page_pdf(["Invoice No. 4471", "Amount due: 12800 dollars"], pdf)

    pages = build_pages(pdf, tmp_path / "out", accession="0000000000-99-000001")
    assert len(pages) == 1
    # OCR recovered enough text to make the page indexable (empty text -> dropped from the index).
    assert pages[0].text.strip() != ""
    assert "4471" in pages[0].text.replace(" ", "")


def test_ocr_image_degrades_gracefully(monkeypatch, tmp_path):
    # If no OCR backend is available, ocr_image returns "" rather than raising — ingestion of a
    # scanned page then just yields an unindexed page, exactly the pre-OCR behaviour.
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_unavailable", True)
    assert ocr.ocr_image(tmp_path / "does-not-matter.png") == ""
