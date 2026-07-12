"""Highlight tests — salient-term extraction (pure) and box location on a real generated PDF."""
from __future__ import annotations

from sightline.highlight import page_boxes, salient_terms


def test_salient_terms_pulls_figures_and_proper_nouns():
    terms = salient_terms("Revenue was $215,938 million and R&D was 18,497 [p:x#51].")
    assert "215,938" in terms and "18,497" in terms


def test_salient_terms_skips_tiny_numbers_and_dedupes():
    terms = salient_terms("It rose to 5 then 5 then $44,284.")
    assert "5" not in terms
    assert terms.count("44,284") == 1


def test_salient_terms_keeps_tickers():
    terms = salient_terms("Manufactured by TSMC and Samsung.")
    assert "TSMC" in terms and "Samsung" in terms


def test_page_boxes_locates_text_in_generated_pdf(tmp_path):
    import fitz

    # Build a one-page PDF with a known figure, mimicking data/pages/{acc}/filing.pdf layout.
    work = tmp_path / "pages" / "acc1"
    work.mkdir(parents=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter points
    page.insert_text((100, 200), "Total revenue $215,938 million", fontsize=12)
    doc.save(work / "filing.pdf")
    doc.close()

    boxes = page_boxes(work / "p0001.png", 1, ["215,938"])
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1  # normalized, well-formed
    assert x0 < 0.3            # text starts near left (x=100/612 ≈ 0.16)
    assert 0.2 < y0 < 0.3      # near y=200/792 ≈ 0.25


def test_page_boxes_empty_when_no_pdf(tmp_path):
    assert page_boxes(tmp_path / "nope" / "p0001.png", 1, ["anything"]) == []


def test_page_boxes_empty_for_missing_term(tmp_path):
    import fitz

    work = tmp_path / "pages" / "acc1"
    work.mkdir(parents=True)
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), "nothing relevant")
    doc.save(work / "filing.pdf")
    doc.close()
    assert page_boxes(work / "p0001.png", 1, ["215,938"]) == []
