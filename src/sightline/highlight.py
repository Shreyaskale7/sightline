"""Region highlighting: find WHERE on a cited page the answer's facts live.

The demo's payoff feature. Our answer path is text-grounded, so we can do something more honest
than a heatmap: take the salient tokens of the answer (dollar figures, numbers, proper nouns),
locate them in the cited page's text layer (PyMuPDF search over the stored PDF), and return
bounding boxes. The UI overlays them on the page image — so a citation isn't just "page 51",
it's "page 51, this box, right here."

Boxes are returned NORMALIZED (fractions of page width/height) so the browser can scale them to
whatever size it renders the image at, with no DPI coupling.
"""
from __future__ import annotations

import re
from pathlib import Path

# Financial figures ("215,938", "$18,497", "25.03") and ALL-CAPS/Proper tokens ("TSMC", "Intel").
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_PROPER = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]{3,})\b")
_STOP = {"The", "This", "That", "According", "NVIDIA", "About", "Approximately", "What"}


def salient_terms(answer: str, max_terms: int = 8) -> list[str]:
    """Extract the fact-bearing tokens worth locating on the page (dedup, order-preserving)."""
    terms: list[str] = []
    seen: set[str] = set()
    for m in _NUMBER.findall(answer):
        t = m.lstrip("$")
        if len(t.replace(",", "").replace(".", "")) >= 3 and t not in seen:  # skip tiny numbers
            seen.add(t)
            terms.append(t)
    for m in _PROPER.findall(answer):
        if m not in _STOP and m not in seen:
            seen.add(m)
            terms.append(m)
    return terms[:max_terms]


def _pdf_for(image_path: Path) -> Path:
    # PNGs live at data/pages/{acc}/pNNNN.png alongside the rendered filing.pdf.
    return Path(image_path).parent / "filing.pdf"


def page_boxes(image_path: Path, page_no: int, terms: list[str], max_boxes: int = 12
               ) -> list[list[float]]:
    """Return normalized [x0,y0,x1,y1] boxes (0-1) for `terms` on the given 1-based page.

    Silent-empty on any failure (missing PDF, no text layer) — highlighting is a nice-to-have
    that must never break the answer path.
    """
    pdf = _pdf_for(image_path)
    if not pdf.exists() or not terms:
        return []
    try:
        import fitz

        doc = fitz.open(pdf)
        try:
            if not (1 <= page_no <= doc.page_count):
                return []
            page = doc[page_no - 1]
            w, h = page.rect.width, page.rect.height
            if w <= 0 or h <= 0:
                return []
            boxes: list[list[float]] = []
            for term in terms:
                for r in page.search_for(term, quads=False):
                    boxes.append([round(r.x0 / w, 4), round(r.y0 / h, 4),
                                  round(r.x1 / w, 4), round(r.y1 / h, 4)])
                    if len(boxes) >= max_boxes:
                        return boxes
            return boxes
        finally:
            doc.close()
    except Exception:
        return []
