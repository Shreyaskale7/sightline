"""OCR fallback for scanned / image-only pages.

Sightline's prefilter retrieves on each page's *extracted* text. A page from a normal PDF has
a text layer, so `page.get_text()` returns it for free. But many real-world uploads — scanned
contracts, faxed invoices, image-only exports — have NO text layer: `get_text()` returns "",
the page is dropped from the index (see TextRetriever.index), and it becomes unretrievable even
though the answer path could read its image. OCR recovers that text so the page gets indexed.

This is a *fallback*, not the primary path: it runs only when a page's text layer is empty, so
normal PDFs (all the SEC filings) never pay its cost. Backend is RapidOCR (ONNX/onnxruntime —
no torch, no system Tesseract binary, models bundled in the wheel → works offline on CPU). If
the backend isn't installed the function returns "" and ingestion degrades gracefully rather
than crashing (e.g. in a minimal CI image).
"""
from __future__ import annotations

from pathlib import Path

# The OCR engine loads its ONNX models once (~a second) and is reused across pages.
_engine = None
_unavailable = False


def _get_engine():
    global _engine, _unavailable
    if _engine is not None or _unavailable:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    except Exception as e:  # not installed, or model load failed -> disable, don't crash ingest
        _unavailable = True
        print(f"[ocr] disabled ({type(e).__name__}: {e}); scanned pages won't be indexed")
    return _engine


def ocr_image(image_path: str | Path) -> str:
    """Return recognized text for a page image (newline-joined lines), or "" if OCR is
    unavailable or the page has no legible text."""
    engine = _get_engine()
    if engine is None:
        return ""
    result, _ = engine(str(image_path))  # result: list of [box, text, confidence] or None
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()
