"""Turn filing documents into page images + per-page metadata.

The core Sightline move: we treat every page as an IMAGE and retrieve on it directly,
so tables/charts/layout survive. OCR-and-chunk is deliberately NOT the primary path.

Flow:
  1. Normalize the primary document to PDF. EDGAR 10-K/10-Q primary docs are HTML, so we
     render them with headless Chromium (Playwright). Chromium honors the filing's own page
     breaks, so our page numbers line up with the document's intended pages.
  2. Rasterize each PDF page to PNG at ~150-200 DPI, and in the SAME pass pull that page's
     text. We use PyMuPDF (`fitz`) for both: one library, no system deps (poppler-free), and
     image + text share the exact same page boundary -- which keeps citations honest.
  3. The extracted text is ONLY for the BM25/dense prefilter side of hybrid retrieval, never
     for the final answer (the answer path reads page images).

Dependency notes (see CLAUDE.md "explain any new heavy dependency"):
  - Playwright (headless Chromium) is the heavy one. It is the most faithful renderer for the
    multi-column tables in filings and ships its own browser, so there is no fragile system
    dependency on Windows. weasyprint would need GTK native libs; wkhtmltopdf is deprecated.
  - PyMuPDF replaces the plan's pdf2image+poppler: pure pip install, does raster AND text in
    one pass. (License: PyMuPDF is AGPL -- fine for a personal/portfolio project.)

Imports of the heavy deps are deferred into the functions that need them, so importing this
module (and unit-testing the metadata store) works even before the browser is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    accession: str
    page_no: int          # 1-based
    image_path: Path
    text: str             # extracted text for the prefilter side only


def html_to_pdf(html_bytes: bytes, out_pdf: Path) -> Path:
    """Render an EDGAR HTML filing to a paginated PDF with headless Chromium.

    We load the HTML *offline* via ``set_content`` and abort any outbound http(s) subresource
    request. Two reasons: (1) it keeps us strictly within SEC rate-limiting etiquette -- only
    our rate-limited EdgarClient ever touches sec.gov; (2) it makes rendering deterministic.
    The cost is that externally-referenced images (some charts/logos) render blank in M1; that
    is acceptable for the text baseline and is a TODO(M2) once we fetch assets through the
    rate limiter for the visual path.
    """
    from playwright.sync_api import sync_playwright

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    html = html_bytes.decode("utf-8", errors="replace")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(60_000)
            # Block outbound network so we never bypass the EDGAR rate limiter.
            page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.url.startswith(("http://", "https://"))
                    else route.continue_()
                ),
            )
            page.set_content(html, wait_until="load")
            page.pdf(path=str(out_pdf), format="Letter", print_background=True)
        finally:
            browser.close()
    return out_pdf


def _render_page(page, out_path: Path, dpi: int) -> None:
    """Rasterize a single PyMuPDF page to PNG at the given DPI (72 dpi == 1.0 zoom)."""
    import fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
    pix.save(str(out_path))


def rasterize_pdf(pdf_path: Path, out_dir: Path, dpi: int = 175) -> list[Path]:
    """Rasterize each PDF page to a PNG. Returns image paths in page order (1-based names)."""
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            img_path = out_dir / f"p{i:04d}.png"
            _render_page(page, img_path, dpi)
            paths.append(img_path)
    finally:
        doc.close()
    return paths


def build_pages(pdf_path: Path, out_dir: Path, accession: str, dpi: int = 175) -> list[Page]:
    """Produce Page records (image + aligned text) for one filing, in one PDF pass.

    Doing the raster and the text extraction in the same loop guarantees the PNG and the text
    for a given page_no describe the *same* page -- so a retrieval hit on the text side always
    points at the right image.
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            img_path = out_dir / f"p{i:04d}.png"
            _render_page(page, img_path, dpi)
            pages.append(
                Page(
                    accession=accession,
                    page_no=i,
                    image_path=img_path,
                    text=page.get_text().strip(),
                )
            )
    finally:
        doc.close()
    return pages
