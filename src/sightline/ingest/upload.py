"""User-uploaded PDFs → the same page-image pipeline as EDGAR filings (M6).

An upload is just a filing we didn't download: rasterize each page to PNG + extract its text
(build_pages), store idempotently, index. Two upload-specific concerns live here:

1. **Citation-safe identity.** Citations parse tags of the form [p:ACCESSION#PAGE] where
   ACCESSION is digits-and-dashes (real SEC accession numbers). A synthetic accession like
   "upload-abc12" would make every citation to an uploaded document silently unparseable —
   so uploads get a deterministic digits-and-dashes accession derived from the file's hash
   (same bytes → same accession → idempotent re-upload, exactly like EDGAR's accession key).

2. **Guardrails.** Size and page caps keep a public demo from being handed a 2,000-page PDF
   that pins the CPU for an hour.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from .edgar import Filing
from .rasterize import Page, build_pages
from .store import MetadataStore

MAX_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_PAGES = 150

_TICKER_OK = re.compile(r"[^A-Z0-9]")


class UploadError(ValueError):
    """User-correctable problem with an upload (wrong type, too big, unreadable)."""


def synthetic_accession(data: bytes) -> str:
    """Deterministic, citation-parseable accession for uploaded bytes.

    Shaped like a real accession (##########-##-######) so the citation regex, page-image
    routes, and idempotent storage all treat uploads exactly like EDGAR filings. The "99"
    century marker can't collide with real SEC accessions (which encode a filing year).
    """
    h = hashlib.sha256(data).hexdigest()
    a = int(h[:12], 16) % 10_000_000_000
    b = int(h[12:20], 16) % 1_000_000
    return f"{a:010d}-99-{b:06d}"


def label_from_filename(filename: str) -> str:
    """A short uppercase label used where corpus docs show a ticker (e.g. 'ANNUAL-REPORT.pdf'
    → 'ANNUAL'). Display identity only — never used for retrieval filtering."""
    stem = Path(filename or "DOC").stem.upper()
    clean = _TICKER_OK.sub("", stem)[:6]
    return clean or "DOC"


def ingest_upload(
    pdf_bytes: bytes,
    filename: str,
    store: MetadataStore,
    data_dir: Path,
) -> tuple[Filing, list[Page]]:
    """Persist an uploaded PDF as pages (images + text). Idempotent on content hash.

    Returns the Filing record and its pages; the caller indexes the pages (indexing needs
    the shared Qdrant client, which belongs to the serving layer, not ingestion).
    """
    if not pdf_bytes:
        raise UploadError("The file is empty.")
    if len(pdf_bytes) > MAX_BYTES:
        raise UploadError(f"File is {len(pdf_bytes) // (1024*1024)} MB — the limit is 25 MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise UploadError("Only PDF files are supported — this doesn't look like a PDF.")

    accession = synthetic_accession(pdf_bytes)
    filing = Filing(
        cik=0,
        ticker=label_from_filename(filename),
        form="UPLOAD",
        filing_date=date.today().isoformat(),
        accession=accession,
        primary_document=Path(filename or "document.pdf").name,
    )

    work_dir = Path(data_dir) / "pages" / filing.accession_nodash
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = work_dir / "filing.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pages = build_pages(pdf_path, work_dir, accession)
    if not pages:
        raise UploadError("Couldn't read any pages from this PDF.")
    if len(pages) > MAX_PAGES:
        raise UploadError(f"{len(pages)} pages — the limit is {MAX_PAGES} for uploads.")

    store.save_filing_with_pages(filing, pages)
    return filing, pages
