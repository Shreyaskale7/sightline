"""Upload-path benchmark: measure retrieval on USER documents, not the SEC corpus.

The product's real use is arbitrary uploaded PDFs, but every headline metric is on the SEC
golden set. This harness closes that gap with a tiny, self-contained benchmark: it *generates*
its fixtures (so nothing binary is checked in), ingests them exactly as an upload, indexes them,
and scores first-stage Recall@k on the uploaded pages.

Crucially it includes a **scanned / image-only** document (text rendered to a picture, so the
PDF has no text layer). That is the case the text-layer pipeline silently fails on — the page
carries no extractable text, so it is never indexed, so it can never be retrieved. The benchmark
makes that failure a number, and the OCR fallback's lift a measured one.

Run:  python -m sightline.eval.upload_bench
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Case:
    filename: str          # which fixture document holds the answer
    question: str
    gold_page: int         # 1-based page number where the answer lives
    kind: str              # "text" (has a text layer) | "scanned" (image-only, needs OCR)


# --- fixture generation ------------------------------------------------------

def _text_pdf(pages: list[list[str]]) -> bytes:
    """A normal PDF with a real text layer (get_text() returns the lines)."""
    import fitz

    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=612, height=792)
        y = 96
        for ln in lines:
            page.insert_text((72, y), ln, fontsize=13)
            y += 26
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf(pages: list[list[str]]) -> bytes:
    """An image-only PDF: each page's text is rasterized to a picture and placed as an image,
    so the PDF has NO text layer — get_text() returns "" (a real scanned-document stand-in)."""
    import fitz

    doc = fitz.open()
    for lines in pages:
        tmp = fitz.open()
        tp = tmp.new_page(width=612, height=792)
        y = 96
        for ln in lines:
            tp.insert_text((72, y), ln, fontsize=15)
            y += 30
        pix = tp.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))  # 150 DPI render
        tmp.close()
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


# The documents deliberately look like things a real user uploads (a contract, a board summary,
# a scanned invoice) — none of it is SEC data, which is the whole point.
_DOCS: dict[str, tuple[str, list[list[str]]]] = {
    "services-agreement.pdf": ("text", [
        ["MASTER SERVICES AGREEMENT", "Between Northwind Labs and Contoso Ltd."],
        ["2. FEES", "Total Contract Value: $2,450,000 over the 36-month term.",
         "Invoicing is quarterly in advance."],
        ["4. TERMINATION", "Either party may terminate this agreement",
         "with ninety (90) days prior written notice."],
        ["SIGNATURES", "Signed for and on behalf of the parties."],
    ]),
    "q3-board-summary.pdf": ("text", [
        ["Q3 BOARD SUMMARY", "Prepared for the board of directors."],
        ["FINANCIAL HIGHLIGHTS",
         "Net revenue for the quarter was $18.7 million, up 12% year over year.",
         "Gross margin held at 61%."],
        ["LIQUIDITY", "Cash and cash equivalents totaled $46.2 million at quarter end.",
         "The company remains debt-free."],
    ]),
    "scanned-invoice.pdf": ("scanned", [
        ["INVOICE  No. 4471", "Amount due: $12,800.00", "Due date: March 15, 2026."],
        ["REMITTANCE", "Remit payment to Acme Holdings,", "account ending 3391."],
    ]),
}

_CASES: list[Case] = [
    Case("services-agreement.pdf", "What is the total contract value?", 2, "text"),
    Case("services-agreement.pdf", "How many days notice are needed to terminate?", 3, "text"),
    Case("q3-board-summary.pdf", "What was net revenue for the quarter?", 2, "text"),
    Case("q3-board-summary.pdf", "How much cash and cash equivalents at quarter end?", 3, "text"),
    Case("scanned-invoice.pdf", "What is the amount due on the invoice?", 1, "scanned"),
    Case("scanned-invoice.pdf", "Which account should payment be remitted to?", 2, "scanned"),
]


def build_fixtures() -> list[tuple[str, bytes]]:
    """(filename, pdf_bytes) for every benchmark document."""
    out = []
    for filename, (kind, pages) in _DOCS.items():
        out.append((filename, _scanned_pdf(pages) if kind == "scanned" else _text_pdf(pages)))
    return out


# --- run ---------------------------------------------------------------------

def run(k: int = 5) -> dict:
    """Ingest + index the fixtures in a temp workspace, score Recall@k by document kind."""
    from ..ingest.store import MetadataStore
    from ..ingest.upload import ingest_upload
    from ..retrieval.text_baseline import TextRetriever

    work = Path(tempfile.mkdtemp(prefix="upload_bench_"))
    store = MetadataStore(work / "sightline.db")
    retr = TextRetriever(
        chunked=True, qdrant_location=str(work / "qdrant"), collection="upload_bench"
    )
    acc_by_doc: dict[str, str] = {}
    for filename, data in build_fixtures():
        filing, _ = ingest_upload(data, filename, store, work)
        acc_by_doc[filename] = filing.accession
        retr.index(store.iter_pages_for(filing.accession))

    rows = []
    for c in _CASES:
        hits = retr.retrieve(c.question, k=k)
        gold = (acc_by_doc[c.filename], c.gold_page)
        found = any((h.accession, h.page_no) == gold for h in hits)
        rows.append((c, found))

    retr.close()
    store.close()

    def recall(kind: str | None) -> float:
        sub = [f for c, f in rows if kind is None or c.kind == kind]
        return sum(sub) / len(sub) if sub else 0.0

    return {
        "k": k,
        "overall": recall(None),
        "text": recall("text"),
        "scanned": recall("scanned"),
        "rows": rows,
    }


def main() -> None:
    r = run()
    print(f"\nUpload-path Recall@{r['k']}  (self-contained fixtures)\n")
    for c, found in r["rows"]:
        mark = "HIT " if found else "MISS"
        print(f"  [{mark}] ({c.kind:7}) {c.question}")
    print(f"\n  text docs   : {r['text']:.3f}")
    print(f"  scanned docs: {r['scanned']:.3f}   (image-only pages — reachable only via OCR)")
    print(f"  OVERALL     : {r['overall']:.3f}\n")


if __name__ == "__main__":
    main()
