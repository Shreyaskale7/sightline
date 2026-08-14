"""Filing diffs: what changed on a topic between two filings of the same company.

The other capability a chat window can't reach. Answering "what changed in the revenue discussion
between these two quarters" means holding both filings — hundreds of pages — and finding the two
passages that correspond. In a chat you would paste both documents and hope.

Here both filings are already indexed, so the work is a pinned retrieval on each side:

    accessions = the company's N most recent filings of this form   (newest first)
    for each side: retrieve `topic` pages pinned to THAT accession
    ask for the differences, requiring citations from both sides
    verify against the union of the two page sets

The pinning matters: without it a single search would return whichever quarter embeds closest to
the query, and you would end up "comparing" a filing against itself. Accession-level filtering is
what keeps the two sides genuinely separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .answerer import Answerer, Citation
from .observability import span
from .retrieval.filters import QueryFilters
from .verify import verify

_NO_CHANGE = "NO MATERIAL CHANGE"


@dataclass
class FilingRef:
    accession: str
    label: str            # human-readable, e.g. "10-Q filed 2026-05-20"
    filing_date: str = ""


@dataclass
class DiffResult:
    ticker: str
    topic: str
    older: FilingRef | None = None
    newer: FilingRef | None = None
    summary: str = ""
    citations: list[Citation] = field(default_factory=list)
    no_material_change: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.summary) and not self.error


def _label(form: str, date: str) -> str:
    return f"{form} filed {date}" if date else form


def load_filing_refs(store, ticker: str, form: str, limit: int = 2) -> list[FilingRef]:
    """The company's most recent `limit` filings of one form, newest first."""
    rows = store._conn.execute(
        "SELECT accession, form, filing_date FROM filings "
        "WHERE ticker = ? AND form = ? ORDER BY filing_date DESC LIMIT ?",
        (ticker, form, limit),
    ).fetchall()
    return [
        FilingRef(accession=r["accession"], label=_label(r["form"], r["filing_date"]),
                  filing_date=r["filing_date"])
        for r in rows
    ]


def diff_filings(
    topic: str,
    ticker: str,
    retriever,
    store,
    form: str = "10-Q",
    answerer: Answerer | None = None,
    k: int = 4,
) -> DiffResult:
    """Compare how `ticker` described `topic` in its two most recent `form` filings."""
    answerer = answerer or Answerer()
    result = DiffResult(ticker=ticker, topic=topic)

    with span("diff", ticker=ticker, form=form) as s:
        refs = load_filing_refs(store, ticker, form, limit=2)
        if len(refs) < 2:
            # Honest failure: you cannot diff what was never ingested twice.
            result.error = f"needs two {form} filings for {ticker}, found {len(refs)}"
            s["error"] = result.error
            return result

        newer, older = refs[0], refs[1]
        result.newer, result.older = newer, older

        def _pages_for(ref: FilingRef):
            hits = retriever.retrieve(
                topic, k=k, filter_override=QueryFilters(accessions=[ref.accession])
            )
            return [p for h in hits if (p := store.get_page(h.accession, h.page_no))]

        try:
            old_pages, new_pages = _pages_for(older), _pages_for(newer)
            if not old_pages or not new_pages:
                result.error = "topic not found in one of the two filings"
                return result

            out = answerer.answer_diff(
                topic, ticker, older.label, old_pages, newer.label, new_pages
            )
            # A change claim must be grounded in pages we actually retrieved — from either side.
            verdict = verify(out, {(p.accession, p.page_no) for p in old_pages + new_pages})
            out = verdict.result
            if out.abstained:
                result.error = "no supported differences could be cited"
            else:
                result.no_material_change = out.answer.strip().upper().startswith(_NO_CHANGE)
                result.summary = out.answer.strip()
                result.citations = out.citations
        except Exception as e:
            result.error = f"{type(e).__name__}"
        s["ok"] = result.ok
    return result
