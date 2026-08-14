"""Filing diffs: the two sides must stay genuinely separate, and change claims must be grounded.

The failure mode this guards against is subtle — without accession-level pinning a single search
returns whichever period embeds closest, and the system "compares" a filing against itself while
looking perfectly healthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sightline.answerer import AnswerResult, Citation
from sightline.diff import diff_filings
from sightline.retrieval.text_baseline import Hit

OLD = "0001045810-25-000100"
NEW = "0001045810-26-000200"


@dataclass
class _Page:
    accession: str
    page_no: int
    text: str
    ticker: str = "NVDA"
    form: str = "10-Q"
    image_path: Path = Path(".")


class _FakeStore:
    """Two 10-Qs for NVDA; a single 10-K (so the not-enough-filings path is testable)."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [
            {"accession": NEW, "form": "10-Q", "filing_date": "2026-05-20"},
            {"accession": OLD, "form": "10-Q", "filing_date": "2025-11-19"},
        ]

        class _Cursor:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        class _Conn:
            def __init__(self, outer):
                self.outer = outer

            def execute(self, q, args):
                _ticker, form, limit = args
                rows = [r for r in self.outer.rows if r["form"] == form]
                rows.sort(key=lambda r: r["filing_date"], reverse=True)  # newest first, as SQL does
                return _Cursor(rows[:limit])

        self._conn = _Conn(self)

    def get_page(self, accession, page_no):
        return _Page(accession, page_no, f"text from {accession}")


class _FakeRetriever:
    def __init__(self):
        self.pinned = []

    def retrieve(self, query, k=5, filter_override=None):
        acc = filter_override.accessions[0]
        self.pinned.append(acc)
        return [Hit(acc, 7, 1.0, ticker="NVDA", form="10-Q")]


class _FakeAnswerer:
    def __init__(self, answer=None):
        self.answer = answer or (
            f"- Revenue guidance raised [p:{OLD}#7] [p:{NEW}#7]"
        )
        self.seen = None

    def answer_diff(self, topic, company, old_label, old_pages, new_label, new_pages):
        self.seen = (old_label, new_label, [p.accession for p in old_pages],
                     [p.accession for p in new_pages])
        return AnswerResult(
            answer=self.answer,
            citations=[Citation(OLD, 7), Citation(NEW, 7)],
        )


def test_each_side_is_pinned_to_its_own_filing():
    r, a = _FakeRetriever(), _FakeAnswerer()
    res = diff_filings("revenue", "NVDA", r, _FakeStore(), answerer=a)
    # Older retrieved first, then newer — each pinned to its own accession, never mixed.
    assert r.pinned == [OLD, NEW]
    _, _, old_accs, new_accs = a.seen
    assert old_accs == [OLD] and new_accs == [NEW]
    assert res.ok and res.older.accession == OLD and res.newer.accession == NEW


def test_newest_filing_is_the_later_side():
    res = diff_filings("revenue", "NVDA", _FakeRetriever(), _FakeStore(), answerer=_FakeAnswerer())
    assert res.newer.filing_date > res.older.filing_date


def test_cites_both_sides():
    res = diff_filings("revenue", "NVDA", _FakeRetriever(), _FakeStore(), answerer=_FakeAnswerer())
    accs = {c.accession for c in res.citations}
    assert accs == {OLD, NEW}          # a change claim needs support from both filings


def test_no_material_change_is_a_valid_finding():
    a = _FakeAnswerer(answer=f"NO MATERIAL CHANGE [p:{OLD}#7] [p:{NEW}#7]")
    res = diff_filings("revenue", "NVDA", _FakeRetriever(), _FakeStore(), answerer=a)
    assert res.no_material_change and res.ok


def test_ungrounded_change_claim_is_rejected():
    """Citing a page from neither filing means the 'change' was invented."""

    class _Fabricator:
        def answer_diff(self, *a, **k):
            return AnswerResult(answer="- Something changed [p:0000000000-99-000777#1]",
                                citations=[Citation("0000000000-99-000777", 1)])

    res = diff_filings("revenue", "NVDA", _FakeRetriever(), _FakeStore(), answerer=_Fabricator())
    assert not res.ok and "cited" in res.error


def test_needs_two_filings():
    store = _FakeStore(rows=[{"accession": NEW, "form": "10-K", "filing_date": "2026-02-25"}])
    res = diff_filings("risk factors", "NVDA", _FakeRetriever(), store, form="10-K",
                       answerer=_FakeAnswerer())
    assert not res.ok and "found 1" in res.error
