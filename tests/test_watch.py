"""Corpus freshness: what the SEC has that we don't.

The property that matters is that staleness is decided by the accession key — the same key
ingestion is idempotent on — not by dates or counts, and that one bad ticker can't quietly
hide the rest of the report.
"""
from __future__ import annotations

from sightline.ingest.edgar import Filing
from sightline.watch import find_new_filings


def _filing(ticker, form, acc, date):
    return Filing(cik=1, ticker=ticker, form=form, filing_date=date,
                  accession=acc, primary_document="d.htm")


class _FakeClient:
    def __init__(self, by_ticker, cik_map=None, raise_for=(), raise_map=False):
        self.by_ticker = by_ticker
        self.cik_map = cik_map or {t: i for i, t in enumerate(by_ticker, start=1)}
        self.raise_for = set(raise_for)
        self.raise_map = raise_map

    def ticker_to_cik(self):
        if self.raise_map:
            raise ConnectionError("sec unreachable")
        return self.cik_map

    def get_filings(self, cik, ticker, forms=("10-K", "10-Q"), limit=None, since=None):
        if ticker in self.raise_for:
            raise TimeoutError("slow")
        return [f for f in self.by_ticker.get(ticker, []) if f.form in forms][:limit]


class _FakeStore:
    def __init__(self, held):
        self.held = set(held)

    def is_ingested(self, accession):
        return accession in self.held


def test_reports_only_filings_we_do_not_hold():
    have, missing = "0001-26-000001", "0001-26-000002"
    client = _FakeClient({"NVDA": [_filing("NVDA", "10-K", have, "2026-02-25"),
                                   _filing("NVDA", "10-Q", missing, "2026-05-20")]})
    rep = find_new_filings(client, _FakeStore([have]), ["NVDA"])
    assert [f.accession for f in rep.new_filings] == [missing]
    assert rep.is_stale and rep.checked == ["NVDA"]


def test_fully_current_corpus_is_not_stale():
    acc = "0001-26-000001"
    client = _FakeClient({"NVDA": [_filing("NVDA", "10-K", acc, "2026-02-25")]})
    rep = find_new_filings(client, _FakeStore([acc]), ["NVDA"])
    assert rep.new_filings == [] and not rep.is_stale


def test_form_filter_is_respected():
    client = _FakeClient({"NVDA": [_filing("NVDA", "8-K", "a", "2026-01-01"),
                                   _filing("NVDA", "10-K", "b", "2026-02-25")]})
    rep = find_new_filings(client, _FakeStore([]), ["NVDA"], forms=("10-K",))
    assert [f.accession for f in rep.new_filings] == ["b"]


def test_unknown_ticker_is_reported_not_raised():
    client = _FakeClient({"NVDA": []}, cik_map={"NVDA": 1})
    rep = find_new_filings(client, _FakeStore([]), ["NVDA", "NOPE"])
    assert rep.errors["NOPE"] == "unknown ticker at SEC"
    assert rep.checked == ["NVDA"]          # the good ticker still got checked


def test_one_failing_ticker_does_not_hide_the_others():
    client = _FakeClient(
        {"NVDA": [_filing("NVDA", "10-K", "new-1", "2026-02-25")],
         "AMD": [_filing("AMD", "10-K", "new-2", "2026-02-04")]},
        raise_for=["AMD"],
    )
    rep = find_new_filings(client, _FakeStore([]), ["NVDA", "AMD"])
    assert [f.accession for f in rep.new_filings] == ["new-1"]
    assert rep.errors["AMD"] == "TimeoutError"


def test_sec_unreachable_degrades_to_an_error_report():
    rep = find_new_filings(_FakeClient({}, raise_map=True), _FakeStore([]), ["NVDA"])
    assert rep.errors["*"] == "ConnectionError"
    assert not rep.is_stale                  # unknown is not the same as up to date
