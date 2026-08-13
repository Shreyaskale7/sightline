"""Corpus-wide screening: a YES must be evidenced, cited, and verified.

The value of a screen is that you can trust the shortlist. So the tests pin the behaviours that
make it trustworthy: hits carry the sentence that justifies them, a hit citing a page we never
retrieved is not a hit, and one company failing can't corrupt the rest of the list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sightline.answerer import AnswerResult, Citation
from sightline.compare import screen
from sightline.retrieval.text_baseline import Hit

_ACC = {"NVDA": "0001045810-26-000021", "AMD": "0000002488-26-000018",
        "INTC": "0000050863-26-000011"}


def _acc(t):
    return _ACC.get(t, "9999999999-99-000000")


@dataclass
class _Page:
    accession: str
    page_no: int
    text: str
    ticker: str
    form: str = "10-K"
    image_path: Path = Path(".")


class _FakeRetriever:
    def __init__(self):
        self.seen = []

    def retrieve(self, query, k=5, filter_override=None):
        t = filter_override.tickers[0] if filter_override and filter_override.tickers else "?"
        self.seen.append(t)
        return [Hit(_acc(t), 3, 1.0, ticker=t, form="10-K")]


class _FakeStore:
    def get_page(self, accession, page_no):
        return _Page(accession, page_no, "risk text", "X")


class _FakeAnswerer:
    def __init__(self, yes=(), raise_for=()):
        self.yes, self.raise_for = set(yes), set(raise_for)

    def screen_cell(self, criterion, company, pages):
        if company in self.raise_for:
            raise RuntimeError("quota")
        if company in self.yes:
            acc = pages[0].accession
            return True, AnswerResult(
                answer=f"YES — relies on a single foundry partner [p:{acc}#3]",
                citations=[Citation(accession=acc, page_no=3)])
        return False, AnswerResult(answer="NO", citations=[])


def test_returns_only_evidenced_matches():
    res = screen("foundry dependency", ["NVDA", "AMD", "INTC"], _FakeRetriever(), _FakeStore(),
                 _FakeAnswerer(yes=["NVDA", "INTC"]))
    assert [r.ticker for r in res.matches] == ["NVDA", "INTC"]
    hit = res.matches[0]
    assert "foundry" in hit.evidence            # the justifying sentence is kept
    assert not hit.evidence.startswith("YES")   # verdict token stripped from the evidence
    assert "[p:" not in hit.evidence            # tag rendered as a chip, not inline
    assert hit.citations[0].page_no == 3


def test_each_company_screened_under_its_own_filter():
    r = _FakeRetriever()
    screen("x", ["NVDA", "AMD"], r, _FakeStore(), _FakeAnswerer())
    assert r.seen == ["NVDA", "AMD"]


def test_unverifiable_yes_is_not_a_match():
    """A YES citing a page we never retrieved is the model asserting, not finding."""

    class _Fabricator:
        def screen_cell(self, criterion, company, pages):
            return True, AnswerResult(answer="YES — made up [p:0000000000-99-000777#9]",
                                      citations=[Citation("0000000000-99-000777", 9)])

    res = screen("x", ["NVDA"], _FakeRetriever(), _FakeStore(), _Fabricator())
    assert res.matches == []


def test_one_failure_does_not_corrupt_the_list():
    res = screen("x", ["NVDA", "AMD", "INTC"], _FakeRetriever(), _FakeStore(),
                 _FakeAnswerer(yes=["NVDA", "INTC"], raise_for=["AMD"]))
    assert [r.ticker for r in res.matches] == ["NVDA", "INTC"]
    assert res.rows[1].error and not res.rows[1].matched
