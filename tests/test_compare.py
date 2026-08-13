"""Cross-company comparison: per-company scoping, per-cell verification, per-row isolation.

Built with fakes (no Qdrant, no models, no API spend) so the orchestration itself is what's
under test — which is where the guarantees live.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sightline.answerer import AnswerResult, Citation
from sightline.compare import MAX_COMPANIES, compare, resolve_companies
from sightline.retrieval.text_baseline import Hit


@dataclass
class _Page:
    accession: str
    page_no: int
    text: str
    ticker: str
    form: str = "10-K"
    image_path: Path = Path(".")


# Real SEC accessions are digits-and-dashes, and the citation regex only matches that shape —
# so the fakes use realistic ones. (An earlier version of this test used "acc-NVDA" and the
# citation never parsed, which is the regex behaving correctly on an impossible input.)
_ACC = {
    "NVDA": "0001045810-26-000021",
    "AMD": "0000002488-26-000018",
    "INTC": "0000050863-26-000011",
}


def _acc_for(ticker: str) -> str:
    return _ACC.get(ticker, f"9999999999-99-{abs(hash(ticker)) % 1000000:06d}")


class _FakeRetriever:
    """Records the filter it was called with, and returns that company's page."""

    def __init__(self):
        self.seen_tickers = []

    def retrieve(self, query, k=5, filter_override=None):
        t = filter_override.tickers[0] if filter_override and filter_override.tickers else "?"
        self.seen_tickers.append(t)
        return [Hit(_acc_for(t), 1, 1.0, ticker=t, form="10-K")]


class _FakeStore:
    def __init__(self):
        self._by_acc = {_acc_for(t): t for t in list(_ACC) + ["T0", "T1"]}

    def get_page(self, accession, page_no):
        ticker = self._by_acc.get(accession, "UNK")
        return _Page(accession, page_no, f"{ticker} R&D 1,234", ticker)


class _FakeAnswerer:
    """Returns a cited value per company; can be told to abstain or raise for specific ones."""

    def __init__(self, abstain_for=(), raise_for=()):
        self.abstain_for = set(abstain_for)
        self.raise_for = set(raise_for)

    def answer_cell(self, question, company, pages):
        if company in self.raise_for:
            raise RuntimeError("quota")
        if company in self.abstain_for:
            return AnswerResult(answer="", citations=[], abstained=True)
        acc = pages[0].accession
        return AnswerResult(
            answer=f"$1,234 million [p:{acc}#1]",
            citations=[Citation(accession=acc, page_no=1)],
        )


def test_each_company_is_retrieved_under_its_own_filter():
    r = _FakeRetriever()
    res = compare("R&D spend", ["NVDA", "AMD", "INTC"], r, _FakeStore(), _FakeAnswerer())
    # The per-company filter is what prevents one company's page answering another's cell.
    assert r.seen_tickers == ["NVDA", "AMD", "INTC"]
    assert [row.ticker for row in res.rows] == ["NVDA", "AMD", "INTC"]
    assert res.answered == 3


def test_cell_carries_its_own_citation_and_value_is_tag_free():
    res = compare("R&D spend", ["NVDA"], _FakeRetriever(), _FakeStore(), _FakeAnswerer())
    row = res.rows[0]
    assert row.value == "$1,234 million"           # tag stripped — rendered as a chip instead
    assert "[p:" not in row.value
    assert (row.citations[0].accession, row.citations[0].page_no) == (_acc_for("NVDA"), 1)


def test_one_company_abstaining_does_not_sink_the_table():
    res = compare("R&D spend", ["NVDA", "AMD"], _FakeRetriever(), _FakeStore(),
                  _FakeAnswerer(abstain_for=["AMD"]))
    assert res.rows[0].ok and not res.rows[1].ok
    assert res.rows[1].abstained and res.answered == 1


def test_one_company_failing_does_not_sink_the_table():
    # A quota error on company 2 must still leave rows 1 and 3 usable.
    res = compare("R&D spend", ["NVDA", "AMD", "INTC"], _FakeRetriever(), _FakeStore(),
                  _FakeAnswerer(raise_for=["AMD"]))
    assert res.rows[0].ok and res.rows[2].ok
    assert res.rows[1].error and not res.rows[1].ok
    assert res.answered == 2


def test_fabricated_citation_is_stripped_and_cell_abstains():
    """A cell citing a page we never retrieved is unsupported — the verifier must catch it."""

    class _Fabricator:
        def answer_cell(self, question, company, pages):
            # A real-shaped accession we never retrieved — i.e. the model invented a source.
            return AnswerResult(answer="$9,999 million [p:0000000000-99-000777#7]",
                                citations=[Citation(accession="0000000000-99-000777",
                                                    page_no=7)])

    res = compare("R&D spend", ["NVDA"], _FakeRetriever(), _FakeStore(), _Fabricator())
    assert res.rows[0].abstained and not res.rows[0].ok


def test_resolve_companies_uses_named_then_falls_back_to_corpus():
    corpus = ["AMD", "INTC", "NVDA"]
    assert resolve_companies("q", corpus, named=["NVDA", "AMD"])[0] == ["NVDA", "AMD"]
    # No companies named -> compare across everything indexed.
    assert resolve_companies("compare across every company", corpus, named=[])[0] == corpus
    # A single named company isn't a comparison -> widen to the corpus.
    assert resolve_companies("q", corpus, named=["NVDA"])[0] == corpus


def test_resolve_companies_caps_the_table():
    many = [f"T{i}" for i in range(30)]
    chosen, truncated = resolve_companies("q", many, named=many)
    assert len(chosen) == MAX_COMPANIES and truncated == 30
