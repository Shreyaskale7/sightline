"""Verifier tests — the no-uncited-answer guarantee is only as good as its test coverage."""
from __future__ import annotations

from sightline.answerer import AnswerResult, Citation
from sightline.verify import verify

_PAGES = {("0001045810-26-000021", 51), ("0000050863-26-000011", 72)}


def _result(citations: list[Citation], abstained: bool = False) -> AnswerResult:
    return AnswerResult(answer="Revenue was $215,938M.", citations=citations, abstained=abstained)


def test_valid_citation_passes_through():
    r = _result([Citation("0001045810-26-000021", 51)])
    v = verify(r, _PAGES)
    assert not v.forced_abstain and v.result is r


def test_uncited_answer_is_forced_to_abstain():
    v = verify(_result([]), _PAGES)
    assert v.forced_abstain and v.result.abstained and v.result.citations == []


def test_fabricated_citation_is_dropped_keeping_valid_ones():
    r = _result([
        Citation("0001045810-26-000021", 51),   # real
        Citation("9999999999-99-999999", 1),    # fabricated
    ])
    v = verify(r, _PAGES)
    assert not v.forced_abstain
    assert v.dropped_citations == 1
    assert [(c.accession, c.page_no) for c in v.result.citations] == [("0001045810-26-000021", 51)]


def test_all_fabricated_citations_forces_abstention():
    v = verify(_result([Citation("9999999999-99-999999", 1)]), _PAGES)
    assert v.forced_abstain and v.result.abstained and v.dropped_citations == 1


def test_abstention_passes_untouched():
    r = AnswerResult(answer="I don't know.", citations=[], abstained=True)
    v = verify(r, _PAGES)
    assert not v.forced_abstain and v.result is r
