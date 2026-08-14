"""Numeric grounding: right page, wrong number is the hallucination the verifier can't see.

verify.py proves a citation points at a retrieved page. These tests pin the next layer down —
that the figure the answer states is actually present on the page it points at, and that
formatting differences (commas, units, rounding) are not mistaken for fabrication.
"""
from __future__ import annotations

from sightline.grounding import check_answer, extract_figures

_STATEMENT = (
    "CONSOLIDATED STATEMENTS OF INCOME\n"
    "Revenue 60,922 26,974\n"
    "Cost of revenue 16,621 11,618\n"
    "Gross profit 44,301 15,356\n"
    "Research and development 8,675 7,339\n"
    "Net income 29,760 4,368"
)
_PAGE = ("0001045810-26-000021", 51)


def test_extracts_claim_figures_and_ignores_prose_counts():
    figs = dict(extract_figures("Revenue was 60,922 across 38 countries and 3 segments."))
    assert "60,922" in figs
    # Small integers are prose, not claims a citation is expected to carry.
    assert "38" not in figs and "3" not in figs


def test_citation_tags_are_not_read_as_claims():
    """An accession is a long digit string. Left in, "[p:0001045810-26-000021#51]" reads as the
    figures 0001045810 / 000021 / 51 — and every correctly-cited answer looks fabricated."""
    rep = check_answer(
        "Revenue was 60,922 million [p:0001045810-26-000021#51].", {_PAGE: _STATEMENT}
    )
    assert [f.text for f in rep.figures] == ["60,922"]
    assert rep.all_supported


def test_supported_figure_is_matched_to_its_page():
    rep = check_answer("Revenue was $60,922 million.", {_PAGE: _STATEMENT})
    assert rep.all_supported
    assert rep.figures[0].supported_by == [_PAGE]


def test_unit_rescaling_is_not_a_hallucination():
    # "60.9 billion" is a faithful restatement of a statement line reading 60,922 (millions).
    rep = check_answer("Revenue was approximately $60.9 billion.", {_PAGE: _STATEMENT})
    assert rep.all_supported


def test_right_page_wrong_number_is_caught():
    """The hallucination verify.py cannot see: a real citation carrying a fabricated figure."""
    rep = check_answer("Revenue was $70,500 million.", {_PAGE: _STATEMENT})
    assert not rep.all_supported
    assert [f.text for f in rep.unsupported] == ["70,500"]


def test_partially_supported_answer_flags_only_the_bad_figure():
    rep = check_answer(
        "Revenue was 60,922 and R&D was 9,999.", {_PAGE: _STATEMENT}
    )
    assert [f.text for f in rep.unsupported] == ["9,999"]
    assert any(f.text == "60,922" and f.supported for f in rep.figures)


def test_conflicting_cited_pages_are_surfaced():
    other = ("0001045810-25-000230", 12)
    pages = {
        _PAGE: _STATEMENT,
        other: "CONSOLIDATED STATEMENTS OF INCOME\nRevenue 44,100 20,000\nNet income 19,300 3,000",
    }
    rep = check_answer("Revenue was 60,922.", pages)
    assert any("disagree" in d for d in rep.disagreements)


def test_agreeing_pages_raise_no_disagreement():
    dup = ("0001045810-26-000021", 52)
    rep = check_answer("Revenue was 60,922.", {_PAGE: _STATEMENT, dup: _STATEMENT})
    assert rep.disagreements == []


def test_answer_with_no_figures_is_trivially_grounded():
    rep = check_answer("The company relies on third-party foundries.", {_PAGE: _STATEMENT})
    assert rep.figures == [] and rep.all_supported
