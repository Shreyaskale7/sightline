"""Query-filter parser tests — a wrong filter silently hides the right page, so test hard."""
from __future__ import annotations

from sightline.retrieval.filters import parse_query_filters


def test_company_and_form_detected():
    f = parse_query_filters("What was NVIDIA's total revenue in its most recent 10-K?")
    assert f.tickers == ["NVDA"] and f.form == "10-K"


def test_quarterly_phrasing_maps_to_10q():
    f = parse_query_filters("How did AMD's revenue change across its last three 10-Q filings?")
    assert f.tickers == ["AMD"] and f.form == "10-Q"
    f2 = parse_query_filters("What did Micron's latest quarterly report say about demand?")
    assert f2.tickers == ["MU"] and f2.form == "10-Q"


def test_cross_company_keeps_both_tickers():
    f = parse_query_filters("Which spent more on R&D in its most recent 10-K, NVIDIA or Intel?")
    assert set(f.tickers) == {"NVDA", "INTC"} and f.form == "10-K"


def test_out_of_corpus_company_yields_no_ticker_filter():
    f = parse_query_filters("What was Broadcom's total revenue in its most recent 10-K?")
    assert f.tickers == [] and f.form == "10-K"


def test_both_forms_mentioned_disables_form_filter():
    f = parse_query_filters("Compare Intel's 10-K risk factors with its latest 10-Q update.")
    assert f.form is None


def test_no_mentions_yields_empty_filter():
    f = parse_query_filters("What are the biggest supply-chain risks disclosed?")
    assert f.empty


def test_word_boundaries_no_false_ticker_hits():
    # "commute" contains "mu"; "amdahl" contains "amd" — word boundaries must prevent both.
    f = parse_query_filters("Does the commute policy at Amdahl matter?")
    assert f.tickers == []
