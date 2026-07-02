"""Tests for the deterministic numeric judge — grading must be trustworthy before it's free."""
from __future__ import annotations

from sightline.eval.judge import numeric_match


def test_exact_millions_figure_matches():
    assert numeric_match("$215,938 million.", "Revenue was $215,938 million [p:x#51].") is True


def test_billions_rounding_accepted():
    assert numeric_match("$215,938 million.", "About $215.9 billion.") is True
    assert numeric_match("$215,938 million.", "Approximately $216 billion.") is True


def test_wrong_number_fails():
    assert numeric_match("$215,938 million.", "Revenue was $130,497 million.") is False


def test_largest_number_is_the_essential_fact():
    gold = "Approximately 42,000 employees in 38 countries."
    assert numeric_match(gold, "They had about 42,000 employees.") is True
    assert numeric_match(gold, "They operate in 38 countries.") is False  # 38 alone isn't the fact


def test_prose_gold_defers_to_llm():
    assert numeric_match("TSMC and Samsung.", "TSMC and Samsung.") is None


def test_commas_and_plain_forms_equal():
    assert numeric_match("85,100 people.", "Intel employed 85100 people.") is True
