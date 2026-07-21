"""Tests for the income-statement preference detectors — both gates must be precise."""
from __future__ import annotations

from sightline.retrieval.statement_boost import (
    is_financial_metric_question,
    looks_like_income_statement,
)


def test_financial_questions_detected():
    for q in [
        "What was NVIDIA's total revenue in its most recent 10-K?",
        "How much did AMD spend on research and development?",
        "What was Micron's net income?",
        "What were Qualcomm's net sales?",
        "What was the gross margin?",
    ]:
        assert is_financial_metric_question(q), q


def test_non_financial_questions_not_flagged():
    for q in [
        "Approximately how many employees did Micron report?",
        "Which foundries does NVIDIA rely on?",
        "What are Qualcomm's business segments?",
        "Where is Micron's headquarters?",
    ]:
        assert not is_financial_metric_question(q), q


def test_income_statement_fingerprint_matches():
    page = ("Revenue $ 215,938 Cost of revenue 62,475 Gross profit 153,463 "
            "Research and development 18,497 Operating income 130,387 Net income $ 120,067")
    assert looks_like_income_statement(page)


def test_prose_and_other_pages_do_not_match():
    # MD&A prose that mentions revenue but isn't the statement
    assert not looks_like_income_statement(
        "Total revenue for 2025 increased 49% as compared to 2024 primarily due to demand.")
    # human-capital page
    assert not looks_like_income_statement(
        "As of the end of fiscal 2026 we had approximately 42,000 employees in 38 countries.")
    # a balance sheet (has net income mention in prose but no cost-of line)
    assert not looks_like_income_statement(
        "Total assets 1,000 Total liabilities 400 Cash and cash equivalents 200")
