"""Prefer the consolidated financial statement page for figure questions.

Diagnosed weakness: dense retrieval (and even the cross-encoder) under-rank the terse
income-statement TABLE relative to prose pages that also mention a figure (the MD&A "results of
operations" page). For a question like "what was revenue / R&D / net income", the authoritative
source is the consolidated statement — a real analyst goes there — so this encodes that prior.

It is applied ONLY when (a) the question asks for a financial line item, and (b) a candidate page
carries the income-statement fingerprint. Both gates matter: without (a) it would wrongly demote
qualitative pages (employees, risk factors); without (b) it does nothing. It re-orders an existing
candidate pool — it cannot conjure a page the retriever never surfaced.

Kept as a separate, measured config (`routed_boosted`) so its effect on every slice is on the
record, not assumed.
"""
from __future__ import annotations

import re

# The question is about a line item you'd read off the income statement.
_FINANCIAL_Q = re.compile(
    r"\b(revenue|revenues|net sales|sales|research and development|r&d|net income|"
    r"gross profit|gross margin|operating income|operating loss|earnings per share|\beps\b|"
    r"cost of (revenue|sales|goods))\b",
    re.IGNORECASE,
)

# The income-statement fingerprint: a revenue line + a cost line + net income, co-occurring.
# Strong enough to catch statements across formats (Revenue / Net revenue / Net sales), specific
# enough that MD&A prose, risk factors, and human-capital pages don't match.
_REV = re.compile(r"\b(net revenue|net sales|total revenues?|revenues?)\b", re.IGNORECASE)
_COST = re.compile(r"\bcost of (revenue|sales|goods|products)\b", re.IGNORECASE)
_NI = re.compile(r"\bnet income\b", re.IGNORECASE)
_SUPPORT = re.compile(r"\b(gross profit|gross margin|operating income|operating loss|"
                      r"research and development)\b", re.IGNORECASE)


def is_financial_metric_question(question: str) -> bool:
    return bool(_FINANCIAL_Q.search(question))


def looks_like_income_statement(page_text: str) -> bool:
    """True if the page reads like a consolidated income statement, not prose that mentions a
    number. Requires a revenue line AND a cost-of line AND net income AND one more statement row."""
    t = page_text
    return bool(_REV.search(t) and _COST.search(t) and _NI.search(t) and _SUPPORT.search(t))
