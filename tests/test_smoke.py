"""Tests for the pieces that are real today: the EDGAR client and the eval metrics."""
from __future__ import annotations

import pytest

from sightline.eval.metrics import mrr, ndcg_at_k, recall_at_k
from sightline.ingest.edgar import EdgarClient, Filing


def test_user_agent_requires_email():
    with pytest.raises(ValueError):
        EdgarClient(user_agent="no-email-here")


def test_filing_url_construction():
    f = Filing(
        cik=320193, ticker="AAPL", form="10-K", filing_date="2024-11-01",
        accession="0000320193-24-000123", primary_document="aapl-20240928.htm",
    )
    assert f.accession_nodash == "000032019324000123"
    assert f.primary_url.endswith("/000032019324000123/aapl-20240928.htm")
    assert f.primary_url.startswith("https://www.sec.gov/Archives/edgar/data/320193/")


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "z"}
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # found "b" of {b,z}
    assert recall_at_k(retrieved, relevant, k=4) == 0.5


def test_mrr():
    assert mrr(["x", "y", "b"], {"b"}) == pytest.approx(1 / 3)
    assert mrr(["b", "y"], {"b"}) == 1.0
    assert mrr(["x", "y"], {"b"}) == 0.0


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0
