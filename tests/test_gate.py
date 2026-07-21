"""Eval-gate tests — the gate that blocks quality regressions must itself be correct.

Includes the "regression caught" case: this is the CI-for-model-behavior story in a test.
"""
from __future__ import annotations

import json

from sightline.eval.gate import BASELINE, check_regression, load_metrics

_BASE = {"recall@5": 0.603, "ndcg@10": 0.485, "mrr": 0.441}


def test_committed_baseline_is_valid_and_matches_champion():
    b = load_metrics(BASELINE)
    assert b["retriever"] == "routed" and 0.5 < b["recall@5"] <= 1.0


def test_no_regression_when_equal():
    assert check_regression(dict(_BASE), _BASE) == []


def test_small_dip_within_tolerance_passes():
    current = {"recall@5": 0.590, "ndcg@10": 0.485, "mrr": 0.441}  # −0.013, under 0.02
    assert check_regression(current, _BASE) == []


def test_real_regression_is_caught():
    current = {"recall@5": 0.55, "ndcg@10": 0.485, "mrr": 0.441}  # −0.053, over 0.02
    regs = check_regression(current, _BASE)
    assert [r.metric for r in regs] == ["recall@5"]
    assert regs[0].drop > 0.05


def test_improvement_never_fails():
    current = {"recall@5": 0.70, "ndcg@10": 0.60, "mrr": 0.55}
    assert check_regression(current, _BASE) == []


def test_multiple_regressions_all_reported():
    current = {"recall@5": 0.40, "ndcg@10": 0.30, "mrr": 0.20}
    assert {r.metric for r in check_regression(current, _BASE)} == {"recall@5", "ndcg@10", "mrr"}


def test_round_trips_through_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_BASE))
    assert load_metrics(p) == _BASE
