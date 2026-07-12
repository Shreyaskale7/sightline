"""Cost + money-shot tests — the money-shot number goes in the README, so it must be right."""
from __future__ import annotations

import pytest

from sightline.cost import estimate_cost, money_shot, text_tokens


def test_estimate_cost_known_model():
    # 1M input tokens of gpt-4o at $2.50/M = $2.50
    assert estimate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
    assert estimate_cost("gpt-4o", 0, 1_000_000) == pytest.approx(10.00)


def test_estimate_cost_unknown_model_is_free():
    assert estimate_cost("nvidia/nemotron-3-super-120b-a12b:free", 999_999, 999_999) == 0.0


def test_text_tokens_rough():
    assert text_tokens(400) == 100  # 4 chars/token


def test_money_shot_reduction_is_large():
    ms = money_shot(model="gpt-4o", corpus_pages=1329, k=5)
    assert ms.pages_read_naive == 1329 and ms.pages_read_sightline == 5
    assert ms.naive_usd > ms.sightline_usd
    assert ms.reduction_pct > 99.0  # reading 5 of 1329 pages saves ~99.6%


def test_money_shot_scales_with_k():
    a = money_shot(k=5)
    b = money_shot(k=50)
    assert b.sightline_usd > a.sightline_usd  # more pages read -> more cost
    assert b.reduction_pct < a.reduction_pct


def test_money_shot_all_reference_models():
    from sightline.cost import PRICING

    for model in PRICING:
        ms = money_shot(model=model)
        assert ms.reduction_pct > 95.0
