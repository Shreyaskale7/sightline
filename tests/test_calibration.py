"""Cohen's kappa tests — the calibration metric must be right before we report it."""
from __future__ import annotations

import pytest

from sightline.eval.calibration import KappaReport, cohens_kappa, score_labels


def test_perfect_agreement():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_chance_level_agreement_is_zero():
    # Rater B says 1 half the time regardless of A: agreement equals chance -> kappa ~ 0
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_total_disagreement_is_negative():
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_known_textbook_value():
    # 2x2 example: agree on 45 (1,1) + 25 (0,0) of 100; marginals 0.6/0.55.
    a = [1] * 60 + [0] * 40
    b = [1] * 45 + [0] * 15 + [1] * 10 + [0] * 30
    # p_o = 0.75, p_e = 0.6*0.55 + 0.4*0.45 = 0.51 -> kappa = 0.24/0.49
    assert cohens_kappa(a, b) == pytest.approx(0.24 / 0.49, abs=1e-9)


def test_constant_identical_raters():
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_score_labels_reads_csv_and_skips_unlabeled(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "id,question,gold,answer,judge,human\n"
        "a,q,g,ans,1,1\n"
        "b,q,g,ans,0,0\n"
        "c,q,g,ans,1,\n"      # unlabeled -> skipped
        "d,q,g,ans,1,0\n",
        encoding="utf-8",
    )
    r = score_labels(p)
    assert r.n == 3
    assert r.raw_agreement == pytest.approx(2 / 3)


def test_report_reading_bands():
    assert "strong" in KappaReport(10, 0.9, 0.85).reading
    assert "NOT" in KappaReport(10, 0.5, 0.2).reading
