"""Metric definitions checked against hand-computable cases."""
from __future__ import annotations

import numpy as np
import pytest

from heart_risk_calibration.errors import ConfigError
from heart_risk_calibration.metrics import (auroc, bootstrap_ci, bootstrap_indices, brier, ece,
                                            reliability_bins, threshold_metrics,
                                            threshold_sensitivity_table)


def test_ece_is_zero_when_bins_are_perfectly_calibrated():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.full(8, 0.5)
    assert ece(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_equals_gap_for_a_single_overconfident_bin():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.9)
    assert ece(y, p, n_bins=10) == pytest.approx(0.4)


def test_reliability_bins_count_every_row_once_and_put_one_in_the_last_bin():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.0, 0.25, 1.0, 0.999, 0.5])
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins) == 10
    assert sum(b["n"] for b in bins) == 5
    assert bins[9]["n"] == 2
    assert bins[0]["n"] == 1 and bins[0]["mean_predicted"] == 0.0
    assert bins[3]["n"] == 0 and bins[3]["mean_predicted"] is None


def test_brier_matches_hand_computation_and_rejects_bad_probabilities():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    assert brier(y, p) == pytest.approx(((0.2 ** 2) + (0.3 ** 2)) / 2)
    with pytest.raises(ConfigError, match="probabilities"):
        brier(y, np.array([0.5, 1.2]))


def test_auroc_is_one_for_perfect_separation_and_needs_both_classes():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    with pytest.raises(ConfigError, match="both classes"):
        auroc(np.array([1, 1]), np.array([0.5, 0.6]))


def test_threshold_sensitivity_is_monotone_in_flagged_count_and_sensitivity():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    table = threshold_sensitivity_table(y, p, tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)))
    flagged = [r["n_flagged"] for r in table]
    sens = [r["sensitivity"] for r in table]
    assert flagged == sorted(flagged, reverse=True)
    assert sens == sorted(sens, reverse=True)
    assert table[0]["threshold"] == 0.05 and table[-1]["threshold"] == 0.95


def test_threshold_metrics_report_none_instead_of_dividing_by_zero():
    m = threshold_metrics(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3]), threshold=0.9)
    assert m["n_flagged"] == 0 and m["ppv"] is None and m["sensitivity"] == 0.0
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 3


def test_bootstrap_resamples_rows_with_replacement_at_full_size():
    idx = bootstrap_indices(n=50, resamples=20, seed=1)
    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50
    assert any(len(set(row.tolist())) < 50 for row in idx), "resampling with replacement repeats rows"


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded():
    rng = np.random.default_rng(3)
    p = rng.random(150)
    y = (rng.random(150) < p).astype(int)
    a = bootstrap_ci(y, p, n_bins=10, resamples=50, seed=9)
    b = bootstrap_ci(y, p, n_bins=10, resamples=50, seed=9)
    assert a == b
    assert a["unit"] == "row (patient)"
    point = auroc(y, p)
    assert a["auroc"]["lower"] <= point <= a["auroc"]["upper"]
    assert a["auroc"]["n_used"] == 50
    assert a["brier"]["lower"] <= a["brier"]["upper"]
