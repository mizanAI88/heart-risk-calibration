"""Discrimination, calibration, threshold and bootstrap metrics on predicted
probabilities. Every function takes arrays and returns plain Python values so
the results serialise to JSON without further conversion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from heart_risk_calibration.errors import ConfigError

CORE_METRICS: tuple[str, ...] = ("auroc", "auprc", "brier", "ece")


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ConfigError("labels and probabilities must be 1D arrays of equal length")
    if y.size == 0:
        raise ConfigError("no rows to score")
    if ((p < 0) | (p > 1)).any() or np.isnan(p).any():
        raise ConfigError("probabilities must lie in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ConfigError("labels must be 0 or 1")
    return y, p


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    if len(np.unique(y)) < 2:
        raise ConfigError("AUROC needs both classes present")
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    if len(np.unique(y)) < 2:
        raise ConfigError("AUPRC needs both classes present")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    return float(brier_score_loss(y, p))


def bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin on [0, 1]; a probability of exactly 1 falls in the last bin."""
    return np.minimum((np.asarray(p, dtype=float) * n_bins).astype(int), n_bins - 1)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    """Per-bin count, mean predicted probability and observed positive fraction."""
    y, p = _check(y, p)
    if n_bins < 2:
        raise ConfigError("n_bins must be >= 2")
    idx = bin_index(p, n_bins)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append({
            "bin": b,
            "lower": round(b / n_bins, 6),
            "upper": round((b + 1) / n_bins, 6),
            "n": n,
            "mean_predicted": float(p[mask].mean()) if n else None,
            "fraction_positive": float(y[mask].mean()) if n else None,
        })
    return rows


def ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Expected calibration error: count-weighted mean |mean_predicted - fraction_positive|."""
    total = 0.0
    n_all = int(np.asarray(y).size)
    for row in reliability_bins(y, p, n_bins):
        if row["n"]:
            total += row["n"] / n_all * abs(row["mean_predicted"] - row["fraction_positive"])
    return float(total)


def core_metrics(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, float]:
    return {"auroc": auroc(y, p), "auprc": auprc(y, p), "brier": brier(y, p), "ece": ece(y, p, n_bins)}


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion-derived rates at one threshold (predict positive when p >= threshold)."""
    y, p = _check(y, p)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "n": int(y.size),
        "n_flagged": int(pred.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "ppv": tp / (tp + fp) if tp + fp else None,
        "npv": tn / (tn + fn) if tn + fn else None,
    }


def threshold_sensitivity_table(y: np.ndarray, p: np.ndarray,
                                grid: tuple[float, ...]) -> list[dict[str, Any]]:
    return [threshold_metrics(y, p, t) for t in grid]


def bootstrap_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """(resamples, n) row indices drawn with replacement; rows are patients."""
    if n < 2 or resamples < 1:
        raise ConfigError("bootstrap needs at least 2 rows and 1 resample")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(resamples, n))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_bins: int, resamples: int, seed: int,
                 alpha: float = 0.05) -> dict[str, Any]:
    """Percentile intervals for the core metrics over row resamples.

    Resamples in which one class is absent are skipped for AUROC and AUPRC and
    the number actually used is reported per metric.
    """
    y, p = _check(y, p)
    draws: dict[str, list[float]] = {m: [] for m in CORE_METRICS}
    for idx in bootstrap_indices(y.size, resamples, seed):
        yb, pb = y[idx], p[idx]
        draws["brier"].append(brier(yb, pb))
        draws["ece"].append(ece(yb, pb, n_bins))
        if len(np.unique(yb)) == 2:
            draws["auroc"].append(auroc(yb, pb))
            draws["auprc"].append(auprc(yb, pb))
    out: dict[str, Any] = {"resamples": int(resamples), "alpha": float(alpha), "unit": "row (patient)"}
    for m, values in draws.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            out[m] = {"lower": None, "upper": None, "n_used": 0}
            continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = {"lower": float(lo), "upper": float(hi), "n_used": int(arr.size)}
    return out
