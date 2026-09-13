"""Plain-text tables rendered from a metrics document. The CLI prints these;
nothing here prints."""
from __future__ import annotations

from typing import Any

SYNTHETIC_BANNER = ("Synthetic fixture run (authored demonstration data, seed {seed}); "
                    "not a benchmark result.")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_lines(metrics: dict[str, Any]) -> list[str]:
    """One row per model: pooled AUROC, AUPRC, Brier, ECE with bootstrap intervals."""
    lines: list[str] = []
    if metrics.get("synthetic"):
        lines.append(SYNTHETIC_BANNER.format(seed=metrics.get("seed")))
    proto = metrics["protocol"]
    lines.append(
        f"source={metrics['source_id']} mode={metrics['mode']} rows={metrics['n_rows']} "
        f"positive={metrics['n_positive']} outer_folds={proto['outer_folds']} "
        f"inner_folds={proto['inner_folds']} bootstrap={proto['bootstrap_resamples']}"
    )
    lines.append("rows per outer test fold: " + ", ".join(
        f"fold {f['fold']}: {f['n_test']}" for f in metrics["folds"]))
    header = f"{'model':<26}{'n':>5}{'AUROC':>8}{'CI':>17}{'AUPRC':>8}{'Brier':>8}{'CI':>17}{'ECE':>8}{'CI':>17}"
    lines.append(header)
    for name, body in metrics["models"].items():
        p = body["pooled"]
        ci = p["ci"]

        def interval(m: str) -> str:
            return f"[{_fmt(ci[m]['lower'])}, {_fmt(ci[m]['upper'])}]"

        lines.append(
            f"{name:<26}{p['n']:>5}{_fmt(p['auroc']):>8}{interval('auroc'):>17}"
            f"{_fmt(p['auprc']):>8}{_fmt(p['brier']):>8}{interval('brier'):>17}"
            f"{_fmt(p['ece']):>8}{interval('ece'):>17}"
        )
    lines.append("Intervals: percentile bootstrap over rows (one row is one patient).")
    return lines


def calibration_lines(metrics: dict[str, Any], model: str | None = None) -> list[str]:
    """Reliability table, 10 bins by default, per model."""
    lines: list[str] = []
    if metrics.get("synthetic"):
        lines.append(SYNTHETIC_BANNER.format(seed=metrics.get("seed")))
    names = [model] if model else list(metrics["models"])
    for name in names:
        body = metrics["models"][name]
        lines.append(f"model: {name}  ECE={_fmt(body['pooled']['ece'])}  n={body['pooled']['n']}")
        lines.append(f"{'bin':>4}{'lower':>7}{'upper':>7}{'n':>6}{'mean_pred':>11}{'frac_pos':>10}")
        for b in body["reliability_bins"]:
            lines.append(
                f"{b['bin']:>4}{_fmt(b['lower'], 2):>7}{_fmt(b['upper'], 2):>7}{b['n']:>6}"
                f"{_fmt(b['mean_predicted']):>11}{_fmt(b['fraction_positive']):>10}"
            )
    return lines


def threshold_lines(metrics: dict[str, Any], model: str | None = None) -> list[str]:
    lines: list[str] = []
    names = [model] if model else list(metrics["models"])
    for name in names:
        body = metrics["models"][name]
        chosen = ", ".join(f"fold {r['fold']}: {r['threshold']:.2f}" for r in body["per_fold"])
        lines.append(f"model: {name}  thresholds chosen on validation ({chosen})")
        lines.append(f"{'threshold':>10}{'n_flagged':>10}{'sens':>7}{'spec':>7}{'ppv':>7}{'npv':>7}")
        for r in body["threshold_sensitivity"]:
            lines.append(
                f"{r['threshold']:>10.2f}{r['n_flagged']:>10}{_fmt(r['sensitivity']):>7}"
                f"{_fmt(r['specificity']):>7}{_fmt(r['ppv']):>7}{_fmt(r['npv']):>7}"
            )
    return lines
