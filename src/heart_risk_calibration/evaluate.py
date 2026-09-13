"""Nested cross-validation driver and the metrics document it produces.

For each model and each outer fold: `training.fit_fold` selects and refits on
the outer training rows, then the outer test rows are scored exactly once.
Pooled out-of-fold predictions (one per patient) give the reliability bins,
the threshold sensitivity table and the bootstrap intervals.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heart_risk_calibration import PROJECT_ID
from heart_risk_calibration.config import EvalConfig, ModelConfig
from heart_risk_calibration.errors import ValidationError
from heart_risk_calibration.metrics import (CORE_METRICS, bootstrap_ci, core_metrics,
                                            reliability_bins, threshold_metrics,
                                            threshold_sensitivity_table)
from heart_risk_calibration.schema import HeartTable
from heart_risk_calibration.splits import Fold, outer_folds, splits_document
from heart_risk_calibration.training import fit_fold

METRICS_SCHEMA_VERSION = 1
PREDICTION_COLUMNS: tuple[str, ...] = ("patient_id", "fold", "model", "y_true", "p_hat",
                                       "threshold", "y_pred")


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    splits: dict[str, Any]


def _fold_records(folds: tuple[Fold, ...], label: np.ndarray) -> list[dict[str, int]]:
    return [{"fold": f.fold, "n_train": f.n_train, "n_test": f.n_test,
             "n_test_positive": int(label[f.test_idx].sum())} for f in folds]


def _evaluate_model(name: str, table: HeartTable, folds: tuple[Fold, ...],
                    model_cfg: ModelConfig, eval_cfg: EvalConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    per_fold: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for fold in folds:
        fit = fit_fold(name, table.features, table.label, fold, model_cfg, eval_cfg)
        y_test = table.label[fold.test_idx]
        p_test = fit.predict_proba(table.features.iloc[fold.test_idx])
        at_t = threshold_metrics(y_test, p_test, fit.threshold)
        per_fold.append({
            "fold": fold.fold,
            "n_train": fit.n_train,
            "n_test": int(y_test.size),
            "n_test_positive": int(y_test.sum()),
            "best_params": dict(fit.best_params),
            "inner_best_score": fit.inner_best_score,
            "threshold": fit.threshold,
            "threshold_source": fit.threshold_source,
            "n_validation_predictions": fit.n_validation_predictions,
            **core_metrics(y_test, p_test, eval_cfg.n_bins),
            "sensitivity_at_threshold": at_t["sensitivity"],
            "specificity_at_threshold": at_t["specificity"],
        })
        frames.append(pd.DataFrame({
            "patient_id": [table.ids[i] for i in fold.test_idx.tolist()],
            "fold": fold.fold,
            "model": name,
            "y_true": y_test,
            "p_hat": p_test,
            "threshold": fit.threshold,
            "y_pred": (p_test >= fit.threshold).astype(int),
        }))
    pooled_frame = pd.concat(frames, ignore_index=True)
    y_all = pooled_frame["y_true"].to_numpy()
    p_all = pooled_frame["p_hat"].to_numpy()
    y_pred_all = pooled_frame["y_pred"].to_numpy()
    pooled = core_metrics(y_all, p_all, eval_cfg.n_bins)
    pooled["n"] = int(y_all.size)
    pooled["ci"] = bootstrap_ci(y_all, p_all, eval_cfg.n_bins, eval_cfg.bootstrap_resamples,
                                eval_cfg.seed)
    tp = int(((y_pred_all == 1) & (y_all == 1)).sum())
    fn = int(((y_pred_all == 0) & (y_all == 1)).sum())
    tn = int(((y_pred_all == 0) & (y_all == 0)).sum())
    fp = int(((y_pred_all == 1) & (y_all == 0)).sum())
    pooled["at_fold_thresholds"] = {
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "n_flagged": int(y_pred_all.sum()),
    }
    doc = {
        "pooled": pooled,
        "per_fold": per_fold,
        "reliability_bins": reliability_bins(y_all, p_all, eval_cfg.n_bins),
        "threshold_sensitivity": threshold_sensitivity_table(y_all, p_all, eval_cfg.threshold_grid),
    }
    return doc, pooled_frame


def run_nested_cv(table: HeartTable, model_cfg: ModelConfig, eval_cfg: EvalConfig, *,
                  source_id: str, mode: str, load_report: dict[str, Any] | None = None,
                  model_names: tuple[str, ...] | None = None) -> EvaluationResult:
    names = model_names or eval_cfg.models
    folds = outer_folds(table.label, eval_cfg.outer_folds, eval_cfg.seed)
    models: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for name in names:
        doc, frame = _evaluate_model(name, table, folds, model_cfg, eval_cfg)
        models[name] = doc
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)[list(PREDICTION_COLUMNS)]
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "mode": mode,
        "source_id": source_id,
        "synthetic": source_id.startswith("synthetic"),
        "seed": eval_cfg.seed,
        "protocol": eval_cfg.as_dict(),
        "n_rows": table.n_rows,
        "n_positive": table.n_positive,
        "n_negative": table.n_negative,
        "missing_cells": table.missing_cells,
        "load_report": load_report or {},
        "folds": _fold_records(folds, table.label),
        "models": models,
    }
    splits = splits_document(folds, table.ids, table.label, eval_cfg.outer_folds,
                             eval_cfg.inner_folds, eval_cfg.seed)
    return EvaluationResult(metrics=metrics, predictions=predictions, splits=splits)


def _finite_unit(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must lie in [0, 1], got {value!r}")


def validate_metrics(doc: dict[str, Any]) -> None:
    """Raise ValidationError unless `doc` has the metrics.json shape and its counts agree."""
    for key in ("schema_version", "project_id", "mode", "source_id", "synthetic", "n_rows",
                "folds", "models", "protocol"):
        if key not in doc:
            raise ValidationError(f"metrics.json is missing key {key!r}")
    n_rows = int(doc["n_rows"])
    folds = doc["folds"]
    if not folds or any(int(f.get("n_test", 0)) <= 0 for f in folds):
        raise ValidationError("every fold must record a positive n_test")
    if sum(int(f["n_test"]) for f in folds) != n_rows:
        raise ValidationError("fold n_test values do not sum to n_rows")
    if any(int(f["n_train"]) + int(f["n_test"]) != n_rows for f in folds):
        raise ValidationError("n_train + n_test must equal n_rows in every fold")
    if not doc["models"]:
        raise ValidationError("metrics.json holds no models")
    for name, body in doc["models"].items():
        pooled = body.get("pooled", {})
        if int(pooled.get("n", -1)) != n_rows:
            raise ValidationError(f"{name}: pooled n must equal n_rows")
        for m in CORE_METRICS:
            _finite_unit(pooled.get(m), f"{name}.pooled.{m}")
        if len(body.get("per_fold", [])) != len(folds):
            raise ValidationError(f"{name}: per_fold count differs from folds")
        for rec in body["per_fold"]:
            if int(rec.get("n_test", 0)) <= 0:
                raise ValidationError(f"{name}: per_fold record without n_test")
            for m in CORE_METRICS:
                _finite_unit(rec.get(m), f"{name}.per_fold[{rec.get('fold')}].{m}")
        bins = body.get("reliability_bins", [])
        if len(bins) != int(doc["protocol"]["n_bins"]):
            raise ValidationError(f"{name}: expected {doc['protocol']['n_bins']} reliability bins")
        if sum(int(b["n"]) for b in bins) != n_rows:
            raise ValidationError(f"{name}: reliability bin counts do not sum to n_rows")
        if not body.get("threshold_sensitivity"):
            raise ValidationError(f"{name}: threshold_sensitivity table is empty")


def write_result(result: EvaluationResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / "metrics.json",
        "predictions": out_dir / "predictions.csv",
        "splits": out_dir / "splits.json",
    }
    paths["metrics"].write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    result.predictions.to_csv(paths["predictions"], index=False)
    paths["splits"].write_text(json.dumps(result.splits, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def read_metrics(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"metrics file {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"metrics file {path} is not valid JSON: {exc}") from exc
