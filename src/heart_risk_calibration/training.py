"""Per-fold fitting: inner grid search on the outer training rows, refit of
the best pipeline on all outer training rows, and a decision threshold
chosen on inner out-of-fold validation predictions. Outer test rows are never
passed to anything in this module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from heart_risk_calibration.config import EvalConfig, ModelConfig
from heart_risk_calibration.errors import ConfigError
from heart_risk_calibration.models import build_pipeline, grid_size, param_grid
from heart_risk_calibration.splits import Fold, inner_splitter

THRESHOLD_SOURCE = "inner_out_of_fold_validation_predictions"


@dataclass(frozen=True)
class FoldFit:
    """Fitted pipeline for one outer fold plus how its threshold was chosen."""

    model_name: str
    fold: int
    pipeline: Pipeline
    best_params: Mapping[str, Any]
    inner_best_score: float | None
    threshold: float
    threshold_source: str
    n_train: int
    n_validation_predictions: int

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return predict_positive_probability(self.pipeline, features)


def as_matrix(features: pd.DataFrame) -> np.ndarray:
    """Float matrix (n_rows, 13). NaN cells are left for the pipeline imputer."""
    matrix = np.asarray(features.to_numpy(dtype=float))
    if matrix.ndim != 2:
        raise ConfigError("features must be a 2D table of rows by attributes")
    return matrix


def predict_positive_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(as_matrix(features))
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ConfigError("fitted pipeline has no positive class; training rows held one class only")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def _youden(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    pos = y == 1
    sens = pred[pos].mean() if pos.any() else 0.0
    spec = (~pred[~pos]).mean() if (~pos).any() else 0.0
    return float(sens + spec - 1.0)


def _f1(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


_RULES = {"youden": _youden, "f1": _f1}


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray, rule: str,
                     grid: tuple[float, ...]) -> float:
    """Threshold from the grid maximising `rule` on validation predictions only.

    Ties resolve to the lowest threshold. The caller is responsible for passing
    validation predictions, never test predictions; `fit_fold` does so.
    """
    if rule not in _RULES:
        raise ConfigError(f"unknown threshold rule {rule!r}; known: {tuple(_RULES)}")
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val, dtype=float)
    if y_val.shape != p_val.shape:
        raise ConfigError("validation labels and predictions differ in length")
    scores = [(_RULES[rule](y_val, p_val, t), -t) for t in grid]
    best = max(range(len(grid)), key=lambda i: scores[i])
    return float(grid[best])


def fit_rows(model_name: str, features: pd.DataFrame, label: np.ndarray, train_idx: np.ndarray,
             model_cfg: ModelConfig, eval_cfg: EvalConfig, fold_number: int = 0) -> FoldFit:
    """Inner grid search + refit on `train_idx` only; threshold from inner OOF predictions.

    Only the rows named in `train_idx` are ever passed to the imputer, scaler,
    classifier or threshold rule.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    x_train = as_matrix(features.iloc[train_idx])
    y_train = np.asarray(label)[train_idx]
    if len(np.unique(y_train)) < 2:
        raise ConfigError(f"fold {fold_number}: training rows hold a single class")
    inner = inner_splitter(eval_cfg.inner_folds, eval_cfg.seed)
    pipeline = build_pipeline(model_name, model_cfg, eval_cfg.seed)
    grid = param_grid(model_name, model_cfg)
    # A few hundred rows: native thread pools (OpenMP, BLAS) oversubscribe and
    # make each fit slower by an order of magnitude, so fits run single-threaded.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        if grid_size(model_name, model_cfg) == 1:
            # One candidate: the search would only refit the same pipeline.
            best_params = {k: v[0] for k, v in grid.items()}
            best: Pipeline = clone(pipeline).set_params(**best_params)
            best.fit(x_train, y_train)
            inner_best_score = None
        else:
            search = GridSearchCV(pipeline, grid, scoring=eval_cfg.inner_scoring, cv=inner,
                                  refit=True, n_jobs=None)
            search.fit(x_train, y_train)
            best = search.best_estimator_
            best_params = dict(search.best_params_)
            inner_best_score = float(search.best_score_)
        p_val = cross_val_predict(clone(best), x_train, y_train, cv=inner, method="predict_proba")
    p_val = np.asarray(p_val)[:, list(best.classes_).index(1)]
    threshold = choose_threshold(y_train, p_val, eval_cfg.threshold_rule, eval_cfg.threshold_grid)
    return FoldFit(
        model_name=model_name,
        fold=fold_number,
        pipeline=best,
        best_params={k.replace("classifier__", ""): v for k, v in best_params.items()},
        inner_best_score=inner_best_score,
        threshold=threshold,
        threshold_source=THRESHOLD_SOURCE,
        n_train=int(y_train.size),
        n_validation_predictions=int(p_val.size),
    )


def fit_fold(model_name: str, features: pd.DataFrame, label: np.ndarray, fold: Fold,
             model_cfg: ModelConfig, eval_cfg: EvalConfig) -> FoldFit:
    """`fit_rows` on `fold.train_idx`; `fold.test_idx` is never touched here."""
    return fit_rows(model_name, features, label, fold.train_idx, model_cfg, eval_cfg, fold.fold)
