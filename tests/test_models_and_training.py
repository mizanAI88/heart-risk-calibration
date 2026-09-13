"""Pipelines, folds, and the leakage guarantees of per-fold fitting."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from heart_risk_calibration.config import DataConfig, EvalConfig, ModelConfig
from heart_risk_calibration.errors import ConfigError
from heart_risk_calibration.models import (CLASSIFIER_STEP, IMPUTER_STEP, MLP, MODEL_NAMES,
                                           SCALER_STEP, build_pipeline, grid_size, param_grid)
from heart_risk_calibration.paths import PACKAGE_DIR
from heart_risk_calibration.schema import HeartTable
from heart_risk_calibration.splits import Fold, inner_splitter, outer_folds
from heart_risk_calibration.training import (THRESHOLD_SOURCE, as_matrix, choose_threshold,
                                             fit_fold)
from tests.conftest import small_table


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_is_an_imputer_scaler_classifier_pipeline(name: str, model_cfg: ModelConfig):
    pipe = build_pipeline(name, model_cfg, seed=42)
    assert isinstance(pipe, Pipeline)
    assert [s for s, _ in pipe.steps] == [IMPUTER_STEP, SCALER_STEP, CLASSIFIER_STEP]
    assert all(k.startswith(f"{CLASSIFIER_STEP}__") for k in param_grid(name, model_cfg))
    assert grid_size(name, model_cfg) >= 1


def test_unknown_model_name_is_a_config_error(model_cfg: ModelConfig):
    with pytest.raises(ConfigError, match="unknown model"):
        build_pipeline("random_forest", model_cfg, seed=1)


def test_mlp_is_sklearn_and_package_never_imports_torch(model_cfg: ModelConfig):
    pipe = build_pipeline(MLP, model_cfg, seed=1)
    assert isinstance(pipe.named_steps[CLASSIFIER_STEP], MLPClassifier)
    for path in PACKAGE_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import torch" not in source and "from torch" not in source, path.name


def test_pipeline_fits_and_predicts_with_nan_inputs(model_cfg: ModelConfig, small: HeartTable):
    x = as_matrix(small.features)
    assert np.isnan(x).any(), "the small table must contain planted missing cells"
    pipe = build_pipeline("logistic_regression", model_cfg, seed=1).fit(x, small.label)
    proba = pipe.predict_proba(x)
    assert proba.shape == (small.n_rows, 2)
    assert x.ndim == 2 and x.shape[1] == 13, "tabular inputs stay 2D; no reshaping"


def test_outer_folds_are_disjoint_stratified_and_cover_every_row(small: HeartTable):
    folds = outer_folds(small.label, k=5, seed=42)
    assert len(folds) == 5
    all_test = np.concatenate([f.test_idx for f in folds])
    assert sorted(all_test.tolist()) == list(range(small.n_rows))
    overall = small.label.mean()
    for f in folds:
        assert f.n_train + f.n_test == small.n_rows
        assert abs(small.label[f.test_idx].mean() - overall) < 0.2
    with pytest.raises(ConfigError, match="at least 5 rows"):
        outer_folds(np.array([0] * 20 + [1] * 3), k=5, seed=0)
    with pytest.raises(ConfigError, match="overlap"):
        Fold(fold=1, train_idx=np.array([0, 1, 2]), test_idx=np.array([2, 3]))


def test_inner_splitter_only_sees_outer_train_rows(small: HeartTable):
    fold = outer_folds(small.label, k=5, seed=42)[0]
    inner = inner_splitter(3, 42)
    x_train = as_matrix(small.features.iloc[fold.train_idx])
    for fit_idx, val_idx in inner.split(x_train, small.label[fold.train_idx]):
        rows = fold.train_idx[np.concatenate([fit_idx, val_idx])]
        assert not np.intersect1d(rows, fold.test_idx).size


def _perturbed_copy(table: HeartTable, fold: Fold) -> HeartTable:
    """Same table with the outer-test rows' features scaled and labels flipped."""
    features = table.features.copy()
    features.iloc[fold.test_idx] = features.iloc[fold.test_idx] * 10.0 + 1000.0
    label = table.label.copy()
    label[fold.test_idx] = 1 - label[fold.test_idx]
    return HeartTable(ids=table.ids, features=features, num=table.num, label=label,
                      missing_cells=table.missing_cells)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_imputer_and_scaler_are_fitted_on_training_rows_only(name: str, data_cfg: DataConfig,
                                                             smoke_model_cfg: ModelConfig,
                                                             smoke_eval_cfg: EvalConfig):
    table = small_table(data_cfg, seed=11, n_rows=90)
    fold = outer_folds(table.label, smoke_eval_cfg.outer_folds, smoke_eval_cfg.seed)[1]
    a = fit_fold(name, table.features, table.label, fold, smoke_model_cfg, smoke_eval_cfg)
    b = fit_fold(name, _perturbed_copy(table, fold).features, table.label, fold,
                 smoke_model_cfg, smoke_eval_cfg)
    np.testing.assert_array_equal(a.pipeline.named_steps[IMPUTER_STEP].statistics_,
                                  b.pipeline.named_steps[IMPUTER_STEP].statistics_)
    np.testing.assert_array_equal(a.pipeline.named_steps[SCALER_STEP].mean_,
                                  b.pipeline.named_steps[SCALER_STEP].mean_)
    np.testing.assert_array_equal(a.pipeline.named_steps[SCALER_STEP].scale_,
                                  b.pipeline.named_steps[SCALER_STEP].scale_)


def test_threshold_never_uses_test_labels(data_cfg: DataConfig, smoke_model_cfg: ModelConfig,
                                          smoke_eval_cfg: EvalConfig):
    table = small_table(data_cfg, seed=12, n_rows=90)
    fold = outer_folds(table.label, smoke_eval_cfg.outer_folds, smoke_eval_cfg.seed)[2]
    perturbed = _perturbed_copy(table, fold)
    a = fit_fold("logistic_regression", table.features, table.label, fold, smoke_model_cfg, smoke_eval_cfg)
    b = fit_fold("logistic_regression", perturbed.features, perturbed.label, fold, smoke_model_cfg,
                 smoke_eval_cfg)
    assert a.threshold == b.threshold
    assert a.threshold_source == THRESHOLD_SOURCE
    assert a.n_validation_predictions == fold.n_train
    train_pred_a = a.predict_proba(table.features.iloc[fold.train_idx])
    train_pred_b = b.predict_proba(table.features.iloc[fold.train_idx])
    np.testing.assert_allclose(train_pred_a, train_pred_b)


def test_choose_threshold_maximises_youden_and_breaks_ties_low():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    grid = (0.25, 0.5, 0.65, 0.95)
    assert choose_threshold(y, p, "youden", grid) == 0.5
    assert choose_threshold(y, p, "f1", grid) == 0.5
    with pytest.raises(ConfigError, match="unknown threshold rule"):
        choose_threshold(y, p, "accuracy", grid)


def test_fit_fold_is_deterministic_under_the_seed(data_cfg: DataConfig, smoke_model_cfg: ModelConfig,
                                                  smoke_eval_cfg: EvalConfig):
    table = small_table(data_cfg, seed=13, n_rows=90)
    fold = outer_folds(table.label, 5, smoke_eval_cfg.seed)[0]
    a = fit_fold(MLP, table.features, table.label, fold, smoke_model_cfg, smoke_eval_cfg)
    b = fit_fold(MLP, table.features, table.label, fold, smoke_model_cfg, smoke_eval_cfg)
    np.testing.assert_allclose(a.predict_proba(table.features.iloc[fold.test_idx]),
                               b.predict_proba(table.features.iloc[fold.test_idx]))
    assert a.threshold == b.threshold


def test_fit_fold_rejects_single_class_training_rows(data_cfg: DataConfig, smoke_model_cfg: ModelConfig,
                                                     smoke_eval_cfg: EvalConfig):
    table = small_table(data_cfg, seed=14, n_rows=60)
    label = np.zeros(table.n_rows, dtype=int)
    fold = Fold(fold=1, train_idx=np.arange(40), test_idx=np.arange(40, 60))
    with pytest.raises(ConfigError, match="single class"):
        fit_fold("logistic_regression", table.features, label, fold, smoke_model_cfg, smoke_eval_cfg)
