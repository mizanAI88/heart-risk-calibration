"""Final-model checkpoints written by `train`: every configured model fitted
on all supplied rows (inner grid search over the full table), with the
threshold chosen on out-of-fold validation predictions of that table. A
checkpoint is a pickle of plain sklearn objects plus provenance fields.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from heart_risk_calibration import __version__
from heart_risk_calibration.config import EvalConfig, ModelConfig
from heart_risk_calibration.errors import ConfigError
from heart_risk_calibration.schema import HeartTable
from heart_risk_calibration.training import fit_rows, predict_positive_probability

CHECKPOINT_SUFFIX = ".pkl"


@dataclass(frozen=True)
class Checkpoint:
    model_name: str
    pipeline: Pipeline
    threshold: float
    threshold_source: str
    best_params: Mapping[str, Any]
    n_train: int
    source_id: str
    mode: str
    seed: int
    feature_columns: tuple[str, ...]
    package_version: str = __version__

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if tuple(features.columns) != self.feature_columns:
            raise ConfigError(
                f"checkpoint expects columns {self.feature_columns}; got {tuple(features.columns)}"
            )
        return predict_positive_probability(self.pipeline, features)


def fit_final(model_name: str, table: HeartTable, model_cfg: ModelConfig, eval_cfg: EvalConfig,
              *, source_id: str, mode: str) -> Checkpoint:
    """Fit on every row of `table`. No row is held out here by design; the
    evaluation of such a model is the nested CV run, not this fit."""
    fit = fit_rows(model_name, table.features, table.label, np.arange(table.n_rows),
                   model_cfg, eval_cfg, fold_number=0)
    return Checkpoint(
        model_name=model_name,
        pipeline=fit.pipeline,
        threshold=fit.threshold,
        threshold_source=fit.threshold_source,
        best_params=dict(fit.best_params),
        n_train=fit.n_train,
        source_id=source_id,
        mode=mode,
        seed=eval_cfg.seed,
        feature_columns=tuple(table.features.columns),
    )


def save_checkpoint(ckpt: Checkpoint, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(ckpt, handle)
    return path


def load_checkpoint(path: Path) -> Checkpoint:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"checkpoint {path} does not exist; run `train` first")
    with path.open("rb") as handle:
        obj = pickle.load(handle)  # noqa: S301 (local file written by this package)
    if not isinstance(obj, Checkpoint):
        raise ConfigError(f"{path} is not a heart-risk-calibration checkpoint")
    return obj
