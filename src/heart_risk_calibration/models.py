"""Three classifiers, each inside a sklearn Pipeline of imputer -> scaler ->
classifier so that imputation and scaling statistics are fitted only on the
rows passed to `fit` (the training rows of a fold).

The MLP is scikit-learn's `MLPClassifier` rather than a torch module: the
inputs are 13 numeric columns and a few hundred rows, so a one-hidden-layer
network fitted by L-BFGS on CPU is adequate, it composes with `Pipeline` and
`GridSearchCV` without adapter code, it is deterministic given `random_state`,
and it keeps the dependency set to numpy, pandas and scikit-learn. No 2D
reshaping of tabular inputs is performed anywhere.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_risk_calibration.config import ModelConfig, ModelSpec
from heart_risk_calibration.errors import ConfigError

LOGISTIC_REGRESSION = "logistic_regression"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
MLP = "mlp"
MODEL_NAMES: tuple[str, ...] = (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING, MLP)

IMPUTER_STEP = "imputer"
SCALER_STEP = "scaler"
CLASSIFIER_STEP = "classifier"

_ESTIMATORS = {
    LOGISTIC_REGRESSION: LogisticRegression,
    HIST_GRADIENT_BOOSTING: HistGradientBoostingClassifier,
    MLP: MLPClassifier,
}


def _classifier(spec: ModelSpec, seed: int) -> Any:
    if spec.name not in _ESTIMATORS:
        raise ConfigError(f"model {spec.name!r} has no estimator; known: {MODEL_NAMES}")
    params: dict[str, Any] = dict(spec.fixed)
    if spec.name == MLP and "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(int(h) for h in params["hidden_layer_sizes"])
    params["random_state"] = seed
    return _ESTIMATORS[spec.name](**params)


def build_pipeline(name: str, cfg: ModelConfig, seed: int) -> Pipeline:
    """imputer(median) -> StandardScaler -> classifier, unfitted."""
    spec = cfg.spec(name)
    return Pipeline([
        (IMPUTER_STEP, SimpleImputer(strategy="median")),
        (SCALER_STEP, StandardScaler()),
        (CLASSIFIER_STEP, _classifier(spec, seed)),
    ])


def param_grid(name: str, cfg: ModelConfig) -> dict[str, list[Any]]:
    """GridSearchCV grid with the `classifier__` prefix."""
    spec = cfg.spec(name)
    return {f"{CLASSIFIER_STEP}__{k}": list(v) for k, v in spec.grid.items()}


def grid_size(name: str, cfg: ModelConfig) -> int:
    total = 1
    for values in cfg.spec(name).grid.values():
        total *= len(values)
    return total
