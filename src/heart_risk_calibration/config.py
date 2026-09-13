"""Typed, validated views of the yaml files under configs/."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from heart_risk_calibration.errors import ConfigError

THRESHOLD_RULES: tuple[str, ...] = ("youden", "f1")


def _read_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"config {path} must be a mapping at top level")
    return loaded


def _require(mapping: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in mapping:
        raise ConfigError(f"config {path} is missing required key {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class DataConfig:
    source_id: str
    file_name: str
    dataset_url: str
    licence: str
    licence_url: str
    download_url: str
    expected_rows: int
    missing_marker: str
    feature_columns: tuple[str, ...]
    target_column: str
    positive_rule: str

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.feature_columns + (self.target_column,)


def load_data_config(path: Path) -> DataConfig:
    raw = _read_yaml(path)
    features = tuple(str(c) for c in _require(raw, "feature_columns", path))
    if len(features) != 13 or len(set(features)) != 13:
        raise ConfigError(f"config {path}: feature_columns must list 13 distinct names")
    rule = str(_require(raw, "positive_rule", path))
    if rule != "num_greater_than_zero":
        raise ConfigError(f"config {path}: unsupported positive_rule {rule!r}")
    return DataConfig(
        source_id=str(_require(raw, "source_id", path)),
        file_name=str(_require(raw, "file_name", path)),
        dataset_url=str(_require(raw, "dataset_url", path)),
        licence=str(_require(raw, "licence", path)),
        licence_url=str(_require(raw, "licence_url", path)),
        download_url=str(_require(raw, "download_url", path)),
        expected_rows=int(_require(raw, "expected_rows", path)),
        missing_marker=str(_require(raw, "missing_marker", path)),
        feature_columns=features,
        target_column=str(_require(raw, "target_column", path)),
        positive_rule=rule,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fixed: Mapping[str, Any]
    grid: Mapping[str, tuple[Any, ...]]


@dataclass(frozen=True)
class ModelConfig:
    specs: Mapping[str, ModelSpec]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.specs)

    def spec(self, name: str) -> ModelSpec:
        if name not in self.specs:
            raise ConfigError(f"unknown model {name!r}; configured models: {self.names}")
        return self.specs[name]


def load_model_config(path: Path) -> ModelConfig:
    raw = _read_yaml(path)
    models = _require(raw, "models", path)
    if not isinstance(models, dict) or not models:
        raise ConfigError(f"config {path}: 'models' must be a non-empty mapping")
    specs: dict[str, ModelSpec] = {}
    for name, body in models.items():
        body = body or {}
        fixed = dict(body.get("fixed") or {})
        grid_raw = body.get("grid") or {}
        grid = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in grid_raw.items()}
        for k, values in grid.items():
            if not values:
                raise ConfigError(f"config {path}: grid for {name}.{k} is empty")
        specs[str(name)] = ModelSpec(name=str(name), fixed=fixed, grid=grid)
    return ModelConfig(specs=specs)


@dataclass(frozen=True)
class EvalConfig:
    protocol: str
    outer_folds: int
    inner_folds: int
    seed: int
    inner_scoring: str
    threshold_rule: str
    threshold_grid: tuple[float, ...]
    n_bins: int
    bootstrap_resamples: int
    models: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "outer_folds": self.outer_folds,
            "inner_folds": self.inner_folds,
            "seed": self.seed,
            "inner_scoring": self.inner_scoring,
            "threshold_rule": self.threshold_rule,
            "threshold_grid": list(self.threshold_grid),
            "n_bins": self.n_bins,
            "bootstrap_resamples": self.bootstrap_resamples,
            "models": list(self.models),
        }


def _threshold_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    if not (0.0 < start < stop < 1.0) or step <= 0:
        raise ConfigError("threshold grid must satisfy 0 < start < stop < 1 and step > 0")
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 6) for i in range(count))


def load_eval_config(path: Path) -> EvalConfig:
    raw = _read_yaml(path)
    protocol = str(_require(raw, "protocol", path))
    if protocol != "nested_cv":
        raise ConfigError(f"config {path}: only protocol 'nested_cv' is implemented, got {protocol!r}")
    outer = int(_require(raw, "outer_folds", path))
    inner = int(_require(raw, "inner_folds", path))
    if outer < 2 or inner < 2:
        raise ConfigError(f"config {path}: outer_folds and inner_folds must be >= 2")
    rule = str(_require(raw, "threshold_rule", path))
    if rule not in THRESHOLD_RULES:
        raise ConfigError(f"config {path}: threshold_rule must be one of {THRESHOLD_RULES}")
    n_bins = int(_require(raw, "n_bins", path))
    if n_bins < 2:
        raise ConfigError(f"config {path}: n_bins must be >= 2")
    boots = int(_require(raw, "bootstrap_resamples", path))
    if boots < 1:
        raise ConfigError(f"config {path}: bootstrap_resamples must be >= 1")
    models = tuple(str(m) for m in _require(raw, "models", path))
    if not models:
        raise ConfigError(f"config {path}: models list is empty")
    return EvalConfig(
        protocol=protocol,
        outer_folds=outer,
        inner_folds=inner,
        seed=int(_require(raw, "seed", path)),
        inner_scoring=str(_require(raw, "inner_scoring", path)),
        threshold_rule=rule,
        threshold_grid=_threshold_grid(
            float(_require(raw, "threshold_grid_start", path)),
            float(_require(raw, "threshold_grid_stop", path)),
            float(_require(raw, "threshold_grid_step", path)),
        ),
        n_bins=n_bins,
        bootstrap_resamples=boots,
        models=models,
    )


def configuration_hash(*parts: Mapping[str, Any]) -> str:
    """sha256 of the resolved configuration, serialised deterministically."""
    payload = json.dumps([dict(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
