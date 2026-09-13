"""Table schema: the 13 attribute columns, the `?` missing marker, and the
binary target mapping `label = 1 if num > 0 else 0`."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.errors import SchemaError

ID_COLUMN = "patient_id"
LABEL_COLUMN = "label"
NUM_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class HeartTable:
    """Validated feature matrix, raw `num`, binary label and row identifiers."""

    ids: tuple[str, ...]
    features: pd.DataFrame
    num: np.ndarray
    label: np.ndarray
    missing_cells: int

    @property
    def n_rows(self) -> int:
        return len(self.ids)

    @property
    def n_positive(self) -> int:
        return int(self.label.sum())

    @property
    def n_negative(self) -> int:
        return int(self.n_rows - self.n_positive)


def map_target(num: np.ndarray) -> np.ndarray:
    """Binary label: 1 when `num` is greater than zero, else 0.

    The source field `num` takes integer values 0..4. Every value above zero
    is treated as the positive class; zero is the negative class.
    """
    arr = np.asarray(num)
    if arr.ndim != 1:
        raise SchemaError("num must be one-dimensional")
    if np.isnan(arr.astype(float)).any():
        raise SchemaError("num contains missing values; the target cannot be missing")
    ints = arr.astype(float)
    if not np.array_equal(ints, np.round(ints)) or not np.isin(ints, NUM_VALUES).all():
        bad = sorted(set(ints[~np.isin(ints, NUM_VALUES)].tolist()))
        raise SchemaError(f"num must take values in {NUM_VALUES}; found {bad}")
    return (ints > 0).astype(int)


def replace_missing_marker(frame: pd.DataFrame, marker: str) -> pd.DataFrame:
    """Return a copy in which every cell equal to `marker` (after stripping) is NaN."""
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        # pandas 3 reads text as the `str` dtype, not `object`; test for non-numeric instead
        if not pd.api.types.is_numeric_dtype(series):
            stripped = series.astype(str).str.strip()
            out[col] = series.where(stripped != marker, other=np.nan)
    return out


def standardise(frame: pd.DataFrame, cfg: DataConfig) -> HeartTable:
    """Validate a raw frame and build a HeartTable.

    Accepts an optional leading `patient_id` column; otherwise identifiers are
    row positions prefixed `ROW-`. Feature columns are coerced to float; the
    missing marker becomes NaN and is counted in `missing_cells`. The target is
    mapped with `map_target`.
    """
    frame = replace_missing_marker(frame, cfg.missing_marker)
    if ID_COLUMN in frame.columns:
        ids = tuple(str(v) for v in frame[ID_COLUMN].tolist())
        body = frame.drop(columns=[ID_COLUMN])
    else:
        ids = tuple(f"ROW-{i + 1:04d}" for i in range(len(frame)))
        body = frame
    expected = list(cfg.all_columns)
    if list(body.columns) != expected:
        raise SchemaError(
            f"expected columns {expected}; got {list(body.columns)}"
        )
    if len(body) == 0:
        raise SchemaError("the table has no rows")
    if len(set(ids)) != len(ids):
        raise SchemaError("patient_id values are not unique")
    features = pd.DataFrame(index=body.index)
    for col in cfg.feature_columns:
        try:
            features[col] = pd.to_numeric(body[col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"column {col!r} holds a non-numeric value: {exc}") from exc
    try:
        num = pd.to_numeric(body[cfg.target_column], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"target column {cfg.target_column!r} holds a non-numeric value: {exc}") from exc
    label = map_target(num)
    missing = int(features.isna().sum().sum())
    features = features.reset_index(drop=True)
    return HeartTable(ids=ids, features=features, num=num.astype(int), label=label,
                      missing_cells=missing)
