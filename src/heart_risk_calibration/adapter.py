"""Readers for the user-obtained Cleveland file and for the synthetic fixture.

No data is shipped. `load_cleveland` reads `DATA_ROOT/processed.cleveland.data`
and raises `DataRootError` naming what to obtain and from where when it is
absent. The `?` marker is turned into NaN here; imputation of those NaNs is
performed inside each cross-validation fold by the model pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.errors import DataRootError, SchemaError
from heart_risk_calibration.schema import HeartTable, standardise

DATA_ROOT_ENV = "DATA_ROOT"
FIXTURE_SOURCE_ID = "synthetic-fixture-v1"


@dataclass(frozen=True)
class LoadReport:
    """What the adapter actually read, next to what the source documents."""

    path: str
    source_id: str
    rows_read: int
    rows_documented: int
    missing_cells: int

    @property
    def row_count_matches_documented(self) -> bool:
        return self.rows_read == self.rows_documented

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source_id": self.source_id,
            "rows_read": self.rows_read,
            "rows_documented": self.rows_documented,
            "row_count_matches_documented": self.row_count_matches_documented,
            "missing_cells": self.missing_cells,
        }


def resolve_data_root(data_root: str | Path | None, env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    value = data_root if data_root not in (None, "") else environ.get(DATA_ROOT_ENV)
    if not value:
        raise DataRootError(
            "no data root given: pass --data-root DIR or set the DATA_ROOT environment "
            "variable to the directory holding processed.cleveland.data"
        )
    root = Path(value)
    if not root.is_dir():
        raise DataRootError(f"data root {root} is not a directory")
    return root


def cleveland_path(root: Path, cfg: DataConfig) -> Path:
    path = Path(root) / cfg.file_name
    if not path.is_file():
        raise DataRootError(
            f"{cfg.file_name} not found under {root}. Obtain the UCI Heart Disease data "
            f"({cfg.dataset_url}, licence {cfg.licence}) and place the Cleveland processed "
            f"file at {path}, or run `python -m heart_risk_calibration download --accept-terms`."
        )
    return path


def read_raw_cleveland(path: Path, cfg: DataConfig) -> pd.DataFrame:
    """Read the header-less comma-separated source file as strings."""
    try:
        frame = pd.read_csv(path, header=None, dtype=str, skipinitialspace=True,
                            skip_blank_lines=True)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"{path} is empty") from exc
    if frame.shape[1] != len(cfg.all_columns):
        raise SchemaError(
            f"{path}: expected {len(cfg.all_columns)} comma-separated columns, found {frame.shape[1]}"
        )
    frame.columns = list(cfg.all_columns)
    return frame


def load_cleveland(data_root: str | Path | None, cfg: DataConfig,
                   env: Mapping[str, str] | None = None) -> tuple[HeartTable, LoadReport]:
    root = resolve_data_root(data_root, env)
    path = cleveland_path(root, cfg)
    table = standardise(read_raw_cleveland(path, cfg), cfg)
    report = LoadReport(path=str(path.name), source_id=cfg.source_id, rows_read=table.n_rows,
                        rows_documented=cfg.expected_rows, missing_cells=table.missing_cells)
    return table, report


def read_fixture_csv(path: Path, cfg: DataConfig) -> pd.DataFrame:
    """Read the authored synthetic fixture: `#` comment lines, then a header row."""
    path = Path(path)
    if not path.is_file():
        raise DataRootError(
            f"fixture {path} not found; regenerate it with `python tools/make_fixtures.py --seed 42`"
        )
    return pd.read_csv(path, comment="#", dtype=str, skipinitialspace=True)


def load_fixture(path: Path, cfg: DataConfig) -> tuple[HeartTable, LoadReport]:
    table = standardise(read_fixture_csv(path, cfg), cfg)
    report = LoadReport(path=str(Path(path).name), source_id=FIXTURE_SOURCE_ID,
                        rows_read=table.n_rows, rows_documented=table.n_rows,
                        missing_cells=table.missing_cells)
    return table, report
