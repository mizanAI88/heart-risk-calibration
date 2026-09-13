"""Target mapping, `?` handling, and the data-root adapter's failure modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_risk_calibration.adapter import (cleveland_path, load_cleveland, load_fixture,
                                            read_raw_cleveland, resolve_data_root)
from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.errors import DataRootError, SchemaError
from heart_risk_calibration.paths import FIXTURE_CSV
from heart_risk_calibration.schema import map_target, replace_missing_marker, standardise

ROW_A = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0"
ROW_B = "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2"
ROW_MISSING = "52.0,1.0,3.0,138.0,223.0,0.0,0.0,169.0,0.0,0.0,1.0,?,3.0,0"


def write_cleveland(root: Path, rows: list[str], name: str = "processed.cleveland.data") -> Path:
    path = root / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_target_mapping_zero_is_negative_and_one_to_four_positive():
    label = map_target(np.array([0, 1, 2, 3, 4, 0]))
    assert label.tolist() == [0, 1, 1, 1, 1, 0]


def test_target_mapping_rejects_out_of_range_and_missing_values():
    with pytest.raises(SchemaError, match="num must take values"):
        map_target(np.array([0, 5]))
    with pytest.raises(SchemaError, match="cannot be missing"):
        map_target(np.array([0.0, np.nan]))


def test_question_mark_becomes_nan_and_is_counted(data_cfg: DataConfig, tmp_path: Path):
    path = write_cleveland(tmp_path, [ROW_A, ROW_B, ROW_MISSING])
    table = standardise(read_raw_cleveland(path, data_cfg), data_cfg)
    assert table.missing_cells == 1
    assert np.isnan(table.features.loc[2, "ca"])
    assert table.features.dtypes.eq(float).all()
    assert table.label.tolist() == [0, 1, 0]


def test_replace_missing_marker_only_touches_exact_marker():
    frame = pd.DataFrame({"a": ["1", " ? ", "?x"], "b": [1.0, 2.0, 3.0]})
    out = replace_missing_marker(frame, "?")
    assert out["a"].isna().tolist() == [False, True, False]
    assert frame["a"].tolist() == ["1", " ? ", "?x"], "input frame must not be mutated"


def test_standardise_rejects_wrong_columns_and_non_numeric_cells(data_cfg: DataConfig, tmp_path: Path):
    with pytest.raises(SchemaError, match="expected columns"):
        standardise(pd.DataFrame({"age": ["1"], "num": ["0"]}), data_cfg)
    path = write_cleveland(tmp_path, [ROW_A.replace("233.0", "abc"), ROW_B])
    with pytest.raises(SchemaError, match="non-numeric"):
        standardise(read_raw_cleveland(path, data_cfg), data_cfg)


def test_missing_data_root_names_env_var_and_flag():
    with pytest.raises(DataRootError, match="DATA_ROOT") as info:
        resolve_data_root(None, env={})
    assert "--data-root" in str(info.value)


def test_data_root_without_file_names_file_and_source(data_cfg: DataConfig, tmp_path: Path):
    with pytest.raises(DataRootError) as info:
        cleveland_path(tmp_path, data_cfg)
    message = str(info.value)
    assert "processed.cleveland.data" in message
    assert data_cfg.dataset_url in message
    assert "download --accept-terms" in message


def test_load_cleveland_reads_env_root_and_reports_row_count(data_cfg: DataConfig, tmp_path: Path):
    write_cleveland(tmp_path, [ROW_A, ROW_B, ROW_MISSING])
    table, report = load_cleveland(None, data_cfg, env={"DATA_ROOT": str(tmp_path)})
    assert table.n_rows == 3
    assert report.rows_read == 3
    assert report.rows_documented == 303
    assert report.row_count_matches_documented is False
    assert table.ids[0] == "ROW-0001"


def test_wrong_column_count_is_a_schema_error(data_cfg: DataConfig, tmp_path: Path):
    path = write_cleveland(tmp_path, ["1,2,3", "4,5,6"])
    with pytest.raises(SchemaError, match="expected 14"):
        read_raw_cleveland(path, data_cfg)


def test_fixture_loads_with_300_rows_13_features_and_synthetic_ids(data_cfg: DataConfig):
    table, report = load_fixture(FIXTURE_CSV, data_cfg)
    assert table.n_rows == 300
    assert list(table.features.columns) == list(data_cfg.feature_columns)
    assert all(pid.startswith("SYN-P") for pid in table.ids)
    assert report.source_id == "synthetic-fixture-v1"
    assert table.missing_cells > 0, "the fixture plants `?` cells so the imputer path is exercised"
    with pytest.raises(DataRootError, match="make_fixtures"):
        load_fixture(FIXTURE_CSV.with_name("nope.csv"), data_cfg)
