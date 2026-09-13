"""The nested CV driver, the metrics document, and the run manifest."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from heart_risk_calibration.config import DataConfig, EvalConfig, ModelConfig
from heart_risk_calibration.errors import ConfigError, ValidationError
from heart_risk_calibration.evaluate import read_metrics, run_nested_cv, validate_metrics
from heart_risk_calibration.manifest import (build_manifest, check_mode_for_source,
                                             nested_cv_sample_counts)
from tests.conftest import CommandRun, small_table


@pytest.fixture(scope="module")
def small_result(data_cfg: DataConfig, smoke_model_cfg: ModelConfig, smoke_eval_cfg: EvalConfig):
    table = small_table(data_cfg, seed=21, n_rows=90)
    return run_nested_cv(table, smoke_model_cfg, smoke_eval_cfg, source_id="synthetic-test",
                         mode="smoke", model_names=("logistic_regression",))


def test_metrics_carry_n_per_fold_that_sum_to_n_rows(small_result):
    m = small_result.metrics
    assert len(m["folds"]) == 5
    assert sum(f["n_test"] for f in m["folds"]) == m["n_rows"] == 90
    for f in m["folds"]:
        assert f["n_train"] + f["n_test"] == 90
        assert f["n_test_positive"] >= 1
    per_fold = m["models"]["logistic_regression"]["per_fold"]
    assert [r["n_test"] for r in per_fold] == [f["n_test"] for f in m["folds"]]
    assert all(r["threshold_source"] == "inner_out_of_fold_validation_predictions" for r in per_fold)


def test_each_patient_is_predicted_exactly_once_per_model(small_result):
    preds = small_result.predictions
    counts = preds.groupby("model")["patient_id"].value_counts()
    assert (counts == 1).all()
    assert set(preds["y_pred"]) <= {0, 1}
    assert ((preds["p_hat"] >= preds["threshold"]).astype(int) == preds["y_pred"]).all()


def test_metrics_document_validates_and_rejects_count_mismatch(small_result):
    validate_metrics(small_result.metrics)
    broken = copy.deepcopy(small_result.metrics)
    broken["folds"][0]["n_test"] += 1
    with pytest.raises(ValidationError, match="do not sum"):
        validate_metrics(broken)
    broken = copy.deepcopy(small_result.metrics)
    broken["models"]["logistic_regression"]["pooled"]["auroc"] = 1.5
    with pytest.raises(ValidationError, match="auroc"):
        validate_metrics(broken)
    broken = copy.deepcopy(small_result.metrics)
    broken["models"]["logistic_regression"]["reliability_bins"].pop()
    with pytest.raises(ValidationError, match="reliability bins"):
        validate_metrics(broken)


def test_reliability_bins_and_sensitivity_table_shapes(small_result, smoke_eval_cfg: EvalConfig):
    body = small_result.metrics["models"]["logistic_regression"]
    assert len(body["reliability_bins"]) == smoke_eval_cfg.n_bins == 10
    assert len(body["threshold_sensitivity"]) == len(smoke_eval_cfg.threshold_grid)
    assert body["pooled"]["ci"]["unit"] == "row (patient)"
    assert body["pooled"]["n"] == 90


def test_synthetic_source_cannot_run_in_experiment_mode():
    with pytest.raises(ConfigError, match="synthetic"):
        check_mode_for_source("experiment", "synthetic-fixture-v1")
    check_mode_for_source("demo", "synthetic-fixture-v1")
    check_mode_for_source("experiment", "uci-heart-disease-cleveland")
    with pytest.raises(ConfigError, match="mode must be"):
        check_mode_for_source("production", "uci-heart-disease-cleveland")


def test_manifest_shape_and_sample_counts_from_folds():
    folds = [{"fold": 1, "n_train": 240, "n_test": 60}, {"fold": 2, "n_train": 240, "n_test": 60}]
    counts = nested_cv_sample_counts(folds, inner_folds=3)
    assert counts == {"train": 240, "val": 80, "test": 60}
    manifest = build_manifest(mode="demo", status="completed", source_id="synthetic-fixture-v1",
                              data_version="v1", split_manifest_hash="abc", sample_counts=counts,
                              configuration_hash="def", seed=42, metrics_file="metrics.json",
                              predictions_file="predictions.csv",
                              started_at="2026-09-13T00:00:00+00:00",
                              finished_at="2026-09-13T00:00:10+00:00")
    assert manifest["schema_version"] == 1 and manifest["git_commit"] is None
    assert manifest["project_id"] == "heart-risk-calibration"
    assert manifest["run_id"].endswith("heart-risk-calibration")
    assert manifest["environment"]["device"] == "cpu"
    with pytest.raises(ConfigError):
        build_manifest(mode="demo", status="done", source_id="x", data_version="v1",
                       split_manifest_hash="a", sample_counts=counts, configuration_hash="d",
                       seed=1, metrics_file=None, predictions_file=None, started_at="t", finished_at="t")


def test_smoke_run_writes_manifest_whose_split_hash_matches_splits_json(smoke_run: CommandRun):
    assert smoke_run.code == 0, smoke_run.stderr
    manifest = yaml.safe_load((smoke_run.out_dir / "manifest.yaml").read_text(encoding="utf-8"))
    digest = hashlib.sha256((smoke_run.out_dir / "splits.json").read_bytes()).hexdigest()
    assert manifest["data"]["split_manifest_hash"] == digest
    assert manifest["mode"] == "smoke"
    assert manifest["data"]["source_id"] == "synthetic-fixture-v1"
    assert manifest["data"]["sample_counts"] == {"train": 240, "val": 80, "test": 60}
    assert len(manifest["cv"]["per_fold"]) == 5
    assert manifest["status"] == "completed"


def test_smoke_run_is_a_synthetic_end_to_end_under_sixty_seconds(smoke_run: CommandRun):
    assert smoke_run.code == 0
    assert smoke_run.seconds < 60, f"smoke took {smoke_run.seconds:.1f} s"
    metrics = read_metrics(smoke_run.out_dir / "metrics.json")
    assert metrics["synthetic"] is True and metrics["mode"] == "smoke"
    assert set(metrics["models"]) == {"logistic_regression", "hist_gradient_boosting", "mlp"}
    preds = pd.read_csv(smoke_run.out_dir / "predictions.csv")
    assert len(preds) == 300 * 3


def test_read_metrics_rejects_missing_and_malformed_files(tmp_path: Path):
    with pytest.raises(ValidationError, match="does not exist"):
        read_metrics(tmp_path / "metrics.json")
    bad = tmp_path / "metrics.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        read_metrics(bad)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert read_metrics(good) == {"a": 1}
