"""Subcommand implementations. These are the only functions that print.

Exit codes: 0 success; 1 an output failed validation; 2 missing input,
unaccepted terms, or a configuration problem.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from heart_risk_calibration import checkpoint as ckpt
from heart_risk_calibration.adapter import FIXTURE_SOURCE_ID, LoadReport, load_cleveland, load_fixture
from heart_risk_calibration.config import (EvalConfig, ModelConfig, configuration_hash,
                                           load_data_config, load_eval_config, load_model_config)
from heart_risk_calibration.download import download_cleveland, terms_message
from heart_risk_calibration.errors import HeartRiskError, ValidationError
from heart_risk_calibration.evaluate import read_metrics, run_nested_cv, validate_metrics, write_result
from heart_risk_calibration.export_web import build_web_document, write_web_document
from heart_risk_calibration.manifest import (build_manifest, check_mode_for_source,
                                             nested_cv_sample_counts, sha256_file, utc_now,
                                             write_manifest)
from heart_risk_calibration.paths import (DATA_CONFIG, EVAL_CONFIG, EVAL_SMOKE_CONFIG, FIXTURE_CSV,
                                          MODEL_CONFIG, MODEL_SMOKE_CONFIG)
from heart_risk_calibration.report import calibration_lines, summary_lines, threshold_lines
from heart_risk_calibration.schema import HeartTable

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_MISSING = 2


def _say(*lines: str) -> None:
    for line in lines:
        print(line)


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _load_source(input_path: Path | None, data_root: str | None,
                 env: Mapping[str, str] | None = None) -> tuple[HeartTable, LoadReport]:
    data_cfg = load_data_config(DATA_CONFIG)
    if input_path is not None:
        return load_fixture(Path(input_path), data_cfg)
    return load_cleveland(data_root, data_cfg, env)


def _run(table: HeartTable, report: LoadReport, mode: str, out_dir: Path, eval_cfg: EvalConfig,
         model_cfg: ModelConfig, models: tuple[str, ...] | None) -> dict[str, Any]:
    """Nested CV on `table`, files under `out_dir`, manifest written. Returns metrics."""
    check_mode_for_source(mode, report.source_id)
    started = utc_now()
    result = run_nested_cv(table, model_cfg, eval_cfg, source_id=report.source_id, mode=mode,
                           load_report=report.as_dict(), model_names=models)
    paths = write_result(result, out_dir)
    manifest = build_manifest(
        mode=mode, status="completed", source_id=report.source_id,
        data_version="v1" if report.source_id == FIXTURE_SOURCE_ID else "cleveland-processed",
        split_manifest_hash=sha256_file(paths["splits"]),
        sample_counts=nested_cv_sample_counts(result.metrics["folds"], eval_cfg.inner_folds),
        configuration_hash=configuration_hash(eval_cfg.as_dict(),
                                              {"models": {k: {"fixed": dict(v.fixed), "grid": dict(v.grid)}
                                                          for k, v in model_cfg.specs.items()}}),
        seed=eval_cfg.seed, metrics_file=str(paths["metrics"].name),
        predictions_file=str(paths["predictions"].name), started_at=started, finished_at=utc_now(),
        extra={"cv": {"protocol": "nested_cv", "outer_folds": eval_cfg.outer_folds,
                      "inner_folds": eval_cfg.inner_folds, "unit": "row (one row is one patient)",
                      "per_fold": result.metrics["folds"],
                      "sample_counts_note": "train/val/test are the fold 1 sizes; see per_fold"},
               "load_report": report.as_dict()},
    )
    write_manifest(Path(out_dir) / "manifest.yaml", manifest)
    return result.metrics


def cmd_evaluate(input_path: Path | None, data_root: str | None, out_dir: Path, mode: str,
                 eval_config: Path = EVAL_CONFIG, models: tuple[str, ...] | None = None) -> int:
    try:
        table, report = _load_source(input_path, data_root)
        eval_cfg = load_eval_config(eval_config)
        model_cfg = load_model_config(MODEL_CONFIG)
        metrics = _run(table, report, mode, out_dir, eval_cfg, model_cfg, models)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    if not report.row_count_matches_documented:
        _say(f"note: read {report.rows_read} rows; the source documents {report.rows_documented}")
    _say(*summary_lines(metrics), f"wrote {Path(out_dir) / 'metrics.json'}")
    return EXIT_OK


def cmd_train(input_path: Path | None, data_root: str | None, out_dir: Path, mode: str,
              models: tuple[str, ...] | None = None) -> int:
    try:
        table, report = _load_source(input_path, data_root)
        check_mode_for_source(mode, report.source_id)
        eval_cfg = load_eval_config(EVAL_CONFIG)
        model_cfg = load_model_config(MODEL_CONFIG)
        started = utc_now()
        out_dir = Path(out_dir)
        written: dict[str, Any] = {}
        for name in models or eval_cfg.models:
            c = ckpt.fit_final(name, table, model_cfg, eval_cfg, source_id=report.source_id, mode=mode)
            path = ckpt.save_checkpoint(c, out_dir / f"{name}{ckpt.CHECKPOINT_SUFFIX}")
            written[name] = {"path": path.name, "threshold": c.threshold,
                             "threshold_source": c.threshold_source, "best_params": dict(c.best_params),
                             "n_train": c.n_train, "sha256": sha256_file(path)}
        report_path = out_dir / "train_report.json"
        report_path.write_text(json.dumps({"source_id": report.source_id, "mode": mode,
                                           "n_rows": table.n_rows, "models": written}, indent=2),
                               encoding="utf-8")
        manifest = build_manifest(
            mode=mode, status="completed", source_id=report.source_id,
            data_version="v1" if report.source_id == FIXTURE_SOURCE_ID else "cleveland-processed",
            split_manifest_hash=sha256_file(report_path),
            sample_counts={"train": table.n_rows, "val": 0, "test": 0},
            configuration_hash=configuration_hash(eval_cfg.as_dict()), seed=eval_cfg.seed,
            metrics_file=None, predictions_file=None, started_at=started, finished_at=utc_now(),
            checkpoint_hash=written[next(iter(written))]["sha256"] if written else None,
            extra={"checkpoints": written, "note": "final fit on all rows; evaluation is the nested CV run"},
        )
        write_manifest(out_dir / "manifest.yaml", manifest)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    for name, info in written.items():
        _say(f"{name}: {info['path']} threshold={info['threshold']:.2f} n_train={info['n_train']}")
    _say(f"wrote {out_dir / 'manifest.yaml'}")
    return EXIT_OK


def cmd_calibration_table(run_dir: Path, model: str | None, csv_out: Path | None) -> int:
    try:
        metrics = read_metrics(Path(run_dir) / "metrics.json")
        validate_metrics(metrics)
        if model is not None and model not in metrics["models"]:
            raise ValidationError(f"model {model!r} not in run; available: {list(metrics['models'])}")
    except ValidationError as exc:
        return _fail(str(exc), EXIT_MISSING)
    _say(*calibration_lines(metrics, model), *threshold_lines(metrics, model))
    if csv_out is not None:
        rows = ["model,bin,lower,upper,n,mean_predicted,fraction_positive"]
        for name in ([model] if model else metrics["models"]):
            for b in metrics["models"][name]["reliability_bins"]:
                rows.append(f"{name},{b['bin']},{b['lower']},{b['upper']},{b['n']},"
                            f"{'' if b['mean_predicted'] is None else b['mean_predicted']},"
                            f"{'' if b['fraction_positive'] is None else b['fraction_positive']}")
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_out).write_text("\n".join(rows) + "\n", encoding="utf-8")
        _say(f"wrote {csv_out}")
    return EXIT_OK


def cmd_smoke(out_dir: Path) -> int:
    try:
        table, report = _load_source(FIXTURE_CSV, None)
        eval_cfg = load_eval_config(EVAL_SMOKE_CONFIG)
        model_cfg = load_model_config(MODEL_SMOKE_CONFIG)
        _run(table, report, "smoke", out_dir, eval_cfg, model_cfg, None)
        metrics_path = Path(out_dir) / "metrics.json"
        validate_metrics(read_metrics(metrics_path))
        for name in ("predictions.csv", "splits.json", "manifest.yaml"):
            if not (Path(out_dir) / name).is_file():
                raise ValidationError(f"expected output {name} missing under {out_dir}")
    except ValidationError as exc:
        return _fail(f"smoke output failed validation: {exc}", EXIT_INVALID)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    _say(f"smoke ok: {metrics_path} validated (synthetic fixture, mode smoke)")
    return EXIT_OK


def cmd_demo(out_dir: Path) -> int:
    try:
        table, report = _load_source(FIXTURE_CSV, None)
        eval_cfg = load_eval_config(EVAL_CONFIG)
        model_cfg = load_model_config(MODEL_CONFIG)
        metrics = _run(table, report, "demo", out_dir, eval_cfg, model_cfg, None)
        validate_metrics(read_metrics(Path(out_dir) / "metrics.json"))
    except ValidationError as exc:
        return _fail(f"demo output failed validation: {exc}", EXIT_INVALID)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    _say(*summary_lines(metrics), "", *calibration_lines(metrics), "", *threshold_lines(metrics),
         f"wrote {Path(out_dir) / 'metrics.json'}")
    return EXIT_OK


def cmd_export_web(run_dir: Path, out_path: Path) -> int:
    metrics_path = Path(run_dir) / "metrics.json"
    try:
        if not metrics_path.is_file():
            _say(f"{metrics_path} not found; running the demo evaluation first")
            code = cmd_demo(Path(run_dir))
            if code != EXIT_OK:
                return code
        doc = build_web_document(read_metrics(metrics_path))
        write_web_document(doc, out_path)
    except ValidationError as exc:
        return _fail(str(exc), EXIT_INVALID)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    _say(f"wrote {out_path} (synthetic={doc['synthetic']}, models={len(doc['models'])})")
    return EXIT_OK


def cmd_download(data_root: str | None, accept_terms: bool, env: Mapping[str, str] | None = None,
                 fetcher=None) -> int:
    data_cfg = load_data_config(DATA_CONFIG)
    _say(terms_message(data_cfg))
    if not accept_terms:
        return EXIT_MISSING
    try:
        value = data_root or (env if env is not None else os.environ).get("DATA_ROOT")
        if not value:
            raise HeartRiskError("pass --data-root DIR or set DATA_ROOT to say where the file goes")
        root = Path(value)
        root.mkdir(parents=True, exist_ok=True)
        kwargs = {"fetcher": fetcher} if fetcher is not None else {}
        dest = download_cleveland(root, data_cfg, accept_terms=True, env=env, **kwargs)
    except HeartRiskError as exc:
        return _fail(str(exc), EXIT_MISSING)
    _say(f"fetched {dest}")
    return EXIT_OK
