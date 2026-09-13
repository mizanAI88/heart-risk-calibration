"""Command-line behaviour: help text, exit codes, output files, wording."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
import yaml

from heart_risk_calibration import commands
from heart_risk_calibration.checkpoint import load_checkpoint
from heart_risk_calibration.cli import build_parser, main
from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.download import download_cleveland
from heart_risk_calibration.errors import TermsNotAcceptedError
from heart_risk_calibration.paths import FIXTURE_CSV
from tests.conftest import CommandRun

FORBIDDEN = ("recommend", "diagnos")


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:  # argparse --help exits
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def all_help_text() -> str:
    parser = build_parser()
    texts = [parser.format_help()]
    for action in parser._subparsers._group_actions:  # noqa: SLF001 (introspecting our own parser)
        for sub in action.choices.values():
            texts.append(sub.format_help())
    return "\n".join(texts)


def test_help_output_has_no_clinical_use_language():
    text = all_help_text().lower()
    for word in FORBIDDEN:
        assert word not in text, word
    code, out, _ = run(["--help"])
    assert code == 0 and "smoke" in out and "export-web" in out


def test_demo_output_has_no_clinical_use_language_and_is_labelled_synthetic(demo_run: CommandRun):
    assert demo_run.code == 0, demo_run.stderr
    lowered = (demo_run.stdout + demo_run.stderr).lower()
    for word in FORBIDDEN:
        assert word not in lowered, word
    assert "synthetic" in lowered and "not a benchmark result" in lowered
    assert (demo_run.out_dir / "metrics.json").is_file()
    manifest = yaml.safe_load((demo_run.out_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["mode"] == "demo"


def test_smoke_exits_zero_and_output_file_exists(smoke_run: CommandRun):
    assert smoke_run.code == 0
    assert (smoke_run.out_dir / "metrics.json").is_file()
    assert "smoke ok" in smoke_run.stdout


def test_evaluate_without_data_root_or_file_exits_two_with_guidance(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    code, _, err = run(["evaluate", "--out", str(tmp_path)])
    assert code == 2
    assert "DATA_ROOT" in err and "--data-root" in err
    code, _, err = run(["evaluate", "--data-root", str(tmp_path), "--out", str(tmp_path / "o")])
    assert code == 2
    assert "processed.cleveland.data" in err
    assert not (tmp_path / "o" / "metrics.json").exists()


def test_evaluate_refuses_experiment_mode_on_the_synthetic_fixture(tmp_path: Path):
    code, _, err = run(["evaluate", "--input", str(FIXTURE_CSV), "--out", str(tmp_path), "--mode", "experiment"])
    assert code == 2
    assert "synthetic" in err and "never experiment" in err


def test_download_without_accept_terms_prints_licence_and_fetches_nothing(tmp_path: Path):
    code, out, _ = run(["download", "--data-root", str(tmp_path)])
    assert code == 2
    assert "CC BY 4.0" in out and "creativecommons.org" in out
    assert "Nothing has been downloaded" in out
    assert not any(tmp_path.iterdir())


def test_download_is_refused_in_ci_even_with_accept_terms(data_cfg: DataConfig, tmp_path: Path):
    calls: list[str] = []

    def fetcher(url: str, dest: Path) -> int:
        calls.append(url)
        return 1

    with pytest.raises(TermsNotAcceptedError, match="CI"):
        download_cleveland(tmp_path, data_cfg, accept_terms=True, fetcher=fetcher, env={"CI": "true"})
    assert calls == []


def test_download_with_accept_terms_uses_the_fetcher_and_writes_the_file(tmp_path: Path):
    def fetcher(url: str, dest: Path) -> int:
        dest.write_text("1,2,3\n", encoding="utf-8")
        return 6

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = commands.cmd_download(str(tmp_path), True, env={}, fetcher=fetcher)
    assert code == 0
    assert (tmp_path / "processed.cleveland.data").read_text(encoding="utf-8") == "1,2,3\n"
    assert "fetched" in out.getvalue()


def test_export_web_json_is_flagged_synthetic_with_per_model_fields(demo_run: CommandRun, tmp_path: Path):
    target = tmp_path / "web.json"
    code, out, err = run(["export-web", "--run-dir", str(demo_run.out_dir), "--out", str(target)])
    assert code == 0, err
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["synthetic"] is True
    assert "not a benchmark result" in doc["label"]
    assert doc["n"] == 300
    names = {m["name"] for m in doc["models"]}
    assert names == {"logistic_regression", "hist_gradient_boosting", "mlp"}
    for m in doc["models"]:
        assert set(m) >= {"auroc", "brier", "ece", "n", "reliability_bins"}
        assert len(m["reliability_bins"]) == 10
        assert sum(b["n"] for b in m["reliability_bins"]) == 300


def test_calibration_table_prints_ten_bins_per_model(smoke_run: CommandRun, tmp_path: Path):
    csv_path = tmp_path / "bins.csv"
    code, out, _ = run(["calibration-table", "--run-dir", str(smoke_run.out_dir), "--csv", str(csv_path)])
    assert code == 0
    assert out.count("model: ") == 6, "three reliability headers and three threshold headers"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("model,bin,lower,upper,n")
    assert len(lines) == 1 + 3 * 10
    for word in FORBIDDEN:
        assert word not in out.lower()
    code, _, err = run(["calibration-table", "--run-dir", str(tmp_path / "missing")])
    assert code == 2 and "does not exist" in err


def test_train_writes_loadable_checkpoints_and_manifest(tmp_path: Path):
    code, out, err = run(["train", "--input", str(FIXTURE_CSV), "--out", str(tmp_path), "--mode", "demo",
                          "--models", "logistic_regression"])
    assert code == 0, err
    ckpt = load_checkpoint(tmp_path / "logistic_regression.pkl")
    assert ckpt.n_train == 300 and 0 < ckpt.threshold < 1
    assert ckpt.threshold_source == "inner_out_of_fold_validation_predictions"
    from heart_risk_calibration.adapter import load_fixture
    from heart_risk_calibration.config import load_data_config
    from heart_risk_calibration.paths import DATA_CONFIG

    table, _ = load_fixture(FIXTURE_CSV, load_data_config(DATA_CONFIG))
    proba = ckpt.predict_proba(table.features.head(5))
    assert proba.shape == (5,) and ((proba >= 0) & (proba <= 1)).all()
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["mode"] == "demo" and manifest["checkpoint_hash"]
    assert "threshold=" in out
    code, _, err = run(["train", "--input", str(FIXTURE_CSV), "--out", str(tmp_path / "exp")])
    assert code == 2 and "synthetic" in err, "experiment mode is refused on the synthetic fixture"
