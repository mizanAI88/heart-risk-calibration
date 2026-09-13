"""Shared fixtures: configs, the authored fixture table, small in-memory
tables, and one real smoke run and one real demo run shared across tests."""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from heart_risk_calibration import commands
from heart_risk_calibration.adapter import load_fixture
from heart_risk_calibration.cli import main
from heart_risk_calibration.config import (DataConfig, EvalConfig, ModelConfig, load_data_config,
                                           load_eval_config, load_model_config)
from heart_risk_calibration.paths import (DATA_CONFIG, EVAL_CONFIG, EVAL_SMOKE_CONFIG, FIXTURE_CSV,
                                          MODEL_CONFIG, MODEL_SMOKE_CONFIG, REPO_ROOT)
from heart_risk_calibration.schema import HeartTable, standardise
from heart_risk_calibration.synthetic import FixtureSpec, generate


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_cfg() -> DataConfig:
    return load_data_config(DATA_CONFIG)


@pytest.fixture(scope="session")
def model_cfg() -> ModelConfig:
    return load_model_config(MODEL_CONFIG)


@pytest.fixture(scope="session")
def smoke_model_cfg() -> ModelConfig:
    return load_model_config(MODEL_SMOKE_CONFIG)


@pytest.fixture(scope="session")
def eval_cfg() -> EvalConfig:
    return load_eval_config(EVAL_CONFIG)


@pytest.fixture(scope="session")
def smoke_eval_cfg() -> EvalConfig:
    return load_eval_config(EVAL_SMOKE_CONFIG)


@pytest.fixture(scope="session")
def fixture_table(data_cfg: DataConfig) -> HeartTable:
    assert FIXTURE_CSV.is_file(), "run tools/make_fixtures.py --seed 42 first"
    table, _ = load_fixture(FIXTURE_CSV, data_cfg)
    return table


def small_table(data_cfg: DataConfig, seed: int = 7, n_rows: int = 90) -> HeartTable:
    """A small synthetic table built in memory (never written to disk)."""
    return standardise(generate(FixtureSpec(seed=seed, n_rows=n_rows)), data_cfg)


@pytest.fixture(scope="session")
def small(data_cfg: DataConfig) -> HeartTable:
    return small_table(data_cfg)


@dataclass(frozen=True)
class CommandRun:
    code: int
    out_dir: Path
    stdout: str
    stderr: str
    seconds: float


def _run_command(argv: list[str], out: Path) -> CommandRun:
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    seconds = time.perf_counter() - started
    return CommandRun(code=code, out_dir=out, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                      seconds=seconds)


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> CommandRun:
    out = tmp_path_factory.mktemp("smoke")
    return _run_command(["smoke", "--out", str(out)], out)


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> CommandRun:
    out = tmp_path_factory.mktemp("demo")
    return _run_command(["demo", "--out", str(out)], out)


@pytest.fixture
def commands_module():
    return commands
