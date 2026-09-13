"""Run manifest writer (contract section 5). Hashes are computed from the
actual inputs; git_commit stays null until tooling fills it at push time."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from heart_risk_calibration import PROJECT_ID
from heart_risk_calibration.errors import ConfigError

MANIFEST_SCHEMA_VERSION = 1
MODES: tuple[str, ...] = ("smoke", "demo", "experiment")
STATUSES: tuple[str, ...] = ("completed", "failed", "interrupted")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(started_at: str, slug: str = PROJECT_ID) -> str:
    stamp = started_at.replace("+00:00", "Z").replace("-", "").replace(":", "")
    return f"{stamp}-{slug}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_mode_for_source(mode: str, source_id: str) -> None:
    """Synthetic sources may only run in smoke or demo mode."""
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
    if source_id.startswith("synthetic") and mode == "experiment":
        raise ConfigError(
            f"source {source_id!r} is synthetic; use --mode demo or --mode smoke, never experiment"
        )


def nested_cv_sample_counts(folds: list[dict[str, Any]], inner_folds: int) -> dict[str, int]:
    """Counts for fold 1: outer train, one inner validation fold of it, outer test.

    Under nested cross-validation every row is a test row exactly once, so a
    single train/val/test triple cannot describe the run; the manifest carries
    the fold 1 triple here and the full per-fold table under `cv`.
    """
    if not folds:
        raise ConfigError("no folds to summarise")
    first = folds[0]
    n_train = int(first["n_train"])
    return {"train": n_train, "val": n_train // inner_folds, "test": int(first["n_test"])}


def build_manifest(
    *,
    mode: str,
    status: str,
    source_id: str,
    data_version: str,
    split_manifest_hash: str,
    sample_counts: Mapping[str, int],
    configuration_hash: str,
    seed: int | None,
    metrics_file: str | None,
    predictions_file: str | None,
    started_at: str,
    finished_at: str,
    checkpoint_hash: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
    if status not in STATUSES:
        raise ConfigError(f"status must be one of {STATUSES}, got {status!r}")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "run_id": make_run_id(started_at),
        "status": status,
        "mode": mode,
        "git_commit": None,
        "data": {
            "source_id": source_id,
            "version": data_version,
            "split_manifest_hash": split_manifest_hash,
            "sample_counts": {
                "train": int(sample_counts.get("train", 0)),
                "val": int(sample_counts.get("val", 0)),
                "test": int(sample_counts.get("test", 0)),
            },
        },
        "configuration_hash": configuration_hash,
        "seed": seed,
        "environment": {"lockfile_hash": None, "device": "cpu"},
        "checkpoint_hash": checkpoint_hash,
        "metrics_file": metrics_file,
        "predictions_file": predictions_file,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
