"""Repository-relative paths. Nothing here is absolute; everything derives
from the location of this file so the package works from any checkout."""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
EXAMPLES_DIR = REPO_ROOT / "examples"
FIXTURE_DIR = EXAMPLES_DIR / "fixtures" / "v1"
FIXTURE_CSV = FIXTURE_DIR / "synthetic_heart.csv"
DEMO_OUT_DIR = EXAMPLES_DIR / "output" / "demo"
WEB_FIXTURE_JSON = EXAMPLES_DIR / "output" / "web_fixture.json"
DEFAULT_OUT_DIR = REPO_ROOT / "out"

DATA_CONFIG = CONFIGS_DIR / "data.yaml"
MODEL_CONFIG = CONFIGS_DIR / "model.yaml"
EVAL_CONFIG = CONFIGS_DIR / "eval.yaml"
EVAL_SMOKE_CONFIG = CONFIGS_DIR / "eval_smoke.yaml"
MODEL_SMOKE_CONFIG = CONFIGS_DIR / "model_smoke.yaml"
