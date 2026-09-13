"""Compact JSON for a web page: per-model AUROC, Brier, ECE, n and the
reliability bins of one run, flagged synthetic when the source was."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from heart_risk_calibration import PROJECT_ID, __version__
from heart_risk_calibration.errors import ValidationError
from heart_risk_calibration.evaluate import validate_metrics

WEB_SCHEMA_VERSION = 1
SYNTHETIC_LABEL = ("synthetic fixture run on authored demonstration data; "
                   "not a benchmark result")


def build_web_document(metrics: dict[str, Any]) -> dict[str, Any]:
    validate_metrics(metrics)
    synthetic = bool(metrics["synthetic"])
    models = []
    for name, body in metrics["models"].items():
        pooled = body["pooled"]
        models.append({
            "name": name,
            "n": int(pooled["n"]),
            "auroc": pooled["auroc"],
            "auprc": pooled["auprc"],
            "brier": pooled["brier"],
            "ece": pooled["ece"],
            "ci": {m: [pooled["ci"][m]["lower"], pooled["ci"][m]["upper"]]
                   for m in ("auroc", "auprc", "brier", "ece")},
            "reliability_bins": [
                {"lower": b["lower"], "upper": b["upper"], "n": b["n"],
                 "mean_predicted": b["mean_predicted"], "fraction_positive": b["fraction_positive"]}
                for b in body["reliability_bins"]
            ],
        })
    return {
        "schema_version": WEB_SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "package_version": __version__,
        "synthetic": synthetic,
        "label": SYNTHETIC_LABEL if synthetic else "user-supplied data run",
        "source_id": metrics["source_id"],
        "mode": metrics["mode"],
        "seed": metrics["seed"],
        "protocol": metrics["protocol"],
        "n": int(metrics["n_rows"]),
        "n_positive": int(metrics["n_positive"]),
        "folds": [{"fold": f["fold"], "n_test": f["n_test"]} for f in metrics["folds"]],
        "models": models,
    }


def validate_web_document(doc: dict[str, Any]) -> None:
    for key in ("schema_version", "synthetic", "label", "source_id", "n", "models"):
        if key not in doc:
            raise ValidationError(f"web document is missing key {key!r}")
    if not doc["models"]:
        raise ValidationError("web document has no models")
    for m in doc["models"]:
        for key in ("name", "n", "auroc", "brier", "ece", "reliability_bins"):
            if key not in m:
                raise ValidationError(f"web model entry is missing {key!r}")
        if sum(int(b["n"]) for b in m["reliability_bins"]) != int(doc["n"]):
            raise ValidationError(f"{m['name']}: bin counts do not sum to n")


def write_web_document(doc: dict[str, Any], path: Path) -> Path:
    validate_web_document(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
