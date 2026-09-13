"""argparse entry point: python -m heart_risk_calibration <subcommand>."""
from __future__ import annotations

import argparse
from pathlib import Path

from heart_risk_calibration import __version__, commands
from heart_risk_calibration.manifest import MODES
from heart_risk_calibration.models import MODEL_NAMES
from heart_risk_calibration.paths import (DEFAULT_OUT_DIR, DEMO_OUT_DIR, EVAL_CONFIG,
                                          WEB_FIXTURE_JSON)


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", type=Path, default=None,
                   help="fixture-format CSV (header row, `?` for missing); default is the data root")
    p.add_argument("--data-root", default=None,
                   help="directory holding processed.cleveland.data (or set DATA_ROOT)")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heart_risk_calibration",
        description=("Discrimination and calibration comparison of three tabular classifiers "
                     "under nested cross-validation. Educational benchmark code; not a "
                     "patient-facing service."),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit each model on all supplied rows and save checkpoints")
    _add_source_args(p)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "train")
    p.add_argument("--mode", choices=MODES, default="experiment")

    p = sub.add_parser("evaluate", help="nested cross-validation: metrics.json, predictions.csv, manifest")
    _add_source_args(p)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "evaluate")
    p.add_argument("--mode", choices=MODES, default="experiment")
    p.add_argument("--eval-config", type=Path, default=EVAL_CONFIG)

    p = sub.add_parser("calibration-table", help="print reliability bins and threshold table of a run")
    p.add_argument("--run-dir", type=Path, default=DEMO_OUT_DIR, help="directory holding metrics.json")
    p.add_argument("--model", choices=MODEL_NAMES, default=None)
    p.add_argument("--csv", type=Path, default=None, help="also write the bins as CSV")

    p = sub.add_parser("smoke", help="offline check on the synthetic fixture; validates outputs")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "smoke")

    p = sub.add_parser("demo", help="documented demonstration on the synthetic fixture")
    p.add_argument("--out", type=Path, default=DEMO_OUT_DIR)

    p = sub.add_parser("export-web", help="JSON with per-model AUROC, Brier, ECE, bins and n")
    p.add_argument("--run-dir", type=Path, default=DEMO_OUT_DIR)
    p.add_argument("--out", type=Path, default=WEB_FIXTURE_JSON)

    p = sub.add_parser("download", help="print the data licence; fetch the file only with --accept-terms")
    p.add_argument("--data-root", default=None)
    p.add_argument("--accept-terms", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = tuple(args.models) if getattr(args, "models", None) else None
    if args.command == "train":
        return commands.cmd_train(args.input, args.data_root, args.out, args.mode, models)
    if args.command == "evaluate":
        return commands.cmd_evaluate(args.input, args.data_root, args.out, args.mode,
                                     args.eval_config, models)
    if args.command == "calibration-table":
        return commands.cmd_calibration_table(args.run_dir, args.model, args.csv)
    if args.command == "smoke":
        return commands.cmd_smoke(args.out)
    if args.command == "demo":
        return commands.cmd_demo(args.out)
    if args.command == "export-web":
        return commands.cmd_export_web(args.run_dir, args.out)
    if args.command == "download":
        return commands.cmd_download(args.data_root, args.accept_terms)
    raise SystemExit(f"unknown command {args.command!r}")
