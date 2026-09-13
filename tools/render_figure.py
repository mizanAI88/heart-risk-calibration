"""Draw docs/figures/protocol.svg by hand: the nested cross-validation layout,
what each stage may see, and where the threshold and the metrics come from.

Usage: python tools/render_figure.py [--out docs/figures/protocol.svg]
White background, black labels, no external assets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 560
FONT = "font-family='Helvetica, Arial, sans-serif'"
TEST_FILL = "#bfbfbf"
TRAIN_FILL = "#ffffff"
VAL_FILL = "#e6e6e6"


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#000") -> str:
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}' stroke-width='1'/>"


def text(x: float, y: float, s: str, size: int = 13, anchor: str = "start", weight: str = "normal") -> str:
    return (f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' "
            f"font-weight='{weight}' fill='#000' {FONT}>{s}</text>")


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#000' stroke-width='1.2' "
            "marker-end='url(#head)'/>")


def outer_block(x0: float, y0: float, n_folds: int = 5, cell_w: float = 70, cell_h: float = 22) -> list[str]:
    parts: list[str] = []
    for fold in range(n_folds):
        y = y0 + fold * (cell_h + 8)
        parts.append(text(x0 - 10, y + 16, f"outer fold {fold + 1}", 12, "end"))
        for cell in range(n_folds):
            fill = TEST_FILL if cell == fold else TRAIN_FILL
            parts.append(rect(x0 + cell * cell_w, y, cell_w, cell_h, fill))
            label = "test" if cell == fold else "train"
            parts.append(text(x0 + cell * cell_w + cell_w / 2, y + 15, label, 11, "middle"))
    return parts


def inner_block(x0: float, y0: float, n_inner: int = 3, cell_w: float = 90, cell_h: float = 22) -> list[str]:
    parts: list[str] = []
    for fold in range(n_inner):
        y = y0 + fold * (cell_h + 8)
        parts.append(text(x0 - 10, y + 16, f"inner {fold + 1}", 12, "end"))
        for cell in range(n_inner):
            fill = VAL_FILL if cell == fold else TRAIN_FILL
            parts.append(rect(x0 + cell * cell_w, y, cell_w, cell_h, fill))
            label = "validation" if cell == fold else "fit"
            parts.append(text(x0 + cell * cell_w + cell_w / 2, y + 15, label, 11, "middle"))
    return parts


def build_svg() -> str:
    p: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{WIDTH}' height='{HEIGHT}' viewBox='0 0 {WIDTH} {HEIGHT}'>",
        "<defs><marker id='head' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'>"
        "<path d='M0,0 L6,3 L0,6 z' fill='#000'/></marker></defs>",
        rect(0, 0, WIDTH, HEIGHT, "#ffffff", "#ffffff"),
        text(WIDTH / 2, 28, "heart-risk-calibration: nested cross-validation protocol", 17, "middle", "bold"),
        text(WIDTH / 2, 48, "rows are patients; every row is an outer test row exactly once", 12, "middle"),
    ]
    # Outer folds (left)
    p.append(text(120, 84, "Outer: 5 stratified folds, seed 42", 13, "start", "bold"))
    p.extend(outer_block(120, 96))
    p.append(text(120, 262, "Per outer fold, using outer-train rows only:", 12))
    p.append(text(120, 280, "1. imputer (median) and scaler fitted", 12))
    p.append(text(120, 296, "2. inner 3-fold grid search (neg log loss)", 12))
    p.append(text(120, 312, "3. refit best pipeline on all outer-train rows", 12))
    p.append(text(120, 328, "4. threshold from inner out-of-fold validation predictions", 12))
    p.append(text(120, 344, "5. outer-test rows scored once", 12))
    # Arrow to inner block
    p.append(arrow(480, 150, 560, 150))
    p.append(text(520, 140, "outer-train rows", 11, "middle"))
    # Inner folds (right)
    p.append(text(620, 84, "Inner: 3 stratified folds of the outer-train rows", 13, "start", "bold"))
    p.extend(inner_block(620, 96))
    p.append(text(620, 200, "grid candidates: logistic C; boosting learning rate; MLP alpha", 11))
    p.append(text(620, 216, "validation predictions -> Youden threshold (never test labels)", 11))
    # Legend
    lx, ly = 620, 262
    p.append(rect(lx, ly, 18, 14, TRAIN_FILL)); p.append(text(lx + 26, ly + 12, "fit / train rows", 11))
    p.append(rect(lx, ly + 22, 18, 14, VAL_FILL)); p.append(text(lx + 26, ly + 34, "inner validation rows", 11))
    p.append(rect(lx, ly + 44, 18, 14, TEST_FILL)); p.append(text(lx + 26, ly + 56, "outer test rows", 11))
    # Outputs strip
    y = 390
    p.append(rect(60, y, 840, 130, "#ffffff"))
    p.append(text(80, y + 24, "Pooled outer-test predictions (n = all rows, one per patient)", 13, "start", "bold"))
    p.append(text(80, y + 48, "AUROC, AUPRC, Brier, ECE (10 equal-width bins)", 12))
    p.append(text(80, y + 66, "reliability bins: n, mean predicted, fraction positive", 12))
    p.append(text(80, y + 84, "threshold sensitivity table over a fixed grid 0.05 .. 0.95", 12))
    p.append(text(80, y + 102, "percentile bootstrap intervals over rows (patients), seeded", 12))
    p.append(text(520, y + 48, "per-fold: n_train, n_test, chosen threshold, best params", 12))
    p.append(text(520, y + 66, "metrics.json, predictions.csv, splits.json, manifest.yaml", 12))
    p.append(text(520, y + 84, "mode smoke | demo | experiment recorded in every file", 12))
    p.append(text(520, y + 102, "synthetic sources cannot run in experiment mode", 12))
    p.append(text(WIDTH / 2, HEIGHT - 12, "drawn by tools/render_figure.py", 10, "middle"))
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures" / "protocol.svg")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
