"""Write the authored synthetic fixture: examples/fixtures/v1/synthetic_heart.csv,
fixture_meta.json and FIXTURE.md.

Usage: python tools/make_fixtures.py --seed 42 [--rows 300] [--out examples/fixtures/v1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heart_risk_calibration.adapter import FIXTURE_SOURCE_ID  # noqa: E402
from heart_risk_calibration.synthetic import (DEFAULT_ROWS, FixtureSpec, fixture_header,  # noqa: E402
                                              generate, provenance_statement)

FIXTURE_MD = """# Fixture v1

{statement}

## Contents

- `synthetic_heart.csv`: {n_rows} fictional rows. Columns: `patient_id`
  (`SYN-P0001` ...), the 13 attribute columns of the Cleveland processed file
  (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,
  ca, thal`) and `num` in 0..4. The first lines are `#` comments naming the
  source id and this provenance. `?` marks a missing value in {n_missing}
  cells, as in the source file format.
- `fixture_meta.json`: the generator specification, including the planted
  coefficients.

## How the values are made

Each attribute is drawn independently from a simple distribution over the
coded range used by the source file (for example `cp` in 1..4, `thal` in
{{3, 6, 7}}). The positive class is then planted: a logit is formed as a
linear function of `age`, `sex`, `cp == 4`, `thalach`, `oldpeak`, `ca`,
`thal == 7` and `exang` with the coefficients in `fixture_meta.json`, plus
Gaussian noise; the label is a Bernoulli draw from that probability, and
positive rows receive a `num` severity in 1..4. Negative rows have `num` 0.
The distributions and coefficients are chosen for readability; they are not
fitted to any data and carry no information about any patient.

Regenerate with:

```
python tools/make_fixtures.py --seed 42 --out examples/fixtures/v1
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "examples" / "fixtures" / "v1")
    args = parser.parse_args(argv)

    spec = FixtureSpec(seed=args.seed, n_rows=args.rows)
    frame = generate(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "synthetic_heart.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(fixture_header(spec, FIXTURE_SOURCE_ID))
        frame.to_csv(handle, index=False, lineterminator="\n")
    meta = dict(spec.as_dict())
    meta["source_id"] = FIXTURE_SOURCE_ID
    meta["n_positive"] = int((frame["num"].astype(int) > 0).sum())
    n_missing = int((frame == "?").sum().sum())
    meta["n_missing_cells"] = n_missing
    (out / "fixture_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "FIXTURE.md").write_text(
        FIXTURE_MD.format(statement=provenance_statement(spec.seed, spec.version),
                          n_rows=spec.n_rows, n_missing=n_missing),
        encoding="utf-8")
    print(f"wrote {csv_path} ({spec.n_rows} rows, {meta['n_positive']} positive, "
          f"{n_missing} missing cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
