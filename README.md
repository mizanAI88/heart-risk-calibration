# heart-risk-calibration

Given a small structured clinical table, how well separated are the two
classes, and how far do a model's predicted probabilities sit from the
observed frequencies? This package compares logistic regression, histogram
gradient boosting and a small multilayer perceptron on the 13 attribute
columns of the UCI Heart Disease Cleveland file under nested
cross-validation, and reports AUROC, AUPRC, Brier score, expected calibration
error, 10-bin reliability tables, a threshold sensitivity table and bootstrap
intervals over patients.

**Status:** `demo_ready` - runs end to end on authored synthetic fixtures. No
real-data evaluation has been performed in this repository.

## Relationship to research

`coauthored_research`. New companion tabular benchmark for the coauthored article *Hybrid Deep
Learning Framework for Enhanced Heart Disease Prediction: Integrating XGBoost
and Capsule Networks with CNN-Transformer Architectures* (Journal of Computer
Science and Technology Studies 3(2), 116-123, 2021, DOI
10.32996/jcsts.2021.3.2.9), on which MD Mizanur Rahman is a coauthor. This
repository is not the article's code and reproduces none of its results.

The article's model family (XGBoost, capsule networks, CNN-Transformer) is not
implemented here. This repository is a plain tabular baseline suite with an
evaluation protocol written to be leakage-resistant; it does not reshape the
13 columns into a 2D input for any model.

## What is implemented

- **Data adapter** (`adapter.py`): reads `DATA_ROOT/processed.cleveland.data`
  (14 comma-separated columns, no header). The `?` marker becomes NaN at load;
  the NaNs are imputed inside each cross-validation fold by the model
  pipeline. The row count actually read is reported next to the count the
  source documents. A missing root or file raises an error naming what to
  obtain and from where.
- **Target mapping** (`schema.py`): the source field `num` takes values 0 to 4;
  the binary label is `1` when `num > 0` and `0` otherwise. Values outside
  0 to 4, or a missing `num`, are rejected.
- **Download gate** (`download.py`): `download` prints the dataset licence
  (CC BY 4.0) and the file URL, then exits 2. Only `--accept-terms` fetches,
  and never when `CI` or `GITHUB_ACTIONS` is set.
- **Synthetic fixture** (`tools/make_fixtures.py --seed 42`): 300 fictional rows
  with the same 13 columns, a `num` field in 0 to 4, six planted `?` cells,
  and a planted linear signal documented in `examples/fixtures/v1/FIXTURE.md`.
- **Models** (`models.py`): three scikit-learn Pipelines, each
  `SimpleImputer(median) -> StandardScaler -> classifier`, with
  `LogisticRegression`, `HistGradientBoostingClassifier` and `MLPClassifier`
  (one hidden layer of 16 units, L-BFGS). The MLP is scikit-learn rather than
  torch because the inputs are 13 numeric columns and a few hundred rows, the
  estimator composes with `Pipeline` and `GridSearchCV` directly, it is
  deterministic given the seed, and it keeps the dependency set small.
- **Protocol** (`splits.py`, `training.py`, `evaluate.py`): nested
  cross-validation, 5 stratified outer folds and 3 inner folds, seed 42.
  Imputer, scaler and classifier are fitted on outer-training rows only;
  hyperparameters are selected by inner grid search on negative log loss; the
  decision threshold is chosen on inner out-of-fold validation predictions
  by Youden's index and never sees outer-test labels; outer-test rows are
  scored once.
- **Metrics** (`metrics.py`): AUROC, AUPRC, Brier, ECE with 10 equal-width bins,
  reliability bins, a threshold sensitivity table over 0.05 to 0.95, and
  percentile bootstrap intervals over rows. Rows are patients, so the
  bootstrap unit is the patient.
- **Run manifest** (`manifest.py`): schema version, mode (`smoke`, `demo`,
  `experiment`), computed hashes, sample counts and the per-fold row counts.
  Synthetic sources cannot run in `experiment` mode.
- **CLI**: `train`, `evaluate`, `calibration-table`, `smoke`, `demo`,
  `export-web`, `download`.

## Quickstart (offline, CPU, < 1 minute)

```
pip install -e .            # or: pip install -r requirements.txt
make smoke                  # nested CV on the fixture with single-point grids; validates metrics.json
make test                   # python -m pytest -q
make demo                   # writes examples/output/demo/ and web_fixture.json
```

Without installing, prefix commands with `PYTHONPATH=src`. On the Cleveland
file you obtained yourself:

```
python -m heart_risk_calibration download --data-root /path/to/data              # prints the licence, exits 2
python -m heart_risk_calibration download --data-root /path/to/data --accept-terms
python -m heart_risk_calibration evaluate --data-root /path/to/data --out out/run1 --mode experiment
python -m heart_risk_calibration calibration-table --run-dir out/run1
python -m heart_risk_calibration train --data-root /path/to/data --out out/train --mode experiment
```

## Demonstration output

`make demo` runs the full protocol on the synthetic fixture and writes
`examples/output/demo/{metrics.json, predictions.csv, splits.json, manifest.yaml}`
and `examples/output/web_fixture.json`. The values below were computed by
`make demo` on the authored synthetic fixture `examples/fixtures/v1`, seed 42,
not a benchmark result; they describe a fictional table with a planted
signal and say nothing about any real dataset or model.

| model | n | AUROC [bootstrap 95] | AUPRC | Brier [bootstrap 95] | ECE [bootstrap 95] |
|---|---|---|---|---|---|
| logistic_regression | 300 | 0.682 [0.627, 0.737] | 0.731 | 0.223 [0.207, 0.240] | 0.050 [0.038, 0.115] |
| hist_gradient_boosting | 300 | 0.706 [0.642, 0.767] | 0.677 | 0.225 [0.197, 0.253] | 0.091 [0.073, 0.153] |
| mlp | 300 | 0.651 [0.597, 0.715] | 0.683 | 0.256 [0.224, 0.285] | 0.143 [0.100, 0.204] |

Each outer test fold holds 60 rows. Intervals are percentile bootstrap over
rows (patients), 200 resamples. The printed demo also shows the 10-bin
reliability table and the threshold sensitivity table for each model.

## Data access

See `docs/data.md`. The UCI Heart Disease data (CC BY 4.0) must be obtained by
the user; nothing is redistributed here. Place `processed.cleveland.data`
under a directory and pass it as `--data-root` or `DATA_ROOT`.

## Evaluation protocol

See `docs/evaluation.md` and `docs/figures/protocol.svg`. In one sentence:
nested cross-validation (5 outer stratified folds, 3 inner folds, seed 42),
preprocessing and selection inside the outer-training rows, threshold from
inner validation predictions, outer-test rows scored once, metrics pooled
over all rows with bootstrap intervals over patients.

## Limitations

- No real-data result exists in this repository. The demo numbers describe a
  synthetic table.
- The Cleveland file is small (303 rows as documented by the source, 6 of them
  with `?` cells); intervals on it will be wide and calibration bins sparse.
- The pipelines are baselines with small grids; no feature engineering, no
  class reweighting, no post-hoc recalibration.
- Nested cross-validation reports a protocol-level estimate, not a single
  deployable model. `train` fits on all rows for convenience and its
  checkpoint carries no held-out evaluation of its own.
- This is an educational benchmark. It is not a patient-facing risk service
  and its outputs are not intended to inform care.

## Related work

- Hybrid Deep Learning Framework for Enhanced Heart Disease Prediction:
  Integrating XGBoost and Capsule Networks with CNN-Transformer Architectures.
  Journal of Computer Science and Technology Studies 3(2), 116-123, 2021.
  DOI 10.32996/jcsts.2021.3.2.9.
- UCI Machine Learning Repository, Heart Disease data set,
  https://archive.ics.uci.edu/dataset/45/heart+disease (CC BY 4.0).

## Contribution and provenance

See `NOTICE.md` and `CONTRIBUTING.md`.

## License

MIT, copyright 2026 MD Mizanur Rahman. See `LICENSE`.
