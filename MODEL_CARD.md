# Model card: heart-risk-calibration

## Purpose

Educational benchmark of three tabular classifiers under a leakage-resistant
evaluation protocol. It is not a patient-facing risk service, not a clinical
decision aid, and its outputs are not intended to inform the care of any
person.

## Models

All three are scikit-learn Pipelines: `SimpleImputer(strategy="median")` ->
`StandardScaler()` -> classifier. Imputation and scaling statistics are fitted
on the training rows of each fold only.

| name | classifier | fixed settings | searched (inner CV) |
|---|---|---|---|
| `logistic_regression` | `LogisticRegression` | `solver=lbfgs`, `max_iter=1000` | `C` in {0.1, 1.0, 10.0} |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | `max_iter=60`, `max_bins=32`, `max_leaf_nodes=15`, `min_samples_leaf=10`, `early_stopping=False` | `learning_rate` in {0.05, 0.1} |
| `mlp` | `MLPClassifier` | `hidden_layer_sizes=(16,)`, `solver=lbfgs`, `max_iter=300` | `alpha` in {0.3, 3.0} |

Every classifier receives `random_state=42` (the protocol seed). Settings live
in `configs/model.yaml`; `configs/model_smoke.yaml` holds single-point grids
for the smoke path only.

Why a scikit-learn MLP rather than a torch module: the inputs are 13 numeric
columns and a few hundred rows; a one-hidden-layer network fitted by L-BFGS on
CPU is adequate, composes with `Pipeline` and `GridSearchCV` without adapter
code, is deterministic given the seed, and keeps the dependency set to numpy,
pandas and scikit-learn. The strong `alpha` grid exists because a 16-unit
layer on this many rows otherwise saturates to 0 or 1 outputs.

## Inputs and target

The 13 attribute columns of the UCI Heart Disease Cleveland processed file:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,
ca, thal`, all treated as numeric. `?` cells become NaN and are imputed in
fold. The target is binary: `1` when the source field `num` is greater than
zero, `0` when it equals zero. See `docs/data.md`.

## Evaluation

Nested cross-validation: 5 stratified outer folds, 3 inner folds, seed 42.
Reported per model: AUROC, AUPRC, Brier score, expected calibration error
(10 equal-width bins), reliability bins, a threshold sensitivity table, the
per-fold threshold chosen on inner validation predictions, and percentile
bootstrap intervals over rows (rows are patients). See `docs/evaluation.md`.

No real-data evaluation has been performed in this repository. The only
numbers here come from `make demo` on the synthetic fixture and are labelled
as such in the README.

## Training data

Whatever the user supplies under `DATA_ROOT` (the Cleveland file they obtained
under CC BY 4.0), or the authored synthetic fixture `examples/fixtures/v1`
(300 fictional rows). No real data is shipped.

## Intended use

Learning and teaching about discrimination versus calibration, comparing
baselines under a protocol that keeps preprocessing and selection inside the
training folds, and extending with further models or recalibration steps.

## Out of scope

- Any use in the care, triage, screening or counselling of a person.
- Any claim about the related article's models or results.
- Any population other than the one the user's data describes; the Cleveland
  cohort is small and dated, and its coding conventions are specific to the
  source.
- Fairness or subgroup analysis, which is not implemented.

## Checkpoints

`train` fits each pipeline on all supplied rows (inner grid search over the
full table) and pickles the result with its threshold, chosen on out-of-fold
validation predictions of that table. A checkpoint carries no held-out
evaluation; the nested CV run is the evaluation. Checkpoints are ignored by
git.
