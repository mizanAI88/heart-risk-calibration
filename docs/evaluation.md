# Evaluation

## Protocol: nested cross-validation

Of the two options the spec allowed (nested cross-validation, or a seeded
train/validation/test split), this repository implements nested
cross-validation. With a few hundred rows a single held-out test set would be
too small to say anything with useful precision; nested cross-validation
lets every row serve as a test row exactly once while keeping selection
inside the training folds.

Configured in `configs/eval.yaml`:

| parameter | value | meaning |
|---|---|---|
| `outer_folds` | 5 | `StratifiedKFold(5, shuffle=True, random_state=42)` over rows |
| `inner_folds` | 3 | `StratifiedKFold(3, shuffle=True, random_state=42)` over the outer-training rows |
| `seed` | 42 | protocol seed; also every classifier's `random_state` |
| `inner_scoring` | `neg_log_loss` | grid search criterion |
| `threshold_rule` | `youden` | applied to inner out-of-fold validation predictions |
| `threshold_grid` | 0.05 to 0.95 step 0.05 | candidate thresholds and the sensitivity table grid |
| `n_bins` | 10 | equal-width reliability bins on [0, 1] |
| `bootstrap_resamples` | 200 | percentile intervals over rows |

For each model and each outer fold: fit imputer, scaler and classifier on
the outer-training rows with inner grid search; choose the threshold on the
inner out-of-fold validation predictions of those rows; score the outer-test
rows once. Rows are patients, one row per patient, and a row belongs to
exactly one outer test fold. `docs/figures/protocol.svg` (drawn by
`tools/render_figure.py`) shows the layout.

## Where the validation set is

Under nested cross-validation there is no single validation set. Each outer
fold has its own inner validation predictions, obtained by `cross_val_predict`
with the inner splitter on the outer-training rows. Those predictions choose
the threshold and never include an outer-test row. The manifest's
`sample_counts` therefore records the fold 1 sizes (train 240, one inner
validation fold 80, test 60 on the fixture) with a note, and the full per-fold
table under `cv.per_fold`.

## Metrics

Per model, on the pooled outer-test predictions (n = all rows):

- `auroc`, `auprc` (average precision), `brier`, `ece`
- `ci`: percentile bootstrap intervals (alpha 0.05) over rows for the four
  metrics, with the seed and the number of resamples actually used (a
  resample missing one class is skipped for AUROC and AUPRC and counted)
- `reliability_bins`: 10 bins with `n`, `mean_predicted`, `fraction_positive`
- `threshold_sensitivity`: sensitivity, specificity, PPV, NPV, flagged count,
  and the confusion counts at every grid threshold
- `at_fold_thresholds`: pooled sensitivity, specificity and flagged count when
  each row is classified with its own fold's chosen threshold

Per fold: `n_train`, `n_test`, `n_test_positive`, `best_params`,
`inner_best_score`, `threshold`, `threshold_source`,
`n_validation_predictions`, the four metrics on that fold's test rows, and
sensitivity and specificity at the fold's threshold.

ECE is the count-weighted mean over bins of the absolute difference between
mean predicted probability and observed positive fraction. A predicted
probability of exactly 1 falls in the last bin.

Metrics are pooled over folds rather than averaged because the fold sizes
are small and the reliability bins would otherwise be too sparse to read.
Per-fold values are kept so that the spread is visible.

## Outputs of `evaluate`, `demo` and `smoke`

| file | content |
|---|---|
| `metrics.json` | the document above plus protocol settings, load report and fold sizes |
| `predictions.csv` | `patient_id, fold, model, y_true, p_hat, threshold, y_pred` for every row and model |
| `splits.json` | which patient sits in which outer test fold; its sha256 is `split_manifest_hash` in the manifest |
| `manifest.yaml` | contract section 5 manifest with `mode`, hashes, sample counts and `cv.per_fold` |

`validate_metrics` checks the document shape, that fold sizes sum to the row
count, that every fold and bin count is present, that the four metrics lie in
[0, 1], and that reliability bin counts sum to the row count. `smoke` reloads
`metrics.json` from disk and runs this check; a mismatch exits 1.

## Smoke versus demo

`smoke` uses `configs/eval_smoke.yaml` and `configs/model_smoke.yaml`: the
same 5 x 3 nested protocol with single-point grids and 25 bootstrap
resamples, so that the offline path stays short. `demo` uses the full
configuration. Neither produces a benchmark result; both run on the
synthetic fixture and are recorded with `mode: smoke` or `mode: demo`.

## Modes

`--mode experiment` is refused for any synthetic source. Runs on the user's
Cleveland file should use `experiment` so that the manifest distinguishes
them from fixture runs.

## Local test run (2026-09-13)

```
python -m pytest -q
60 passed in 23.33s
```

Run on Python 3.10 with numpy 1.26, pandas 2.3, scikit-learn 1.7. The demo
and smoke commands were also run and their outputs committed under
`examples/output/`.
