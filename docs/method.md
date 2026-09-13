# Method

## Pipelines

Every model is a scikit-learn `Pipeline`:

```
SimpleImputer(strategy="median") -> StandardScaler() -> classifier
```

`Pipeline.fit` receives only the training rows of a fold, so the median used
for imputation and the mean and scale used for standardisation are computed
from those rows alone. Test rows pass through `transform` with the fitted
statistics. Inputs stay a 2D matrix of rows by 13 attributes; no model
reshapes them.

| name | classifier | notes |
|---|---|---|
| `logistic_regression` | `LogisticRegression(solver="lbfgs", max_iter=1000)` | `C` searched |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier(max_iter=60, max_bins=32, max_leaf_nodes=15, min_samples_leaf=10, early_stopping=False)` | `learning_rate` searched; early stopping is off so that no internal validation split is carved out of the training rows behind the protocol's back |
| `mlp` | `MLPClassifier(hidden_layer_sizes=(16,), solver="lbfgs", max_iter=300)` | `alpha` searched |

`max_bins=32` for the boosting model: the 13 columns hold few distinct values
each and the tables are a few hundred rows, so 255 bins would add binning
cost without changing the trees.

### Why `MLPClassifier` and not torch

The spec allowed either. scikit-learn was chosen because the inputs are 13
numeric columns and a few hundred rows: a one-hidden-layer network fitted by
L-BFGS on CPU is adequate for that size; it composes with `Pipeline` and
`GridSearchCV` without adapter code, so it is subject to exactly the same
in-fold preprocessing and selection as the other two models; it is
deterministic given `random_state`; and it keeps the dependency set to numpy,
pandas and scikit-learn, which keeps CI fast and the install small. The L2
penalty grid is deliberately strong because a 16-unit layer on this many rows
otherwise saturates to 0 or 1 outputs and calibration becomes meaningless.

## Threading

Fits run under `threadpoolctl.threadpool_limits(limits=1)`. On tables of this
size the native OpenMP and BLAS pools oversubscribe and make each boosting
fit slower by an order of magnitude; single-threaded fits are faster and
make timings comparable across machines. `threadpoolctl` is a dependency of
scikit-learn and is not listed separately.

## Per-fold procedure (`training.fit_rows`)

Given the outer-training row indices of one fold:

1. Build the pipeline for the model with the protocol seed.
2. If the grid has more than one candidate, run `GridSearchCV` with the inner
   `StratifiedKFold(3, shuffle=True, random_state=seed)` splitter and
   `scoring="neg_log_loss"`, refit the best candidate on all outer-training
   rows. If the grid has exactly one candidate (the smoke configuration),
   fit that pipeline directly; the search would only refit the same object.
3. Obtain inner out-of-fold validation predictions with `cross_val_predict`
   on the same inner splitter and a clone of the refitted pipeline.
4. Choose the decision threshold on those validation predictions only, by
   Youden's index (sensitivity + specificity - 1) over the grid 0.05 to 0.95
   in steps of 0.05; ties resolve to the lowest threshold. An `f1` rule is
   available in `configs/eval.yaml`.
5. Return the fitted pipeline, the chosen threshold, the best parameters, the
   inner best score, and `threshold_source =
   inner_out_of_fold_validation_predictions`.

Outer-test rows are never passed to this function; the driver scores them
afterwards, once. `test_threshold_never_uses_test_labels` flips the
outer-test labels and asserts the threshold and the fitted model are
unchanged.

Negative log loss was chosen as the inner criterion because the protocol
reports calibration, and log loss rewards probability quality rather than
ranking alone.

## Final fit (`train`)

`train` calls the same `fit_rows` with every row as training row and pickles
the result as a checkpoint with its threshold, best parameters, row count,
source id, mode and the expected column order. It exists so that a fitted
object can be inspected or reused; it carries no evaluation of its own.

## Settings taken from the source description

None. No configuration value in this repository is copied from the related
article; all settings are the ones listed above and in `configs/`.
