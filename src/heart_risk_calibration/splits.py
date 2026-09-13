"""Outer stratified folds and the inner splitter used for selection.

Rows are patients; each row belongs to exactly one outer test fold. The inner
splitter only ever sees the outer training rows, which is enforced by
`training.fit_fold` passing those rows alone.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from heart_risk_calibration.errors import ConfigError


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.train_idx, self.test_idx).size:
            raise ConfigError(f"fold {self.fold}: train and test rows overlap")
        if self.train_idx.size == 0 or self.test_idx.size == 0:
            raise ConfigError(f"fold {self.fold}: empty train or test set")

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)


def outer_folds(label: np.ndarray, k: int, seed: int) -> tuple[Fold, ...]:
    """Stratified k folds over row positions, shuffled with `seed`."""
    label = np.asarray(label)
    if k < 2:
        raise ConfigError("outer_folds must be >= 2")
    if min(np.bincount(label, minlength=2)) < k:
        raise ConfigError(f"each class needs at least {k} rows for {k} stratified folds")
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    dummy = np.zeros((label.size, 1))
    return tuple(
        Fold(fold=i + 1, train_idx=np.asarray(tr, dtype=int), test_idx=np.asarray(te, dtype=int))
        for i, (tr, te) in enumerate(splitter.split(dummy, label))
    )


def inner_splitter(k: int, seed: int) -> StratifiedKFold:
    if k < 2:
        raise ConfigError("inner_folds must be >= 2")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def splits_document(folds: tuple[Fold, ...], ids: tuple[str, ...], label: np.ndarray,
                    outer_k: int, inner_k: int, seed: int) -> dict[str, Any]:
    """Serialisable record of which patient sits in which outer test fold."""
    return {
        "protocol": "nested_cv",
        "outer_folds": outer_k,
        "inner_folds": inner_k,
        "seed": seed,
        "unit": "row (one row is one patient)",
        "n_rows": int(len(ids)),
        "folds": [
            {
                "fold": f.fold,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "n_test_positive": int(np.asarray(label)[f.test_idx].sum()),
                "test_ids": [ids[i] for i in f.test_idx.tolist()],
            }
            for f in folds
        ],
    }


def splits_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
