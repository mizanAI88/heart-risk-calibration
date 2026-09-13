"""The fixture generator: determinism, labelling, planted signal."""
from __future__ import annotations

import json

import numpy as np
import pytest

from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.paths import FIXTURE_CSV, FIXTURE_DIR
from heart_risk_calibration.schema import standardise
from heart_risk_calibration.synthetic import (PLANTED_COEFFICIENTS, FixtureSpec, generate,
                                              planted_logit, provenance_statement)


def test_generator_is_deterministic_for_a_seed_and_differs_across_seeds():
    a = generate(FixtureSpec(seed=3, n_rows=50))
    assert a.equals(generate(FixtureSpec(seed=3, n_rows=50)))
    assert not a.equals(generate(FixtureSpec(seed=4, n_rows=50)))


def test_generator_shape_ids_and_missing_cells():
    frame = generate(FixtureSpec(seed=42, n_rows=300))
    assert frame.shape == (300, 15)
    assert frame["patient_id"].iloc[0] == "SYN-P0001"
    assert frame["patient_id"].is_unique
    assert int((frame == "?").sum().sum()) == 6
    assert set(frame["num"].astype(int)) <= {0, 1, 2, 3, 4}


def test_generator_refuses_tiny_tables():
    with pytest.raises(ValueError):
        generate(FixtureSpec(seed=1, n_rows=5))


def test_planted_signal_separates_classes(data_cfg: DataConfig):
    """The planted logit must be higher for positive rows on average; otherwise
    the demonstration would be learning noise."""
    frame = generate(FixtureSpec(seed=42, n_rows=300))
    table = standardise(frame, data_cfg)
    filled = table.features.fillna(table.features.median())
    logit = planted_logit(filled)
    assert logit[table.label == 1].mean() > logit[table.label == 0].mean() + 0.5
    assert PLANTED_COEFFICIENTS["ca_per_vessel"] > 0


def test_committed_fixture_carries_provenance_header_and_sidecars():
    header = FIXTURE_CSV.read_text(encoding="utf-8").splitlines()[:4]
    assert any(provenance_statement(42) in line for line in header)
    assert (FIXTURE_DIR / "FIXTURE.md").is_file()
    meta = json.loads((FIXTURE_DIR / "fixture_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42 and meta["n_rows"] == 300 and meta["fixture_version"] == "v1"
    assert meta["planted_coefficients"] == PLANTED_COEFFICIENTS


def test_committed_fixture_matches_regeneration():
    """Byte-level reproducibility of the committed fixture from the generator."""
    regenerated = generate(FixtureSpec(seed=42, n_rows=300))
    import pandas as pd

    on_disk = pd.read_csv(FIXTURE_CSV, comment="#", dtype=str)
    assert on_disk.equals(regenerated.astype(str))
    assert np.array_equal(on_disk.columns, regenerated.columns)
