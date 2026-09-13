PYTHON ?= python
DATA_ROOT ?=
OUT ?= out/evaluate
MODE ?= experiment

.PHONY: setup smoke demo test evaluate train calibration-table export-web download fixtures figure clean

setup:
	pip install -e . && pip install pytest

smoke:
	$(PYTHON) -m heart_risk_calibration smoke

demo:
	$(PYTHON) -m heart_risk_calibration demo
	$(PYTHON) -m heart_risk_calibration export-web --out examples/output/web_fixture.json

test:
	$(PYTHON) -m pytest -q

# Usage: make evaluate DATA_ROOT=/path/to/dir OUT=out/run1
# A missing data root or file exits 2 with a message naming what to obtain.
evaluate:
	$(PYTHON) -m heart_risk_calibration evaluate --data-root "$(DATA_ROOT)" --out $(OUT) --mode $(MODE)

# Usage: make train DATA_ROOT=/path/to/dir OUT=out/train
train:
	$(PYTHON) -m heart_risk_calibration train --data-root "$(DATA_ROOT)" --out $(OUT) --mode $(MODE)

calibration-table:
	$(PYTHON) -m heart_risk_calibration calibration-table --run-dir examples/output/demo

export-web:
	$(PYTHON) -m heart_risk_calibration export-web --out examples/output/web_fixture.json

# Prints the licence and terms. Fetches only when ACCEPT_TERMS=1. Never run in CI.
download:
	$(PYTHON) -m heart_risk_calibration download --data-root "$(DATA_ROOT)" $(if $(ACCEPT_TERMS),--accept-terms,)

fixtures:
	$(PYTHON) tools/make_fixtures.py --seed 42 --out examples/fixtures/v1

figure:
	$(PYTHON) tools/render_figure.py --out docs/figures/protocol.svg

clean:
	rm -rf out .pytest_cache build dist src/*.egg-info
