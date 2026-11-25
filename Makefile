PYTHON ?= python3
PIP ?= pip

.PHONY: setup test lint data data-check train eval serve

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

data:
	$(PYTHON) -m src.collect.fault_injector.runner --dry-run

data-check:
	$(PYTHON) -m src.preprocess.validate

train:
	@echo "Training pipeline not yet implemented (scheduled for Phase 4)."

eval:
	@echo "Evaluation pipeline not yet implemented (scheduled for Phase 5)."

serve:
	@echo "Serving pipeline not yet implemented (scheduled for Phase 6)."
