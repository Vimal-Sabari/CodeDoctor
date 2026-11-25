PYTHON ?= python3
PIP ?= pip

.PHONY: setup test lint data train eval serve

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

data:
	@echo "Data pipeline not yet implemented (scheduled for Phase 2)."

train:
	@echo "Training pipeline not yet implemented (scheduled for Phase 4)."

eval:
	@echo "Evaluation pipeline not yet implemented (scheduled for Phase 5)."

serve:
	@echo "Serving pipeline not yet implemented (scheduled for Phase 6)."
