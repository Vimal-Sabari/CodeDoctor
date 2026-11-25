# LaunchForge AI — Progress Log

## Phase 1: Project Scaffold and Environment Verification

**Status:** Complete (awaiting user verification on WSL2 RTX 5050)

### What Was Done
- Created the repository folder structure:
  - `configs/`, `data/{raw,synthetic,processed,gold}/`, `knowledge_base/`, `models/{adapters,merged,gguf}/`, `reports/` (with `.gitkeep` placeholders).
  - `src/{collect,preprocess,train,rag,eval,serve}/` Python package hierarchy.
  - `scripts/`, `tests/`, `docs/`.
- Added `pyproject.toml` with Python `>=3.11`, base runtime dependencies (`typer`, `pyyaml`, `pydantic`, `python-dotenv`, `psutil`), and optional dependency groups (`train`, `rag`, `dev`, `all`).
- Added `.gitignore` (excluding data, model weights, checkpoints, `.env`, `reports/`, `llama.cpp/`, and vector store DBs while preserving `.gitkeep` files), `.env.example`, and `Makefile` (`setup`, `test`, `lint`, `data`, `train`, `eval`, `serve`).
- Implemented `scripts/check_env.py` (Typer CLI) to verify Python >= 3.11, system RAM, PyTorch + CUDA build, GPU name, compute capability `12.0` (`sm_120`), free VRAM, `bitsandbytes` 4-bit NF4 CUDA forward pass, and `unsloth` import, exiting non-zero with targeted fix hints on failure.
- Wrote `docs/SETUP.md` with exact `.wslconfig`, venv, PyTorch `cu128`, dependency install, and `llama.cpp` CUDA (`sm_120`) build commands.
- Added `tests/test_smoke.py` verifying package imports and `scripts/check_env.py` CLI `--help`.

### Commands to Run
```bash
# In WSL2 Ubuntu with .venv activated (after following docs/SETUP.md):
make setup && python scripts/check_env.py && make test
```

### Decisions Made
- Used `pyproject.toml` with optional dependency extras (`[train]`, `[rag]`, `[dev]`, `[all]`) instead of multiple `requirements-*.txt` files so `pip install -e ".[dev]"` stays clean and PyTorch `cu128` can be installed separately before GPU packages.
- Preserved empty directory structure in Git via `.gitkeep` while ignoring all generated/downloaded files inside `data/`, `knowledge_base/`, `models/`, and `reports/`.

### Open Issues / Next Steps
- User needs to run `make setup && python scripts/check_env.py && make test` in WSL2 Ubuntu and report the output (especially `bitsandbytes` 4-bit NF4 and `unsloth` compatibility on `sm_120` + `cu128`).
- Phase 2 (Data Collection) will begin after Phase 1 verification.

---

## Phase 2: Label Taxonomy, Data Schema, and Output Contract

**Status:** Complete

### What Was Done
- Created `configs/taxonomy.yaml` with 18 frontend build failure categories (17 specific + `unknown`), each defining `id`, `description`, `typical_log_signatures` (regex hints), and `example_fix`.
- Implemented `src/schema.py` with `ErrorCategory` (`StrEnum`), `TaxonomyEntry`, `DatasetRecord`, and `ModelOutput` Pydantic v2 models, `load_taxonomy()`, `build_model_output_schema()`, `export_model_output_schema()`, and a Typer CLI (`export-schema`, `validate`).
- Generated `schema.json` at the repo root from `ModelOutput` with the `ErrorCategory` enum for `llama.cpp` grammar-constrained decoding.
- Created `configs/prompts.yaml` containing `system`, `user_no_rag`, and `user_with_rag` templates, instructing the model to return `unknown` with low confidence when evidence is insufficient and to quote only verbatim log lines.
- Added `tests/test_schema.py` covering taxonomy count/enum consistency, regex compilation, `schema.json` sync, `DatasetRecord`/`ModelOutput` validation, and `prompts.yaml` rendering.

### Commands to Run
```bash
# Run schema and smoke tests
python -m pytest -v

# Regenerate or validate schema via CLI (in WSL2 .venv)
python -m src.schema export-schema
python -m src.schema --help
```

### Decisions Made
- Kept `ErrorCategory` in `src/schema.py` strictly synchronized with `configs/taxonomy.yaml` and enforced 1-to-1 equality (including ordering) in `tests/test_schema.py`.
- Lazy-imported `typer` inside `get_cli_app()` in `src/schema.py` so schema validation imports work even without CLI dependencies installed.

### Open Issues / Next Steps
- Awaiting Phase 3 prompt (data collection / synthetic generation).
