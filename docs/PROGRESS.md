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

---

## Phase 3: Synthetic Failure Generation (Fault Injector)

**Status:** Complete (initial 3 faults + dry-run verified; awaiting user repo SHA verification & small-batch Docker run)

### What Was Done
- Created `configs/repos.yaml` with 10 real public frontend template/starter candidates (Vite+React, Vue, Svelte, Next.js, Angular, CRA, Astro), all marked `verified: false` with `commit: VERIFY_SHA` for user verification.
- Created `configs/fault_injector.yaml` holding all paths, Docker limits (`memory_limit: 4g`, `timeout_sec: 300`, `max_containers: 2`), `seed: 42`, and `enabled_faults`.
- Built `src/collect/fault_injector/faults.py` with an extensible `BaseFault` ABC + `@register_fault` registry and 3 initial end-to-end faults:
  - `remove_dependency` (`dependency_missing`)
  - `wrong_import_path` (`import_path_error`)
  - `ts_type_error` (`type_error`)
- Built `src/collect/fault_injector/docker_runner.py` (`clone_repo`, `verify_baseline`, `run_docker_build` with `CI=true`, memory cap, and timeout).
- Built `src/collect/fault_injector/runner.py` (Typer CLI) with deterministic per-pair seeds, baseline verification, parallelism capped at 2 containers, resumable `manifest.json`, JSONL output (`data/synthetic/synthetic_failures.jsonl`), and `--dry-run`.
- Added `tests/test_fault_injector.py` with unit tests for all 3 faults and a mocked-Docker end-to-end test covering dry-run, JSONL output, and resumability.
- Updated `Makefile` `data` target to run `python -m src.collect.fault_injector.runner --dry-run`.

### Commands to Run
```bash
# 1. Dry-run (no Docker, no cloning)
python -m src.collect.fault_injector.runner --dry-run --limit 6

# 2. Small-batch live generation in WSL2 (after verifying 1+ repo SHA in configs/repos.yaml and Docker is running)
python -m src.collect.fault_injector.runner --repo vite-react-ts-starter --limit 3
```

### Decisions Made
- Implemented 3 faults end-to-end first as requested (`remove_dependency`, `wrong_import_path`, `ts_type_error`), with `FAULT_REGISTRY` ready for the remaining 7 faults (`import_case_mismatch`, `eslint_ci_error`, `lockfile_mismatch`, `wrong_node_version`, `missing_env_var`, `bad_config_key`, `wrong_output_dir`).
- Marked all candidate repos `verified: false` and `commit: VERIFY_SHA` so no unverified commit SHAs are fabricated.

### Open Issues / Next Steps
- Verify commit SHAs and standalone build commands in `configs/repos.yaml` and run a small-batch Docker generation.
- Implement the remaining 7 faults in `src/collect/fault_injector/faults.py` when ready to scale data generation.

---

## Phase 4: Real-World Data Collection Scripts

**Status:** Complete (offline fixture tests passing; no live scraping run)

### What Was Done
- Created `configs/collect.yaml` holding all collection queries, star thresholds, permissive license allowlist, Stack Exchange tags/license, and `data/raw/` output paths.
- Added `src/collect/common.py` defining `RawLogRecord` and append-only `append_raw_records()` so raw JSONL files are never modified after writing.
- Implemented `src/collect/github_actions.py` (Typer CLI) to search public frontend repos, skip repos without permissive licenses, download failed job logs with `GITHUB_TOKEN` from `.env`, cache raw responses under `data/raw/cache/github_actions/`, back off on `403`/`429` rate-limit headers, and resume via manifest.
- Implemented `src/collect/stackexchange.py` (Typer CLI) to stream-parse Stack Exchange `Posts.xml` dumps (`iterparse`), filter for frontend build-error questions with accepted answers, and preserve attribution (`post_url`, `author`, `answer_author`, `CC BY-SA 4.0`).
- Implemented `src/collect/public_datasets.py` (Typer CLI) as an optional loader stub (`BasePublicDatasetLoader`, `LogChunksLoader`).
- Added `tests/test_collect.py` with recorded offline fixtures (zero network calls).

### Commands to Run Yourself (Expected Runtime)
```bash
# 1. GitHub Actions collector (requires GITHUB_TOKEN in .env; ~2-5 min for 20 repos)
python -m src.collect.github_actions

# 2. Stack Exchange dump parser (after downloading Posts.xml; ~1-3 min for subset, ~15-30 min for full SO dump)
python -m src.collect.stackexchange /path/to/Posts.xml --limit 500

# 3. Optional LogChunks loader (seconds for local JSONL export)
python -m src.collect.public_datasets /path/to/logchunks.jsonl
```

### Decisions Made
- Used Python's standard-library `urllib.request` behind an injectable `HttpClient` protocol in `github_actions.py` so both live collection and offline fixture testing work without extra HTTP dependencies.
- Used two-pass `xml.etree.ElementTree.iterparse` in `stackexchange.py` to keep memory usage bounded well within the 12 GB WSL2 cap.

### Open Issues / Next Steps
- Run live collection commands when ready, then proceed to Phase 5 (preprocessing & dataset formatting).

---

## Phase 5: Preprocessing Pipeline (`src/preprocess/`)

**Status:** Complete

### What Was Done
- Created `configs/preprocess.yaml` holding input/output paths, windowing parameters (`context_before: 40`, `context_after: 40`, `tail_lines: 60`, `max_tokens: 1500`, `Qwen/Qwen3-4B-Instruct-2507`), MinHash dedupe settings, and repo-level split ratios.
- Updated `pyproject.toml` with a `preprocess` extra (`detect-secrets`, `datasketch`).
- Implemented composable preprocessing modules:
  - `src/preprocess/clean.py`: ANSI & timestamp stripping, progress-bar/repeated-line collapsing, plus separate signature normalization.
  - `src/preprocess/window.py`: first-error +/-40 + tail 60 windowing with `[... N lines omitted ...]` markers and hard `max_tokens` cap (using cached Qwen tokenizer or deterministic fallback).
  - `src/preprocess/redact.py`: `detect-secrets` integration + regex redaction for tokens, AWS keys, JWTs, emails, IPs, and internal/auth URLs.
  - `src/preprocess/signature.py`: normalized error signature + short hash.
  - `src/preprocess/dedupe.py`: MinHash (`datasketch` or pure-Python fallback) near-duplicate removal + per-signature count cap.
  - `src/preprocess/label.py`: taxonomy regex heuristic pre-labeling for non-synthetic records (`verified=false`) + `unverified_review.csv` export.
  - `src/preprocess/split.py`: deterministic 80/10/10 repo-grouped split with held-out repo support, zero-overlap assertion, and `reports/split_report.md`.
  - `src/preprocess/stats.py`: `reports/data_stats.md` generator (class distribution, token-length histogram, per-source, per-framework).
  - `src/preprocess/run.py`: unified Typer CLI `python -m src.preprocess.run`.
- Added `tests/test_preprocess.py` covering secret redaction, token budget, dedupe caps, zero-overlap repo splitting, and end-to-end execution.

### Commands to Run
```bash
# Run preprocessing pipeline (after collecting raw/synthetic data)
python -m src.preprocess.run
```

### Decisions Made
- Provided deterministic pure-Python fallbacks for `datasketch` and `AutoTokenizer` (local cache) so unit tests and offline environments run cleanly while using the full libraries when installed/cached.

### Open Issues / Next Steps
- Install `.[preprocess]` in WSL2 (`pip install -e ".[preprocess,dev]"`) before running large-scale preprocessing.
- Ready for Phase 6 (training or RAG) when prompted.

---

## Phase 6: Gold Evaluation Set and Dataset Quality Gates

**Status:** Complete

### What Was Done
- Updated `configs/preprocess.yaml` with `gold`, `validation` (`max_class_share: 0.20`), and `chat` (`prompts_path`, `train_chat_jsonl`, `val_chat_jsonl`) settings.
- Implemented `src/preprocess/gold_tool.py` (Typer CLI): stratified sampling across categories prioritizing real (non-synthetic) logs, interactive confirm/edit/skip/quit loop, resumable saving to `data/gold/gold.jsonl` with `verified=true`.
- Implemented `src/preprocess/validate.py` (Typer CLI): hard quality gates checking zero repo overlap, <=20% max class share in train, `DatasetRecord` schema validity, zero leaked secrets, no train IDs in gold set, and `max_tokens` cap.
- Implemented `src/preprocess/to_chat.py` (Typer CLI): converts train/val splits into no-RAG `system/user/assistant` chat SFT JSONL using `configs/prompts.yaml` with strict `ModelOutput` JSON targets.
- Added `data-check` target to `Makefile` running `python -m src.preprocess.validate`.
- Added `tests/test_gold_and_validate.py` testing stratified sampling, resumability, all 6 validation gates, and chat SFT conversion.

### Commands to Run
```bash
# Run quality gates
make data-check

# Build/resume the 300-example gold evaluation set interactively
python -m src.preprocess.gold_tool

# Convert train/val splits to no-RAG chat SFT JSONL
python -m src.preprocess.to_chat
```

### Decisions Made
- Kept `to_chat.py` strictly on `user_no_rag` for Phase 6 as requested; RAG-augmented prompt conversion will be added alongside the RAG pipeline.

### Open Issues / Next Steps
- Ready for Phase 7 (QLoRA fine-tuning or RAG indexing) when prompted.

---

## Phase 7: Evaluation Harness and Zero-Shot Baselines

**Status:** Complete (offline metric & mock-backend tests passing; GPU baselines ready to run)

### What Was Done
- Created `configs/eval.yaml` with `batch_size: 1` settings and zero-shot baseline definitions for `qwen3_4b_instruct` (`Qwen/Qwen3-4B-Instruct-2507`), `qwen25_coder_3b` (`Qwen/Qwen2.5-Coder-3B-Instruct`), `qwen25_3b_instruct` (`Qwen/Qwen2.5-3B-Instruct`), and `llama_server_local`.
- Implemented `src/eval/metrics.py` computing category accuracy, macro-F1, per-class precision/recall/F1/support, JSON validity rate, root-cause entity match, verbatim evidence hallucination rate, confidence calibration + `unknown` when wrong, and latency percentiles.
- Implemented `src/eval/backends.py` with `ModelBackend` ABC and 3 backends: `HF4BitBackend` (4-bit NF4 `BitsAndBytesConfig`, `low_cpu_mem_usage=True`, `batch_size=1`), `LlamaServerBackend` (OpenAI-compatible HTTP + JSON Schema support), and `MockBackend`.
- Implemented `src/eval/run_eval.py` (Typer CLI) saving `reports/*_predictions.jsonl` and `reports/*_metrics.json` and printing a markdown summary table.
- Updated `Makefile` (`eval` target) and added `tests/test_eval.py`.

### Commands to Run Baselines (in WSL2 RTX 5050)
```bash
# Run unit tests first
python -m pytest -v

# Zero-shot baselines on the gold set (batch_size=1, 4-bit NF4)
python -m src.eval.run_eval --model qwen3_4b_instruct --split gold
python -m src.eval.run_eval --model qwen25_coder_3b --split gold
python -m src.eval.run_eval --model qwen25_3b_instruct --split gold
```

### Decisions Made
- Implemented all classification/F1/entity/hallucination/calibration metrics in pure Python to avoid extra heavy evaluation dependencies.

### Open Issues / Next Steps
- Run the 3 zero-shot baselines in WSL2 after populating `data/gold/gold.jsonl`, then proceed to Phase 8 (QLoRA fine-tuning or RAG) when prompted.
