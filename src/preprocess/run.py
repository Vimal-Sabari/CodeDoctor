"""Composable preprocessing pipeline CLI: `python -m src.preprocess.run`."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.preprocess.clean import clean_log
from src.preprocess.dedupe import dedupe_records
from src.preprocess.label import export_unverified_csv, label_unverified_records
from src.preprocess.redact import redact_secrets
from src.preprocess.signature import compute_error_signature, signature_hash
from src.preprocess.split import split_by_repo, write_jsonl, write_split_report
from src.preprocess.stats import write_data_stats_report
from src.preprocess.window import extract_log_window, load_token_counter

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PREPROCESS_CONFIG = PROJECT_ROOT / "configs" / "preprocess.yaml"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_preprocess_pipeline(config_path: Path = DEFAULT_PREPROCESS_CONFIG) -> dict[str, Any]:
    """Run clean -> redact -> window -> signature -> label -> dedupe -> split -> stats."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs_cfg = cfg["inputs"]
    outputs_cfg = cfg["outputs"]
    win_cfg = cfg["window"]
    dedupe_cfg = cfg["dedupe"]
    split_cfg = cfg["split"]
    taxonomy_path = _resolve(cfg["taxonomy_path"])

    raw_records: list[dict[str, Any]] = []
    syn_path = _resolve(inputs_cfg["synthetic_jsonl"])
    raw_records.extend(_load_jsonl(syn_path))

    for pattern in inputs_cfg.get("raw_globs", []):
        abs_pattern = str(_resolve(pattern))
        for matched in sorted(glob.glob(abs_pattern)):
            raw_records.extend(_load_jsonl(Path(matched)))

    token_counter = load_token_counter(win_cfg["tokenizer_name"])

    processed: list[dict[str, Any]] = []
    for rec in raw_records:
        raw_text = str(rec.get("log_excerpt") or rec.get("raw_log") or "")
        if not raw_text.strip():
            continue

        cleaned = clean_log(raw_text)
        redacted = redact_secrets(cleaned)
        windowed = extract_log_window(
            redacted,
            context_before=win_cfg["context_before"],
            context_after=win_cfg["context_after"],
            tail_lines=win_cfg["tail_lines"],
            max_tokens=win_cfg["max_tokens"],
            token_counter=token_counter,
        )
        sig = compute_error_signature(redacted)
        meta = rec.get("metadata") or {}

        processed.append(
            {
                **rec,
                "repo": rec.get("repo") or meta.get("repo"),
                "framework": rec.get("framework") or meta.get("framework") or "unknown",
                "package_manager": rec.get("package_manager")
                or meta.get("package_manager")
                or "unknown",
                "log_excerpt": windowed,
                "signature": sig,
                "signature_hash": signature_hash(sig),
            }
        )

    labeled, unverified = label_unverified_records(processed, taxonomy_path=taxonomy_path)
    deduped, dedupe_stats = dedupe_records(
        labeled,
        num_perm=dedupe_cfg["minhash_num_perm"],
        jaccard_threshold=dedupe_cfg["jaccard_threshold"],
        max_per_signature=dedupe_cfg["max_per_signature"],
        shingle_k=dedupe_cfg["shingle_k"],
    )

    train, val, test = split_by_repo(
        deduped,
        seed=split_cfg["seed"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        held_out_repos=split_cfg.get("held_out_repos", []),
    )

    train_path = write_jsonl(_resolve(outputs_cfg["train_jsonl"]), train)
    val_path = write_jsonl(_resolve(outputs_cfg["val_jsonl"]), val)
    test_path = write_jsonl(_resolve(outputs_cfg["test_jsonl"]), test)
    unverified_csv = export_unverified_csv(unverified, _resolve(outputs_cfg["unverified_csv"]))
    split_report = write_split_report(
        _resolve(outputs_cfg["split_report"]), train, val, test, seed=split_cfg["seed"]
    )
    stats_report = write_data_stats_report(
        _resolve(outputs_cfg["stats_report"]), deduped, token_counter=token_counter
    )

    return {
        "raw_input_count": len(raw_records),
        "dedupe": dedupe_stats,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "unverified_count": len(unverified),
        "outputs": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "test_jsonl": str(test_path),
            "unverified_csv": str(unverified_csv),
            "split_report": str(split_report),
            "stats_report": str(stats_report),
        },
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="preprocess",
        help="Run the LaunchForge AI preprocessing pipeline.",
        add_completion=False,
    )

    @cli.command("run")
    def run_cmd(
        config: Path = typer.Option(DEFAULT_PREPROCESS_CONFIG, "--config", "-c"),
    ) -> None:
        """Execute all preprocessing stages and write processed splits + reports."""
        summary = run_preprocess_pipeline(config)
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
