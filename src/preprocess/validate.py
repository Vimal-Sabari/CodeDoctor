"""Dataset quality gates that fail loudly on leakage, imbalance, schema, or secret issues."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml
from pydantic import ValidationError

from src.preprocess.redact import redact_secrets
from src.preprocess.window import default_token_counter, load_token_counter
from src.schema import DatasetRecord

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PREPROCESS_CONFIG = PROJECT_ROOT / "configs" / "preprocess.yaml"


class DatasetValidationError(RuntimeError):
    """Raised when one or more dataset quality gates fail."""


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


def _repo_key(rec: dict[str, Any]) -> str | None:
    repo = rec.get("repo")
    return repo.strip() if isinstance(repo, str) and repo.strip() else None


def validate_splits(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    gold: list[dict[str, Any]] | None = None,
    max_class_share: float = 0.20,
    max_tokens: int = 1500,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Run all hard dataset quality gates; raise DatasetValidationError on any failure."""
    errors: list[str] = []
    count_tokens = token_counter or default_token_counter
    gold_records = gold or []

    # 1. Zero repo overlap between splits
    train_repos = {r for rec in train if (r := _repo_key(rec)) is not None}
    val_repos = {r for rec in val if (r := _repo_key(rec)) is not None}
    test_repos = {r for rec in test if (r := _repo_key(rec)) is not None}

    if overlap := (train_repos & val_repos):
        errors.append(f"Repo overlap between train and val: {sorted(overlap)}")
    if overlap := (train_repos & test_repos):
        errors.append(f"Repo overlap between train and test: {sorted(overlap)}")
    if overlap := (val_repos & test_repos):
        errors.append(f"Repo overlap between val and test: {sorted(overlap)}")

    # 2. No class above max_class_share (default 20%) of train
    if train:
        cat_counts = Counter(str(r.get("error_category", "unknown")) for r in train)
        for cat, cnt in cat_counts.items():
            share = cnt / len(train)
            if share > max_class_share:
                errors.append(
                    f"Class '{cat}' is {share:.1%} of train ({cnt}/{len(train)}), "
                    f"exceeding max_class_share={max_class_share:.1%}."
                )

    # 3. Schema validation, 4. Secret detection, 6. Token-length cap
    for split_name, split_recs in [
        ("train", train),
        ("val", val),
        ("test", test),
        ("gold", gold_records),
    ]:
        for idx, rec in enumerate(split_recs):
            rec_id = rec.get("id", f"index_{idx}")
            try:
                DatasetRecord.model_validate(rec)
            except ValidationError as exc:
                errors.append(f"[{split_name}:{rec_id}] Schema validation failed: {exc}")

            excerpt = str(rec.get("log_excerpt", ""))
            if redact_secrets(excerpt) != excerpt:
                errors.append(f"[{split_name}:{rec_id}] Potential secret detected in log_excerpt.")

            tok_len = count_tokens(excerpt)
            if tok_len > max_tokens:
                errors.append(
                    f"[{split_name}:{rec_id}] log_excerpt has {tok_len} tokens > max_tokens={max_tokens}."
                )

    # 5. Gold set contains no train-split record IDs
    train_ids = {str(r.get("id")) for r in train}
    gold_ids = {str(r.get("id")) for r in gold_records}
    if leaked_ids := (train_ids & gold_ids):
        errors.append(f"Gold set contains train-split record IDs: {sorted(leaked_ids)}")

    if errors:
        raise DatasetValidationError("\n".join(errors))

    return {
        "status": "PASS",
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "gold": len(gold_records),
        },
        "unique_repos": {
            "train": len(train_repos),
            "val": len(val_repos),
            "test": len(test_repos),
        },
    }


def validate_from_config(config_path: Path = DEFAULT_PREPROCESS_CONFIG) -> dict[str, Any]:
    """Load splits from `configs/preprocess.yaml` and run all hard quality gates."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = cfg["outputs"]
    win_cfg = cfg["window"]
    val_cfg = cfg.get("validation", {})
    gold_cfg = cfg.get("gold", {})

    train = _load_jsonl(_resolve(outputs["train_jsonl"]))
    val = _load_jsonl(_resolve(outputs["val_jsonl"]))
    test = _load_jsonl(_resolve(outputs["test_jsonl"]))
    gold = _load_jsonl(_resolve(gold_cfg["gold_jsonl"])) if "gold_jsonl" in gold_cfg else []

    token_counter = load_token_counter(win_cfg["tokenizer_name"])
    return validate_splits(
        train=train,
        val=val,
        test=test,
        gold=gold,
        max_class_share=float(val_cfg.get("max_class_share", 0.20)),
        max_tokens=int(win_cfg["max_tokens"]),
        token_counter=token_counter,
    )


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="validate",
        help="Run dataset quality gates and fail loudly on any violation.",
        add_completion=False,
    )

    @cli.command("check")
    def check_cmd(
        config: Path = typer.Option(DEFAULT_PREPROCESS_CONFIG, "--config", "-c"),
    ) -> None:
        """Validate processed splits and gold set against all quality gates."""
        try:
            summary = validate_from_config(config)
        except DatasetValidationError as exc:
            typer.echo(f"DATASET VALIDATION FAILED:\n{exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
