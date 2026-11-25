"""Interactive, resumable terminal tool for building the gold evaluation set."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.schema import DatasetRecord, ErrorCategory

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


def sample_stratified(
    records: list[dict[str, Any]],
    sample_size: int = 300,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample up to `sample_size` records, stratified by category and prioritizing real logs."""
    excluded = exclude_ids or set()
    candidates = [r for r in records if r.get("id") not in excluded]
    if not candidates or sample_size <= 0:
        return []

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in candidates:
        by_cat[str(rec.get("error_category", "unknown"))].append(rec)

    # Within each category, shuffle deterministically then sort real (non-synthetic) first
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
        by_cat[cat].sort(key=lambda r: 0 if r.get("source") != "synthetic_fault_injector" else 1)

    selected: list[dict[str, Any]] = []
    categories = sorted(by_cat.keys())
    while len(selected) < sample_size and any(by_cat[c] for c in categories):
        for cat in categories:
            if len(selected) >= sample_size:
                break
            if by_cat[cat]:
                selected.append(by_cat[cat].pop(0))

    return selected


def append_gold_record(gold_jsonl: Path, record: DatasetRecord) -> None:
    """Append a verified DatasetRecord to `data/gold/gold.jsonl`."""
    gold_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with gold_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def run_gold_review(
    test_jsonl: Path,
    gold_jsonl: Path,
    sample_size: int = 300,
    seed: int = 42,
) -> int:
    """Interactively review sampled test records and save verified entries to `gold_jsonl`."""
    test_records = _load_jsonl(test_jsonl)
    existing_gold = _load_jsonl(gold_jsonl)
    existing_ids = {r["id"] for r in existing_gold}

    remaining_target = max(0, sample_size - len(existing_ids))
    queue = sample_stratified(
        test_records,
        sample_size=remaining_target,
        seed=seed,
        exclude_ids=existing_ids,
    )
    if not queue:
        print(f"No remaining records to review (already {len(existing_ids)} in {gold_jsonl}).")
        return 0

    valid_cats = ", ".join(c.value for c in ErrorCategory)
    added = 0
    for idx, raw in enumerate(queue, start=1):
        print(f"\n=== [{idx}/{len(queue)}] ID={raw.get('id')} source={raw.get('source')} ===")
        print(raw.get("log_excerpt", "")[:1200])
        print(f"\nProposed category : {raw.get('error_category', 'unknown')}")
        print(f"Proposed root_cause: {raw.get('root_cause', '')}")
        print(f"Proposed fix_steps : {raw.get('fix_steps', [])}")
        print(f"Valid categories   : {valid_cats}")

        action = input("[Enter=accept / e=edit / s=skip / q=quit]: ").strip().lower()
        if action == "q":
            break
        if action == "s":
            continue

        category = str(raw.get("error_category", "unknown"))
        root_cause = str(raw.get("root_cause", ""))
        fix_steps = list(raw.get("fix_steps") or ["Inspect build log."])

        if action == "e":
            new_cat = input(f"Category [{category}]: ").strip()
            if new_cat:
                category = new_cat
            new_rc = input(f"Root cause [{root_cause}]: ").strip()
            if new_rc:
                root_cause = new_rc
            new_fix = input("Fix steps (';'-separated, blank to keep): ").strip()
            if new_fix:
                fix_steps = [s.strip() for s in new_fix.split(";") if s.strip()]

        verified_rec = DatasetRecord(
            id=str(raw["id"]),
            source=str(raw.get("source", "unknown")),
            repo=raw.get("repo"),
            framework=str(raw.get("framework", "unknown")),
            package_manager=str(raw.get("package_manager", "unknown")),
            node_version=raw.get("node_version"),
            log_excerpt=str(raw["log_excerpt"]),
            error_category=ErrorCategory(category),
            root_cause=root_cause,
            evidence_lines=list(raw.get("evidence_lines") or []),
            fix_steps=fix_steps,
            files_to_change=list(raw.get("files_to_change") or []),
            confidence=1.0,
            verified=True,
        )
        append_gold_record(gold_jsonl, verified_rec)
        added += 1

    print(f"Saved {added} verified gold record(s) to {gold_jsonl}.")
    return added


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="gold-tool",
        help="Stratified, resumable terminal review tool for building data/gold/gold.jsonl.",
        add_completion=False,
    )

    @cli.command("review")
    def review_cmd(
        config: Path = typer.Option(DEFAULT_PREPROCESS_CONFIG, "--config", "-c"),
    ) -> None:
        """Start or resume interactive gold evaluation set labeling."""
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        gold_cfg = cfg["gold"]
        run_gold_review(
            test_jsonl=_resolve(cfg["outputs"]["test_jsonl"]),
            gold_jsonl=_resolve(gold_cfg["gold_jsonl"]),
            sample_size=int(gold_cfg["sample_size"]),
            seed=int(gold_cfg["seed"]),
        )

    return cli


if __name__ == "__main__":
    get_cli_app()()
