"""Repo-grouped deterministic train/val/test splitting (zero repo leakage)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_group_key(rec: dict[str, Any]) -> str:
    repo = rec.get("repo")
    if repo and isinstance(repo, str) and repo.strip():
        return repo.strip()
    return f"__no_repo__:{rec.get('id', 'unknown')}"


def split_by_repo(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    held_out_repos: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by repository with a fixed seed and zero repo overlap."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    held_out = set(held_out_repos or [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[_repo_group_key(rec)].append(rec)

    test_repos: set[str] = {k for k in grouped if k in held_out}
    remaining_repos = sorted(k for k in grouped if k not in test_repos)

    rng = random.Random(seed)
    rng.shuffle(remaining_repos)

    n_rem = len(remaining_repos)
    n_train = int(n_rem * train_ratio)
    n_val = int(n_rem * val_ratio)
    if n_rem >= 3:
        n_train = max(1, min(n_rem - 2, n_train))
        n_val = max(1, min(n_rem - n_train - 1, n_val))

    train_repos = set(remaining_repos[:n_train])
    val_repos = set(remaining_repos[n_train : n_train + n_val])
    test_repos |= set(remaining_repos[n_train + n_val :])

    assert not (train_repos & val_repos), "Repo overlap between train and val"
    assert not (train_repos & test_repos), "Repo overlap between train and test"
    assert not (val_repos & test_repos), "Repo overlap between val and test"

    train = [r for k in sorted(train_repos) for r in grouped[k]]
    val = [r for k in sorted(val_repos) for r in grouped[k]]
    test = [r for k in sorted(test_repos) for r in grouped[k]]
    return train, val, test


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def write_split_report(
    report_path: Path,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
) -> Path:
    """Write a markdown summary of the repo-level dataset split."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    train_repos = sorted({_repo_group_key(r) for r in train})
    val_repos = sorted({_repo_group_key(r) for r in val})
    test_repos = sorted({_repo_group_key(r) for r in test})

    lines = [
        "# LaunchForge AI — Dataset Split Report",
        "",
        f"- **Seed:** `{seed}`",
        f"- **Total records:** {len(train) + len(val) + len(test)}",
        "",
        "| Split | Records | Unique Repos |",
        "| --- | ---: | ---: |",
        f"| train | {len(train)} | {len(train_repos)} |",
        f"| val | {len(val)} | {len(val_repos)} |",
        f"| test | {len(test)} | {len(test_repos)} |",
        "",
        "- **Repo overlap check:** 0 overlapping repos across train/val/test.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
