"""Shared raw-data schema, config loader, and append-only JSONL utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COLLECT_CONFIG = PROJECT_ROOT / "configs" / "collect.yaml"


class RawLogRecord(BaseModel):
    """Unified append-only raw record stored in data/raw/*.jsonl."""

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    raw_log: str = Field(min_length=1)
    title: str | None = None
    answer_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    collected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_collect_config(config_path: Path = DEFAULT_COLLECT_CONFIG) -> dict[str, Any]:
    """Load raw data collection settings from configs/collect.yaml."""
    return dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))


def load_existing_ids(jsonl_path: Path) -> set[str]:
    """Read existing record IDs from an append-only JSONL file for resumability."""
    if not jsonl_path.exists():
        return set()
    ids: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def append_raw_records(jsonl_path: Path, records: list[RawLogRecord]) -> int:
    """Append new RawLogRecord entries to `jsonl_path` without modifying existing lines."""
    if not records:
        return 0
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(jsonl_path)
    written = 0
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            if rec.id in existing_ids:
                continue
            fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
            existing_ids.add(rec.id)
            written += 1
    return written
