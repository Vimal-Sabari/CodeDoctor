"""Optional loader stubs for public CI-log research datasets (e.g., LogChunks)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.collect.common import (
    DEFAULT_COLLECT_CONFIG,
    RawLogRecord,
    append_raw_records,
    load_collect_config,
    resolve_project_path,
)

if TYPE_CHECKING:
    import typer


class BasePublicDatasetLoader(ABC):
    """Interface for importing external CI-log research datasets into data/raw/."""

    dataset_name: str

    @abstractmethod
    def load(self, input_path: Path) -> Iterator[RawLogRecord]:
        """Yield RawLogRecord entries from a locally downloaded dataset file."""


class LogChunksLoader(BasePublicDatasetLoader):
    """Optional loader stub for the LogChunks CI failure dataset (JSONL export)."""

    dataset_name = "logchunks"

    def load(self, input_path: Path) -> Iterator[RawLogRecord]:
        if not input_path.exists():
            raise FileNotFoundError(
                f"LogChunks file not found at {input_path}. "
                "Download the public dataset locally first; this loader is optional."
            )

        with input_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                item: dict[str, Any] = json.loads(line)
                log_text = item.get("log") or item.get("raw_log") or item.get("text") or ""
                if not log_text.strip():
                    continue
                rec_id = str(item.get("id") or f"logchunks_{idx}")
                yield RawLogRecord(
                    id=rec_id,
                    source=self.dataset_name,
                    raw_log=log_text,
                    title=item.get("title"),
                    metadata={
                        "dataset": self.dataset_name,
                        "repo": item.get("repo"),
                        "language": item.get("language"),
                        "license": item.get("license", "see_upstream_dataset"),
                    },
                )


def import_public_dataset(
    input_path: Path,
    output_jsonl: Path,
    loader: BasePublicDatasetLoader | None = None,
) -> dict[str, Any]:
    """Load records via `loader` and append them to `output_jsonl`."""
    active_loader = loader or LogChunksLoader()
    records = list(active_loader.load(input_path))
    written = append_raw_records(output_jsonl, records)
    return {
        "dataset": active_loader.dataset_name,
        "records_loaded": len(records),
        "records_written": written,
        "output_jsonl": str(output_jsonl),
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="public-datasets",
        help="Optional loader for public CI-log research datasets (e.g., LogChunks).",
        add_completion=False,
    )

    @cli.command("load-logchunks")
    def load_logchunks_cmd(
        input_path: Path = typer.Argument(..., help="Path to local LogChunks JSONL file."),
        config: Path = typer.Option(DEFAULT_COLLECT_CONFIG, "--config", "-c"),
    ) -> None:
        """Import a local LogChunks JSONL file into data/raw/public_ci_logs.jsonl."""
        raw_cfg = load_collect_config(config)
        out_path = resolve_project_path(raw_cfg["public_datasets"]["output_jsonl"])
        summary = import_public_dataset(input_path, out_path, LogChunksLoader())
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
