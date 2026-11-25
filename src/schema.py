"""Pydantic schemas, taxonomy loader, and JSON Schema CLI for LaunchForge AI."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "configs" / "taxonomy.yaml"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schema.json"


class ErrorCategory(StrEnum):
    """Canonical frontend build failure categories matching configs/taxonomy.yaml."""

    DEPENDENCY_MISSING = "dependency_missing"
    VERSION_CONFLICT = "version_conflict"
    PEER_DEPENDENCY_CONFLICT = "peer_dependency_conflict"
    TYPE_ERROR = "type_error"
    LINT_ERROR = "lint_error"
    CONFIG_ERROR = "config_error"
    ENV_MISSING = "env_missing"
    NODE_VERSION_MISMATCH = "node_version_mismatch"
    OOM = "oom"
    PATH_CASE = "path_case"
    IMPORT_PATH_ERROR = "import_path_error"
    ASSET_MISSING = "asset_missing"
    LOCKFILE_MISMATCH = "lockfile_mismatch"
    NETWORK_REGISTRY = "network_registry"
    OUTPUT_DIR_WRONG = "output_dir_wrong"
    SYNTAX_ERROR = "syntax_error"
    SSR_BROWSER_API_ERROR = "ssr_browser_api_error"
    UNKNOWN = "unknown"


class TaxonomyEntry(BaseModel):
    """Single category definition loaded from configs/taxonomy.yaml."""

    id: ErrorCategory
    description: str = Field(min_length=1)
    typical_log_signatures: list[str] = Field(default_factory=list)
    example_fix: str = Field(min_length=1)


class DatasetRecord(BaseModel):
    """Training/evaluation dataset record for a frontend build failure."""

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    repo: str | None = None
    framework: str = Field(min_length=1)
    package_manager: str = Field(min_length=1)
    node_version: str | None = None
    log_excerpt: str = Field(min_length=1)
    error_category: ErrorCategory
    root_cause: str = Field(min_length=1)
    evidence_lines: list[str] = Field(default_factory=list)
    fix_steps: list[str] = Field(min_length=1)
    files_to_change: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = False

    @field_validator("log_excerpt", "root_cause")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be blank.")
        return value


class ModelOutput(BaseModel):
    """Structured JSON output contract produced by the fine-tuned model."""

    error_category: ErrorCategory
    root_cause: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    fix_steps: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("root_cause")
    @classmethod
    def _root_cause_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("root_cause must not be blank.")
        return value


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> list[TaxonomyEntry]:
    """Load and validate taxonomy entries from a YAML file."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [TaxonomyEntry.model_validate(item) for item in raw.get("categories", [])]


def build_model_output_schema() -> dict[str, Any]:
    """Generate the JSON Schema for ModelOutput with an explicit error_category enum."""
    schema = ModelOutput.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def export_model_output_schema(output: Path = DEFAULT_SCHEMA_PATH) -> Path:
    """Write the ModelOutput JSON Schema to disk and return the output path."""
    schema = build_model_output_schema()
    output.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return output


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="schema",
        help="Validate datasets/outputs and export ModelOutput JSON Schema.",
        add_completion=False,
    )

    @cli.command("export-schema")
    def export_schema_cmd(
        output: Path = typer.Option(
            DEFAULT_SCHEMA_PATH,
            "--output",
            "-o",
            help="Destination path for ModelOutput JSON Schema.",
        ),
    ) -> None:
        """Export ModelOutput JSON Schema to disk."""
        dest = export_model_output_schema(output)
        typer.echo(f"Exported ModelOutput JSON Schema to {dest}")

    @cli.command("validate")
    def validate_cmd(
        path: Path = typer.Argument(..., help="Path to a JSON or JSONL file to validate."),
        kind: str = typer.Option(
            "dataset",
            "--kind",
            "-k",
            help="Record type to validate: 'dataset' or 'output'.",
        ),
    ) -> None:
        """Validate a JSON or JSONL file against DatasetRecord or ModelOutput."""
        model_cls: type[DatasetRecord] | type[ModelOutput]
        if kind == "dataset":
            model_cls = DatasetRecord
        elif kind == "output":
            model_cls = ModelOutput
        else:
            typer.echo(f"Invalid kind '{kind}'. Expected 'dataset' or 'output'.")
            raise typer.Exit(code=1)

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            typer.echo("File is empty.")
            raise typer.Exit(code=1)

        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            records = parsed if isinstance(parsed, list) else [parsed]

        for item in records:
            model_cls.model_validate(item)
        typer.echo(f"Validated {len(records)} {kind} record(s) successfully.")

    return cli


if __name__ == "__main__":
    get_cli_app()()
