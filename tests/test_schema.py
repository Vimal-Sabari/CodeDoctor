"""Tests for Pydantic schemas, taxonomy consistency, JSON Schema, and prompt templates."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.schema import (
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TAXONOMY_PATH,
    DatasetRecord,
    ErrorCategory,
    ModelOutput,
    build_model_output_schema,
    load_taxonomy,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = PROJECT_ROOT / "configs" / "prompts.yaml"


def test_taxonomy_count_and_enum_consistency() -> None:
    """Verify taxonomy has 15-20 categories, includes unknown, and matches ErrorCategory."""
    entries = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    assert 15 <= len(entries) <= 20

    taxonomy_ids = [entry.id.value for entry in entries]
    assert len(taxonomy_ids) == len(set(taxonomy_ids)), "Duplicate category IDs in taxonomy.yaml"
    assert "unknown" in taxonomy_ids

    enum_values = [member.value for member in ErrorCategory]
    assert set(taxonomy_ids) == set(enum_values)
    assert taxonomy_ids == enum_values


def test_taxonomy_regex_signatures_compile() -> None:
    """Verify all typical_log_signatures regex hints in taxonomy.yaml compile cleanly."""
    entries = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    for entry in entries:
        for pattern in entry.typical_log_signatures:
            compiled = re.compile(pattern)
            assert compiled is not None


def test_schema_json_matches_model_and_taxonomy() -> None:
    """Verify schema.json matches ModelOutput.model_json_schema() and taxonomy enum."""
    assert DEFAULT_SCHEMA_PATH.exists(), "schema.json is missing"
    disk_schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    generated_schema = build_model_output_schema()
    assert disk_schema == generated_schema

    enum_in_schema = disk_schema["$defs"]["ErrorCategory"]["enum"]
    taxonomy_ids = [entry.id.value for entry in load_taxonomy(DEFAULT_TAXONOMY_PATH)]
    assert enum_in_schema == taxonomy_ids


def test_dataset_record_validation() -> None:
    """Verify DatasetRecord accepts valid data and rejects invalid fields."""
    valid = DatasetRecord(
        id="rec-001",
        source="github_actions",
        repo="example/web-app",
        framework="nextjs",
        package_manager="npm",
        node_version="20.11.0",
        log_excerpt="Error: Cannot find module 'lucide-react'",
        error_category=ErrorCategory.DEPENDENCY_MISSING,
        root_cause="lucide-react is imported in src/App.tsx but missing from package.json.",
        evidence_lines=["Error: Cannot find module 'lucide-react'"],
        fix_steps=["Run `npm install lucide-react` and commit package.json."],
        files_to_change=["package.json", "package-lock.json"],
        confidence=0.95,
        verified=True,
    )
    assert valid.error_category == ErrorCategory.DEPENDENCY_MISSING

    with pytest.raises(ValidationError):
        DatasetRecord(
            id="rec-bad-conf",
            source="synthetic",
            framework="vite",
            package_manager="pnpm",
            log_excerpt="SyntaxError: Unexpected token",
            error_category=ErrorCategory.SYNTAX_ERROR,
            root_cause="Bad syntax",
            fix_steps=["Fix syntax"],
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        DatasetRecord(
            id="rec-bad-cat",
            source="synthetic",
            framework="vite",
            package_manager="pnpm",
            log_excerpt="Some error",
            error_category="not_a_real_category",  # type: ignore[arg-type]
            root_cause="Unknown",
            fix_steps=["Check logs"],
            confidence=0.2,
        )


def test_model_output_validation() -> None:
    """Verify ModelOutput enforces required fields, enum, and confidence bounds."""
    output = ModelOutput(
        error_category=ErrorCategory.UNKNOWN,
        root_cause="Log is truncated before the error message appears.",
        evidence=[],
        fix_steps=["Re-run CI with verbose logging to capture the full failure."],
        confidence=0.2,
    )
    assert output.error_category == ErrorCategory.UNKNOWN

    with pytest.raises(ValidationError):
        ModelOutput(
            error_category=ErrorCategory.OOM,
            root_cause="   ",
            evidence=[],
            fix_steps=["Increase heap size"],
            confidence=0.8,
        )

    with pytest.raises(ValidationError):
        ModelOutput(
            error_category=ErrorCategory.OOM,
            root_cause="Heap exhausted",
            evidence=[],
            fix_steps=[],
            confidence=0.8,
        )


def test_prompts_yaml_structure_and_rules() -> None:
    """Verify configs/prompts.yaml contains required templates and grounding rules."""
    prompts = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    for key in ("system", "user_no_rag", "user_with_rag"):
        assert key in prompts and isinstance(prompts[key], str) and prompts[key].strip()

    system_lower = prompts["system"].lower()
    assert "unknown" in system_lower
    assert "verbatim" in system_lower or "exist in" in system_lower

    rendered_no_rag = prompts["user_no_rag"].format(
        framework="nextjs",
        package_manager="npm",
        log_excerpt="error TS2304: Cannot find name 'Foo'.",
    )
    assert "nextjs" in rendered_no_rag
    assert "TS2304" in rendered_no_rag

    rendered_with_rag = prompts["user_with_rag"].format(
        framework="vite",
        package_manager="pnpm",
        rag_context="Vite requires Node 18+.",
        log_excerpt="Unsupported Node.js version",
    )
    assert "Vite requires Node 18+." in rendered_with_rag
    assert "Unsupported Node.js version" in rendered_with_rag
