"""Smoke tests for LaunchForge AI package structure and CLI entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

PACKAGE_MODULES = [
    "src",
    "src.collect",
    "src.preprocess",
    "src.train",
    "src.rag",
    "src.eval",
    "src.serve",
]


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_package_imports(module_name: str) -> None:
    """Verify every scaffolded package module can be imported cleanly."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_check_env_cli_help() -> None:
    """Verify scripts/check_env.py exposes a valid Typer CLI."""
    typer = pytest.importorskip("typer")
    pytest.importorskip("psutil")
    from typer.testing import CliRunner

    spec = importlib.util.spec_from_file_location(
        "check_env",
        Path(__file__).resolve().parent.parent / "scripts" / "check_env.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert isinstance(module.app, typer.Typer)
    runner = CliRunner()
    result = runner.invoke(module.app, ["--help"])
    assert result.exit_code == 0
    assert "Verify Python, RAM, CUDA 12.8+" in result.output
