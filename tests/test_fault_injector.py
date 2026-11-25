"""Tests for fault injection library and runner with a mocked Docker layer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

from src.collect.fault_injector.docker_runner import BuildResult, DockerConfig, RepoSpec
from src.collect.fault_injector.faults import (
    RemoveDependencyFault,
    TypeScriptTypeErrorFault,
    WrongImportPathFault,
    get_fault,
)
from src.collect.fault_injector.runner import (
    FaultInjectorConfig,
    derive_pair_seed,
    load_manifest,
    run_pipeline,
)
from src.schema import ErrorCategory


@pytest.fixture()
def sample_repo_dir(tmp_path: Path) -> Path:
    """Create a minimal TypeScript frontend repository fixture."""
    repo = tmp_path / "sample_frontend"
    (repo / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps(
            {
                "name": "sample-frontend",
                "version": "1.0.0",
                "dependencies": {"react": "^18.3.0", "lucide-react": "^0.400.0"},
                "devDependencies": {"typescript": "^5.5.0"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.ts").write_text(
        "export const greet = (name: string): string => `Hello, ${name}`;\n",
        encoding="utf-8",
    )
    (repo / "src" / "index.ts").write_text(
        "import { greet } from './helper';\nconsole.log(greet('LaunchForge'));\n",
        encoding="utf-8",
    )
    return repo


def test_remove_dependency_fault_apply_and_verify(sample_repo_dir: Path) -> None:
    """RemoveDependencyFault removes a dependency and verifies restoration."""
    fault = RemoveDependencyFault()
    original_pkg = (sample_repo_dir / "package.json").read_text(encoding="utf-8")

    applied = fault.apply(sample_repo_dir, random.Random(42))
    assert applied.error_category == ErrorCategory.DEPENDENCY_MISSING
    assert not fault.verify_fix(sample_repo_dir, applied.fault_params)

    (sample_repo_dir / "package.json").write_text(original_pkg, encoding="utf-8")
    assert fault.verify_fix(sample_repo_dir, applied.fault_params)


def test_wrong_import_path_fault_apply_and_verify(sample_repo_dir: Path) -> None:
    """WrongImportPathFault breaks a relative import and verifies restoration."""
    fault = WrongImportPathFault()
    index_path = sample_repo_dir / "src" / "index.ts"
    original_src = index_path.read_text(encoding="utf-8")

    applied = fault.apply(sample_repo_dir, random.Random(42))
    assert applied.error_category == ErrorCategory.IMPORT_PATH_ERROR
    assert not fault.verify_fix(sample_repo_dir, applied.fault_params)

    index_path.write_text(original_src, encoding="utf-8")
    assert fault.verify_fix(sample_repo_dir, applied.fault_params)


def test_ts_type_error_fault_apply_and_verify(sample_repo_dir: Path) -> None:
    """TypeScriptTypeErrorFault injects a type mismatch and verifies removal."""
    fault = TypeScriptTypeErrorFault()
    applied = fault.apply(sample_repo_dir, random.Random(42))
    assert applied.error_category == ErrorCategory.TYPE_ERROR
    assert not fault.verify_fix(sample_repo_dir, applied.fault_params)

    target = sample_repo_dir / applied.fault_params["file_path"]
    cleaned = target.read_text(encoding="utf-8").split("// Injected type error")[0]
    target.write_text(cleaned, encoding="utf-8")
    assert fault.verify_fix(sample_repo_dir, applied.fault_params)


def test_deterministic_pair_seed() -> None:
    """Pair seed derivation is deterministic and distinct per (repo, fault)."""
    s1 = derive_pair_seed(42, "repo-a", "remove_dependency")
    s2 = derive_pair_seed(42, "repo-a", "remove_dependency")
    s3 = derive_pair_seed(42, "repo-a", "wrong_import_path")
    assert s1 == s2
    assert s1 != s3
    assert get_fault("remove_dependency").fault_id == "remove_dependency"


def test_runner_dry_run_and_mocked_docker(
    tmp_path: Path,
    sample_repo_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dry-run planning, mocked Docker execution, JSONL output, and resumability."""
    repos_yaml = tmp_path / "repos.yaml"
    repos_yaml.write_text(
        yaml.safe_dump(
            {
                "repos": [
                    RepoSpec(
                        id="fixture-react",
                        url="https://example.com/fixture.git",
                        framework="vite-react",
                        package_manager="npm",
                        node_version="20",
                        verified=True,
                    ).model_dump()
                ]
            }
        ),
        encoding="utf-8",
    )

    cfg = FaultInjectorConfig(
        repos_config=repos_yaml,
        output_dir=tmp_path / "synthetic",
        output_jsonl=tmp_path / "synthetic" / "failures.jsonl",
        manifest_path=tmp_path / "synthetic" / "manifest.json",
        cache_dir=tmp_path / "cache",
        logs_dir=tmp_path / "logs",
        seed=42,
        docker=DockerConfig(max_containers=2, timeout_sec=60, memory_limit="2g"),
        enabled_faults=["remove_dependency", "wrong_import_path", "ts_type_error"],
    )

    # 1. Dry-run should plan all 3 pairs without creating JSONL or calling Docker
    dry = run_pipeline(cfg, dry_run=True)
    assert dry["dry_run"] is True
    assert dry["planned_count"] == 3
    assert not cfg.output_jsonl.exists()

    # 2. Mock clone_repo, verify_baseline, and run_docker_build
    monkeypatch.setattr(
        "src.collect.fault_injector.runner.clone_repo",
        lambda repo, cache_dir: sample_repo_dir,
    )
    monkeypatch.setattr(
        "src.collect.fault_injector.runner.verify_baseline",
        lambda repo_dir, repo, docker_cfg: BuildResult(
            success=True, exit_code=0, log="Build succeeded", duration_sec=0.1
        ),
    )
    monkeypatch.setattr(
        "src.collect.fault_injector.runner.run_docker_build",
        lambda repo_dir, repo, docker_cfg: BuildResult(
            success=False,
            exit_code=1,
            log="Error: simulated build failure in Docker",
            duration_sec=0.2,
        ),
    )

    summary = run_pipeline(cfg, dry_run=False)
    assert summary["generated"] == 3
    assert summary["skipped"] == 0

    lines = cfg.output_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert "fault_id" in first and "fault_params" in first
    assert first["verified"] is True

    manifest = load_manifest(cfg.manifest_path)
    assert len(manifest["entries"]) == 3

    # 3. Resume should skip all 3 already-generated pairs
    resumed = run_pipeline(cfg, dry_run=False, resume=True)
    assert resumed["generated"] == 0
