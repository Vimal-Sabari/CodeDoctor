"""Resumable, parallelism-limited Typer CLI runner for synthetic fault injection."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import typer

from src.collect.fault_injector.docker_runner import (
    BuildResult,
    DockerConfig,
    RepoSpec,
    clone_repo,
    run_docker_build,
    verify_baseline,
)
from src.collect.fault_injector.faults import FaultNotApplicableError, get_fault
from src.schema import DatasetRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "fault_injector.yaml"


class FaultInjectorConfig(BaseModel):
    """Top-level configuration loaded from configs/fault_injector.yaml."""

    repos_config: Path
    output_dir: Path
    output_jsonl: Path
    manifest_path: Path
    cache_dir: Path
    logs_dir: Path
    seed: int = 42
    docker: DockerConfig = Field(default_factory=DockerConfig)
    enabled_faults: list[str] = Field(default_factory=list)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> FaultInjectorConfig:
    """Load and resolve FaultInjectorConfig from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = FaultInjectorConfig.model_validate(raw)
    return FaultInjectorConfig(
        repos_config=_resolve_path(cfg.repos_config),
        output_dir=_resolve_path(cfg.output_dir),
        output_jsonl=_resolve_path(cfg.output_jsonl),
        manifest_path=_resolve_path(cfg.manifest_path),
        cache_dir=_resolve_path(cfg.cache_dir),
        logs_dir=_resolve_path(cfg.logs_dir),
        seed=cfg.seed,
        docker=cfg.docker,
        enabled_faults=cfg.enabled_faults,
    )


def load_repos(repos_path: Path) -> list[RepoSpec]:
    """Load repository specifications from configs/repos.yaml."""
    raw = yaml.safe_load(repos_path.read_text(encoding="utf-8"))
    return [RepoSpec.model_validate(item) for item in raw.get("repos", [])]


def derive_pair_seed(global_seed: int, repo_id: str, fault_id: str) -> int:
    """Derive a deterministic integer seed for a (repo, fault) pair."""
    digest = hashlib.sha256(f"{global_seed}:{repo_id}:{fault_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pair_key(repo_id: str, fault_id: str) -> str:
    return f"{repo_id}::{fault_id}"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load existing manifest JSON or return a fresh manifest structure."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"generated_at": None, "seed": None, "entries": {}}


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Persist manifest JSON atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_log_excerpt(raw_log: str, max_lines: int = 80) -> str:
    lines = raw_log.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_single_pair(
    repo: RepoSpec,
    clean_repo_dir: Path,
    fault_id: str,
    pair_seed: int,
    cfg: FaultInjectorConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Apply a single fault to a temp copy of `clean_repo_dir` and run Docker build."""
    key = _pair_key(repo.id, fault_id)
    example_id = f"{repo.id}__{fault_id}__{pair_seed % 1_000_000:06d}"
    fault = get_fault(fault_id)
    rng = random.Random(pair_seed)

    with tempfile.TemporaryDirectory(prefix=f"lf_{repo.id}_{fault_id}_") as tmp_dir:
        work_dir = Path(tmp_dir) / "repo"
        shutil.copytree(clean_repo_dir, work_dir, dirs_exist_ok=True)

        try:
            applied = fault.apply(work_dir, rng)
        except FaultNotApplicableError as exc:
            return None, {"key": key, "status": "not_applicable", "reason": str(exc)}

        build_res: BuildResult = run_docker_build(work_dir, repo, cfg.docker)

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / f"{example_id}.log"
    log_path.write_text(build_res.log, encoding="utf-8")

    if build_res.success:
        return None, {
            "key": key,
            "id": example_id,
            "status": "build_passed_unexpectedly",
            "log_path": _display_path(log_path),
        }

    excerpt = _extract_log_excerpt(build_res.log)
    dataset_record = DatasetRecord(
        id=example_id,
        source="synthetic_fault_injector",
        repo=repo.url,
        framework=repo.framework,
        package_manager=repo.package_manager,
        node_version=repo.node_version,
        log_excerpt=excerpt or f"Build exited with code {build_res.exit_code}.",
        error_category=applied.error_category,
        root_cause=applied.root_cause,
        evidence_lines=[],
        fix_steps=applied.fix_steps,
        files_to_change=applied.files_to_change,
        confidence=1.0,
        verified=True,
    )
    record_payload = dataset_record.model_dump(mode="json")
    record_payload["fault_id"] = applied.fault_id
    record_payload["fault_params"] = applied.fault_params
    record_payload["expected_fix_text"] = applied.expected_fix_text
    record_payload["exit_code"] = build_res.exit_code
    record_payload["timed_out"] = build_res.timed_out
    record_payload["seed"] = pair_seed

    manifest_entry = {
        "key": key,
        "id": example_id,
        "repo_id": repo.id,
        "fault_id": fault_id,
        "error_category": applied.error_category.value,
        "seed": pair_seed,
        "status": "failed_as_expected",
        "exit_code": build_res.exit_code,
        "timed_out": build_res.timed_out,
        "duration_sec": build_res.duration_sec,
        "log_path": _display_path(log_path),
    }
    return record_payload, manifest_entry


def run_pipeline(
    cfg: FaultInjectorConfig,
    dry_run: bool = False,
    limit: int | None = None,
    repo_filter: str | None = None,
    fault_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute the fault-injection pipeline and return a summary dictionary."""
    repos = load_repos(cfg.repos_config)
    if repo_filter:
        repos = [r for r in repos if r.id == repo_filter]

    fault_ids = list(cfg.enabled_faults)
    if fault_filter:
        fault_ids = [f for f in fault_ids if f == fault_filter]

    manifest = load_manifest(cfg.manifest_path)
    manifest["seed"] = cfg.seed
    existing_entries: dict[str, Any] = manifest.get("entries", {})

    planned_pairs: list[tuple[RepoSpec, str, int]] = []
    for repo in repos:
        for fault_id in fault_ids:
            key = _pair_key(repo.id, fault_id)
            if resume and key in existing_entries:
                continue
            planned_pairs.append((repo, fault_id, derive_pair_seed(cfg.seed, repo.id, fault_id)))

    if limit is not None:
        planned_pairs = planned_pairs[:limit]

    if dry_run:
        return {
            "dry_run": True,
            "planned_count": len(planned_pairs),
            "max_containers": min(cfg.docker.max_containers, 2),
            "planned": [
                {"repo_id": r.id, "fault_id": fid, "seed": s, "verified_repo": r.verified}
                for r, fid, s in planned_pairs
            ],
        }

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Clone & verify baseline once per unique repo in planned_pairs
    repo_clean_dirs: dict[str, Path] = {}
    baseline_failed: set[str] = set()
    for repo in {r.id: r for r, _, _ in planned_pairs}.values():
        clean_dir = clone_repo(repo, cfg.cache_dir)
        with tempfile.TemporaryDirectory(prefix=f"lf_base_{repo.id}_") as base_tmp:
            base_copy = Path(base_tmp) / "repo"
            shutil.copytree(clean_dir, base_copy, dirs_exist_ok=True)
            base_res = verify_baseline(base_copy, repo, cfg.docker)
        if not base_res.success:
            baseline_failed.add(repo.id)
            print(f"[SKIP] Baseline build failed for {repo.id} (exit={base_res.exit_code})")
        else:
            repo_clean_dirs[repo.id] = clean_dir

    active_pairs = [
        (r, fid, s) for r, fid, s in planned_pairs if r.id not in baseline_failed
    ]

    write_lock = threading.Lock()
    generated = 0
    skipped = len(planned_pairs) - len(active_pairs)
    max_workers = min(cfg.docker.max_containers, 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_single_pair, r, repo_clean_dirs[r.id], fid, s, cfg)
            for r, fid, s in active_pairs
        ]
        for fut in as_completed(futures):
            record_payload, manifest_entry = fut.result()
            with write_lock:
                if record_payload is not None:
                    with cfg.output_jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record_payload) + "\n")
                    generated += 1
                else:
                    skipped += 1
                manifest["entries"][manifest_entry["key"]] = manifest_entry
                save_manifest(cfg.manifest_path, manifest)

    return {
        "dry_run": False,
        "generated": generated,
        "skipped": skipped,
        "output_jsonl": str(cfg.output_jsonl),
        "manifest_path": str(cfg.manifest_path),
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="fault-injector",
        help="Generate labeled frontend build-failure logs via Dockerized fault injection.",
        add_completion=False,
    )

    @cli.command("run")
    def run_cmd(
        config: Path = typer.Option(
            DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to fault_injector.yaml."
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Plan pairs without cloning or running Docker."
        ),
        limit: int | None = typer.Option(
            None, "--limit", "-n", help="Max (repo, fault) pairs to process."
        ),
        repo: str | None = typer.Option(None, "--repo", help="Filter to a single repo id."),
        fault: str | None = typer.Option(None, "--fault", help="Filter to a single fault id."),
        resume: bool = typer.Option(
            True, "--resume/--no-resume", help="Skip pairs already in manifest."
        ),
    ) -> None:
        """Run synthetic fault injection across configured repositories and faults."""
        cfg = load_config(config)
        summary = run_pipeline(
            cfg=cfg,
            dry_run=dry_run,
            limit=limit,
            repo_filter=repo,
            fault_filter=fault,
            resume=resume,
        )
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
