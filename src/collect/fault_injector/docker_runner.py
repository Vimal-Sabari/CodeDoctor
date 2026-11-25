"""Docker-isolated build execution and repository cloning for fault injection."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field


class RepoSpec(BaseModel):
    """Repository specification loaded from configs/repos.yaml."""

    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    subdir: str = ""
    commit: str = "VERIFY_SHA"
    framework: str = Field(min_length=1)
    package_manager: str = Field(min_length=1)
    node_version: str = "20"
    install_cmd: str = "npm install"
    build_cmd: str = "npm run build"
    verified: bool = False
    notes: str = ""


class DockerConfig(BaseModel):
    """Docker execution limits loaded from configs/fault_injector.yaml."""

    image_template: str = "node:{node_version}-slim"
    memory_limit: str = "4g"
    timeout_sec: int = Field(default=300, ge=10)
    max_containers: int = Field(default=2, ge=1, le=2)


class BuildResult(BaseModel):
    """Captured result from a Docker build attempt."""

    success: bool
    exit_code: int
    log: str
    timed_out: bool = False
    duration_sec: float = 0.0


def clone_repo(repo: RepoSpec, cache_dir: Path) -> Path:
    """Clone `repo` into `cache_dir` (if absent) and return the buildable directory path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = cache_dir / repo.id

    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo.url, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        if repo.commit and repo.commit != "VERIFY_SHA":
            subprocess.run(
                ["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", repo.commit],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout", repo.commit],
                check=True,
                capture_output=True,
                text=True,
            )

    target_dir = clone_dir / repo.subdir if repo.subdir else clone_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Configured subdir '{repo.subdir}' not found in {clone_dir}")
    return target_dir


def run_docker_build(
    repo_dir: Path,
    repo: RepoSpec,
    docker_cfg: DockerConfig,
    env_overrides: dict[str, str] | None = None,
) -> BuildResult:
    """Run `install_cmd && build_cmd` inside a memory- and time-bounded Docker container."""
    image = docker_cfg.image_template.format(node_version=repo.node_version)
    shell_cmd = f"{repo.install_cmd} && {repo.build_cmd}"

    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--memory",
        docker_cfg.memory_limit,
        "-v",
        f"{repo_dir.resolve()}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "CI=true",
    ]
    if env_overrides:
        for key, value in env_overrides.items():
            cmd.extend(["-e", f"{key}={value}"])
    cmd.extend([image, "sh", "-c", shell_cmd])

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=docker_cfg.timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = round(time.monotonic() - start, 2)
        combined_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return BuildResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            log=combined_log.strip(),
            timed_out=False,
            duration_sec=elapsed,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - start, 2)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return BuildResult(
            success=False,
            exit_code=124,
            log=f"{stdout}\n{stderr}\n[TIMEOUT after {docker_cfg.timeout_sec}s]".strip(),
            timed_out=True,
            duration_sec=elapsed,
        )


def verify_baseline(repo_dir: Path, repo: RepoSpec, docker_cfg: DockerConfig) -> BuildResult:
    """Confirm the clean repository builds cleanly inside Docker before fault injection."""
    return run_docker_build(repo_dir, repo, docker_cfg)
