"""GitHub Actions failed workflow log collector with rate-limit backoff and caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from pydantic import BaseModel, Field

from src.collect.common import (
    DEFAULT_COLLECT_CONFIG,
    RawLogRecord,
    append_raw_records,
    load_collect_config,
    resolve_project_path,
)

if TYPE_CHECKING:
    import typer


class GitHubCollectConfig(BaseModel):
    """Configuration for GitHub Actions log collection."""

    queries: list[str]
    min_stars: int = 500
    permissive_licenses: list[str]
    max_repos: int = 20
    max_runs_per_repo: int = 5
    output_jsonl: Path
    cache_dir: Path
    manifest_path: Path
    request_timeout_sec: int = 30
    max_retries: int = 5
    api_base_url: str = "https://api.github.com"


@dataclass(frozen=True)
class HttpResponse:
    """Minimal HTTP response container."""

    status_code: int
    headers: dict[str, str]
    text: str


class HttpClient(Protocol):
    """Protocol for injectable HTTP GET clients (enables offline fixture testing)."""

    def get(self, url: str, headers: dict[str, str], timeout: int) -> HttpResponse:
        """Perform an HTTP GET request and return an HttpResponse."""


class UrllibHttpClient:
    """Standard-library HTTP client using urllib.request."""

    def get(self, url: str, headers: dict[str, str], timeout: int) -> HttpResponse:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return HttpResponse(status_code=resp.status, headers=resp_headers, text=body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            resp_headers = {k.lower(): v for k, v in exc.headers.items()}
            return HttpResponse(status_code=exc.code, headers=resp_headers, text=body)


def _load_github_token() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.getenv("GITHUB_TOKEN") or None


def _build_config(raw_cfg: dict[str, Any]) -> GitHubCollectConfig:
    gh = raw_cfg["github_actions"]
    return GitHubCollectConfig(
        queries=gh["queries"],
        min_stars=gh["min_stars"],
        permissive_licenses=gh["permissive_licenses"],
        max_repos=gh["max_repos"],
        max_runs_per_repo=gh["max_runs_per_repo"],
        output_jsonl=resolve_project_path(gh["output_jsonl"]),
        cache_dir=resolve_project_path(gh["cache_dir"]),
        manifest_path=resolve_project_path(gh["manifest_path"]),
        request_timeout_sec=gh.get("request_timeout_sec", 30),
        max_retries=gh.get("max_retries", 5),
        api_base_url=gh.get("api_base_url", "https://api.github.com").rstrip("/"),
    )


def _compute_backoff_seconds(headers: dict[str, str], attempt: int) -> float:
    lower = {k.lower(): v for k, v in headers.items()}
    if "retry-after" in lower:
        try:
            return max(1.0, float(lower["retry-after"]))
        except ValueError:
            pass
    if lower.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in lower:
        try:
            reset_ts = float(lower["x-ratelimit-reset"])
            wait = max(1.0, reset_ts - time.time() + 1.0)
            return min(wait, 900.0)
        except ValueError:
            pass
    return float(min(2**attempt, 60))


def _get_with_cache_and_backoff(
    client: HttpClient,
    url: str,
    headers: dict[str, str],
    cache_file: Path,
    timeout: int,
    max_retries: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    last_status = 0
    for attempt in range(max_retries):
        resp = client.get(url, headers=headers, timeout=timeout)
        last_status = resp.status_code
        if resp.status_code == 200:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text, encoding="utf-8")
            return resp.text

        if resp.status_code in (403, 429, 500, 502, 503):
            wait_sec = _compute_backoff_seconds(resp.headers, attempt)
            sleep_fn(wait_sec)
            continue

        raise RuntimeError(f"GitHub API request failed ({resp.status_code}) for {url}")

    raise RuntimeError(
        f"GitHub API request exhausted {max_retries} retries (last status={last_status}) for {url}"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed_jobs": []}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def collect_github_actions(
    cfg: GitHubCollectConfig,
    client: HttpClient | None = None,
    token: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Search permissive frontend repos and download failed GitHub Actions job logs."""
    http = client or UrllibHttpClient()
    auth_token = token if token is not None else _load_github_token()

    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LaunchForge-AI-Collector",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    manifest = _load_manifest(cfg.manifest_path)
    processed_jobs: set[str] = set(manifest.get("processed_jobs", []))
    allowed_licenses = set(cfg.permissive_licenses)

    seen_repos: dict[str, dict[str, Any]] = {}
    skipped_unlicensed = 0

    for query in cfg.queries:
        full_q = f"{query} stars:>={cfg.min_stars}"
        encoded_q = urllib.parse.quote(full_q, safe="")
        url = f"{cfg.api_base_url}/search/repositories?q={encoded_q}&sort=stars&order=desc&per_page=20"
        q_hash = hashlib.sha256(full_q.encode("utf-8")).hexdigest()[:12]
        raw_text = _get_with_cache_and_backoff(
            http,
            url,
            headers,
            cfg.cache_dir / f"search_{q_hash}.json",
            cfg.request_timeout_sec,
            cfg.max_retries,
            sleep_fn=sleep_fn,
        )
        items = json.loads(raw_text).get("items", [])
        for repo in items:
            full_name = repo["full_name"]
            if full_name in seen_repos:
                continue
            license_info = repo.get("license") or {}
            spdx = license_info.get("spdx_id")
            if not spdx or spdx not in allowed_licenses:
                skipped_unlicensed += 1
                continue
            seen_repos[full_name] = repo
            if len(seen_repos) >= cfg.max_repos:
                break
        if len(seen_repos) >= cfg.max_repos:
            break

    new_records: list[RawLogRecord] = []
    for full_name, repo in seen_repos.items():
        safe_repo = full_name.replace("/", "__")
        runs_url = (
            f"{cfg.api_base_url}/repos/{full_name}/actions/runs"
            f"?status=failure&per_page={cfg.max_runs_per_repo}"
        )
        runs_text = _get_with_cache_and_backoff(
            http,
            runs_url,
            headers,
            cfg.cache_dir / f"runs_{safe_repo}.json",
            cfg.request_timeout_sec,
            cfg.max_retries,
            sleep_fn=sleep_fn,
        )
        runs = json.loads(runs_text).get("workflow_runs", [])[: cfg.max_runs_per_repo]

        for run in runs:
            run_id = run["id"]
            jobs_url = f"{cfg.api_base_url}/repos/{full_name}/actions/runs/{run_id}/jobs"
            jobs_text = _get_with_cache_and_backoff(
                http,
                jobs_url,
                headers,
                cfg.cache_dir / f"jobs_{safe_repo}_{run_id}.json",
                cfg.request_timeout_sec,
                cfg.max_retries,
                sleep_fn=sleep_fn,
            )
            for job in json.loads(jobs_text).get("jobs", []):
                if job.get("conclusion") != "failure":
                    continue
                job_key = f"{full_name}:{job['id']}"
                if job_key in processed_jobs:
                    continue

                log_url = f"{cfg.api_base_url}/repos/{full_name}/actions/jobs/{job['id']}/logs"
                try:
                    log_text = _get_with_cache_and_backoff(
                        http,
                        log_url,
                        headers,
                        cfg.cache_dir / f"job_{safe_repo}_{job['id']}.log",
                        cfg.request_timeout_sec,
                        cfg.max_retries,
                        sleep_fn=sleep_fn,
                    )
                except RuntimeError:
                    continue

                if not log_text.strip():
                    continue

                new_records.append(
                    RawLogRecord(
                        id=f"gh_{safe_repo}_{job['id']}",
                        source="github_actions",
                        raw_log=log_text,
                        title=f"{run.get('name', 'workflow')} / {job.get('name', 'job')}",
                        metadata={
                            "repo": full_name,
                            "stars": repo.get("stargazers_count"),
                            "license": (repo.get("license") or {}).get("spdx_id"),
                            "run_id": run_id,
                            "job_id": job["id"],
                            "job_name": job.get("name"),
                            "workflow_name": run.get("name"),
                            "html_url": job.get("html_url") or run.get("html_url"),
                        },
                    )
                )
                processed_jobs.add(job_key)

    written = append_raw_records(cfg.output_jsonl, new_records)
    manifest["processed_jobs"] = sorted(processed_jobs)
    _save_manifest(cfg.manifest_path, manifest)

    return {
        "repos_matched": len(seen_repos),
        "repos_skipped_license": skipped_unlicensed,
        "records_written": written,
        "output_jsonl": str(cfg.output_jsonl),
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="github-actions-collect",
        help="Collect failed GitHub Actions build logs from permissive frontend repos.",
        add_completion=False,
    )

    @cli.command("collect")
    def collect_cmd(
        config: Path = typer.Option(DEFAULT_COLLECT_CONFIG, "--config", "-c"),
    ) -> None:
        """Run GitHub Actions log collection using configs/collect.yaml."""
        cfg = _build_config(load_collect_config(config))
        summary = collect_github_actions(cfg)
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
