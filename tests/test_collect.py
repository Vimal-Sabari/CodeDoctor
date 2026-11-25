"""Offline fixture tests for Phase 4 real-world data collection scripts."""

from __future__ import annotations

import json
from pathlib import Path

from src.collect.github_actions import (
    GitHubCollectConfig,
    HttpResponse,
    collect_github_actions,
)
from src.collect.public_datasets import LogChunksLoader, import_public_dataset
from src.collect.stackexchange import StackExchangeConfig, parse_stackexchange_dump


class FakeGitHubClient:
    """Recorded-fixture HTTP client with simulated rate-limit 429 on first request."""

    def __init__(self, responses: dict[str, list[HttpResponse]]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, headers: dict[str, str], timeout: int) -> HttpResponse:
        self.calls.append(url)
        for key, queue in self._responses.items():
            if key in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        return HttpResponse(status_code=404, headers={}, text="not found")


def test_github_actions_collector_license_backoff_and_resume(tmp_path: Path) -> None:
    """Verify permissive license filter, 429 backoff, caching, and resumability."""
    search_payload = json.dumps(
        {
            "items": [
                {
                    "full_name": "org/mit-vite-app",
                    "stargazers_count": 1200,
                    "license": {"spdx_id": "MIT"},
                },
                {
                    "full_name": "org/gpl-app",
                    "stargazers_count": 2000,
                    "license": {"spdx_id": "GPL-3.0"},
                },
                {
                    "full_name": "org/unlicensed-app",
                    "stargazers_count": 900,
                    "license": None,
                },
            ]
        }
    )
    runs_payload = json.dumps(
        {
            "workflow_runs": [
                {
                    "id": 101,
                    "name": "CI",
                    "html_url": "https://github.com/org/mit-vite-app/actions/runs/101",
                }
            ]
        }
    )
    jobs_payload = json.dumps(
        {
            "jobs": [
                {
                    "id": 5001,
                    "name": "build",
                    "conclusion": "failure",
                    "html_url": "https://github.com/org/mit-vite-app/actions/runs/101/jobs/5001",
                },
                {
                    "id": 5002,
                    "name": "lint",
                    "conclusion": "success",
                },
            ]
        }
    )

    client = FakeGitHubClient(
        {
            "/search/repositories": [
                HttpResponse(status_code=429, headers={"Retry-After": "2"}, text="rate limit"),
                HttpResponse(status_code=200, headers={}, text=search_payload),
            ],
            "/actions/runs?": [
                HttpResponse(status_code=200, headers={}, text=runs_payload),
            ],
            "/actions/runs/101/jobs": [
                HttpResponse(status_code=200, headers={}, text=jobs_payload),
            ],
            "/actions/jobs/5001/logs": [
                HttpResponse(
                    status_code=200,
                    headers={},
                    text="error TS2304: Cannot find name 'ViteEnv'.\nProcess completed with exit code 2.",
                ),
            ],
        }
    )

    sleeps: list[float] = []
    cfg = GitHubCollectConfig(
        queries=["language:TypeScript topic:vite"],
        min_stars=500,
        permissive_licenses=["MIT", "Apache-2.0"],
        max_repos=5,
        max_runs_per_repo=2,
        output_jsonl=tmp_path / "raw" / "github_actions.jsonl",
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "raw" / "manifest.json",
    )

    summary = collect_github_actions(cfg, client=client, token="test-token", sleep_fn=sleeps.append)
    assert summary["repos_matched"] == 1
    assert summary["repos_skipped_license"] == 2
    assert summary["records_written"] == 1
    assert sleeps == [2.0]

    lines = cfg.output_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["source"] == "github_actions"
    assert record["metadata"]["license"] == "MIT"
    assert record["metadata"]["repo"] == "org/mit-vite-app"

    # Second run should resume cleanly and write 0 duplicate records
    summary_2 = collect_github_actions(cfg, client=client, token="test-token", sleep_fn=sleeps.append)
    assert summary_2["records_written"] == 0
    assert len(cfg.output_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_stackexchange_dump_parser_preserves_attribution(tmp_path: Path) -> None:
    """Verify streaming XML parsing, build-error filtering, and CC BY-SA 4.0 attribution."""
    posts_xml = tmp_path / "Posts.xml"
    posts_xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="10" PostTypeId="1" AcceptedAnswerId="11" Score="12" Tags="&lt;vite&gt;&lt;typescript&gt;" Title="Vite build fails with TS2307" Body="&lt;p&gt;error TS2307: Cannot find module './App'&lt;/p&gt;" OwnerDisplayName="alice" />
  <row Id="11" PostTypeId="2" ParentId="10" Score="18" Body="&lt;p&gt;Check case sensitivity in your import path.&lt;/p&gt;" OwnerDisplayName="bob" />
  <row Id="12" PostTypeId="1" AcceptedAnswerId="13" Score="20" Tags="&lt;css&gt;" Title="How to center a div" Body="&lt;p&gt;How do I center a div?&lt;/p&gt;" OwnerDisplayName="carol" />
</posts>
""",
        encoding="utf-8",
    )

    cfg = StackExchangeConfig(
        tags=["vite", "webpack", "npm", "next.js", "typescript"],
        min_score=5,
        require_accepted_answer=True,
        license="CC BY-SA 4.0",
        site_base_url="https://stackoverflow.com/questions",
        output_jsonl=tmp_path / "raw" / "stackexchange.jsonl",
    )

    summary = parse_stackexchange_dump(posts_xml, cfg)
    assert summary["questions_matched"] == 1
    assert summary["records_written"] == 1

    rec = json.loads(cfg.output_jsonl.read_text(encoding="utf-8").strip())
    assert rec["source"] == "stackexchange"
    assert rec["metadata"]["license"] == "CC BY-SA 4.0"
    assert rec["metadata"]["post_url"] == "https://stackoverflow.com/questions/10"
    assert rec["metadata"]["author"] == "alice"
    assert rec["metadata"]["answer_author"] == "bob"
    assert "case sensitivity" in rec["answer_text"]


def test_public_dataset_loader_stub(tmp_path: Path) -> None:
    """Verify optional LogChunksLoader imports into unified RawLogRecord JSONL."""
    src_file = tmp_path / "logchunks_sample.jsonl"
    src_file.write_text(
        json.dumps(
            {
                "id": "lc-1",
                "repo": "example/frontend",
                "language": "TypeScript",
                "log": "npm ERR! ERESOLVE unable to resolve dependency tree",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "raw" / "public_ci_logs.jsonl"
    summary = import_public_dataset(src_file, out_jsonl, LogChunksLoader())
    assert summary["records_loaded"] == 1
    assert summary["records_written"] == 1
    rec = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert rec["source"] == "logchunks"
    assert rec["id"] == "lc-1"
