"""Tests for the Phase 5 preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.preprocess.clean import clean_log
from src.preprocess.dedupe import dedupe_records
from src.preprocess.redact import redact_secrets
from src.preprocess.run import run_preprocess_pipeline
from src.preprocess.signature import compute_error_signature
from src.preprocess.split import split_by_repo
from src.preprocess.window import default_token_counter, extract_log_window


def test_redactor_never_leaks_planted_secrets() -> None:
    """Redactor must strip GitHub tokens, npm tokens, AWS keys, JWTs, emails, IPs, internal URLs."""
    planted = [
        "ghp_1234567890abcdefghijklmnopqrstuv",
        "npm_abcdefghijklmnopqrstuvwxyz123456",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "devops@example-corp.com",
        "10.24.55.12",
        "https://artifactory.corp.internal:8443/npm/pkg",
        "https://builder:supersecretpass@registry.example.org/pkg",
    ]
    raw_log = "\n".join(f"log line secret={item}" for item in planted)
    redacted = redact_secrets(raw_log)
    for secret in planted:
        assert secret not in redacted, f"Leaked secret: {secret}"


def test_window_respects_token_budget_and_omission_markers() -> None:
    """Window extractor must cap output <= max_tokens and mark omitted lines."""
    lines = [f"info step {i}: compiling module_{i}.ts" for i in range(250)]
    lines[100] = "error TS2307: Cannot find module './missing_component'."
    raw = "\n".join(lines)

    windowed = extract_log_window(
        raw,
        context_before=40,
        context_after=40,
        tail_lines=60,
        max_tokens=300,
        token_counter=default_token_counter,
    )
    assert default_token_counter(windowed) <= 300
    assert "TS2307" in windowed
    assert "lines omitted" in windowed


def test_dedupe_caps_and_near_duplicates() -> None:
    """Dedupe must cap identical signatures and drop near-duplicate signatures."""
    records = [
        {"id": f"a_{i}", "signature": "Cannot find module '<PKG>'"} for i in range(10)
    ] + [
        {"id": "b_1", "signature": "error TS2304: Cannot find name '<STR>'"},
        {"id": "b_2", "signature": "error TS2304: Cannot find name '<STR>'."},
    ]
    kept, stats = dedupe_records(
        records,
        num_perm=64,
        jaccard_threshold=0.80,
        max_per_signature=3,
        shingle_k=3,
    )
    a_kept = [r for r in kept if r["id"].startswith("a_")]
    assert len(a_kept) == 3
    assert stats["dropped_cap"] == 7
    assert stats["dropped_near_dup"] >= 1


def test_split_by_repo_zero_overlap_and_held_out() -> None:
    """Splitter must guarantee zero repo overlap and place held-out repos in test."""
    records = []
    for repo_idx in range(10):
        repo_name = f"org/repo-{repo_idx}"
        for item_idx in range(3):
            records.append({"id": f"r{repo_idx}_{item_idx}", "repo": repo_name})

    train, val, test = split_by_repo(
        records,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        held_out_repos=["org/repo-9"],
    )
    train_repos = {r["repo"] for r in train}
    val_repos = {r["repo"] for r in val}
    test_repos = {r["repo"] for r in test}

    assert "org/repo-9" in test_repos
    assert not (train_repos & val_repos)
    assert not (train_repos & test_repos)
    assert not (val_repos & test_repos)
    assert len(train) + len(val) + len(test) == len(records)


def test_clean_and_signature_preserve_original_and_normalize() -> None:
    """clean_log removes ANSI/timestamps; signature normalizes paths and packages."""
    raw = "\x1b[31m2026-11-25T10:00:00Z Error: Cannot find module 'lodash'\x1b[0m"
    cleaned = clean_log(raw)
    assert "\x1b" not in cleaned
    assert "lodash" in cleaned

    sig = compute_error_signature(cleaned)
    assert "Cannot find module '<PKG>'" in sig


def test_end_to_end_preprocess_pipeline(tmp_path: Path) -> None:
    """End-to-end preprocessing produces splits, review CSV, split report, and data_stats.md."""
    syn_jsonl = tmp_path / "synthetic.jsonl"
    raw_jsonl = tmp_path / "raw" / "gh.jsonl"
    raw_jsonl.parent.mkdir(parents=True)

    syn_records = [
        {
            "id": f"syn_{i}",
            "source": "synthetic_fault_injector",
            "repo": f"org/syn-repo-{i}",
            "framework": "vite-react",
            "package_manager": "npm",
            "log_excerpt": f"Error: Cannot find module 'pkg-{i}'\nBuild failed.",
            "error_category": "dependency_missing",
            "root_cause": "Missing package",
            "fix_steps": ["Install package"],
            "confidence": 1.0,
            "verified": True,
        }
        for i in range(6)
    ]
    syn_jsonl.write_text("\n".join(json.dumps(r) for r in syn_records) + "\n", encoding="utf-8")

    raw_records = [
        {
            "id": "gh_1",
            "source": "github_actions",
            "raw_log": "error TS2304: Cannot find name 'Window'.\nToken=ghp_1234567890abcdefghijklmnopqrstuv",
            "metadata": {"repo": "org/real-repo-1", "framework": "nextjs"},
        },
        {
            "id": "gh_2",
            "source": "github_actions",
            "raw_log": "ERESOLVE unable to resolve dependency tree",
            "metadata": {"repo": "org/real-repo-2", "framework": "vue"},
        },
    ]
    raw_jsonl.write_text("\n".join(json.dumps(r) for r in raw_records) + "\n", encoding="utf-8")

    cfg_path = tmp_path / "preprocess.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "inputs": {
                    "synthetic_jsonl": str(syn_jsonl),
                    "raw_globs": [str(raw_jsonl)],
                },
                "outputs": {
                    "processed_dir": str(tmp_path / "processed"),
                    "train_jsonl": str(tmp_path / "processed" / "train.jsonl"),
                    "val_jsonl": str(tmp_path / "processed" / "val.jsonl"),
                    "test_jsonl": str(tmp_path / "processed" / "test.jsonl"),
                    "unverified_csv": str(tmp_path / "processed" / "unverified_review.csv"),
                    "split_report": str(tmp_path / "reports" / "split_report.md"),
                    "stats_report": str(tmp_path / "reports" / "data_stats.md"),
                },
                "taxonomy_path": str(Path(__file__).resolve().parent.parent / "configs" / "taxonomy.yaml"),
                "window": {
                    "context_before": 40,
                    "context_after": 40,
                    "tail_lines": 60,
                    "max_tokens": 1500,
                    "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
                },
                "dedupe": {
                    "minhash_num_perm": 64,
                    "jaccard_threshold": 0.99,
                    "max_per_signature": 15,
                    "shingle_k": 3,
                },
                "split": {
                    "seed": 42,
                    "train_ratio": 0.8,
                    "val_ratio": 0.1,
                    "test_ratio": 0.1,
                    "held_out_repos": ["org/real-repo-2"],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_preprocess_pipeline(cfg_path)
    assert summary["raw_input_count"] == 8
    assert summary["unverified_count"] == 2
    assert Path(summary["outputs"]["train_jsonl"]).exists()
    assert Path(summary["outputs"]["unverified_csv"]).exists()
    assert Path(summary["outputs"]["stats_report"]).exists()
    assert Path(summary["outputs"]["split_report"]).exists()
    stats_text = Path(summary["outputs"]["stats_report"]).read_text(encoding="utf-8")
    assert "Class Distribution" in stats_text
    assert "Token-Length Histogram" in stats_text
