"""Tests for Phase 6 gold tool, dataset quality gates, and chat SFT converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.preprocess.gold_tool import append_gold_record, sample_stratified
from src.preprocess.to_chat import convert_jsonl_to_chat
from src.preprocess.validate import DatasetValidationError, validate_splits
from src.schema import DatasetRecord, ErrorCategory, ModelOutput

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = PROJECT_ROOT / "configs" / "prompts.yaml"


def _make_record(
    rec_id: str,
    repo: str,
    category: str = "dependency_missing",
    source: str = "synthetic_fault_injector",
    log_excerpt: str = "Error: Cannot find module 'react'",
) -> dict:
    return {
        "id": rec_id,
        "source": source,
        "repo": repo,
        "framework": "vite-react",
        "package_manager": "npm",
        "log_excerpt": log_excerpt,
        "error_category": category,
        "root_cause": "Missing dependency",
        "evidence_lines": [log_excerpt.splitlines()[0]],
        "fix_steps": ["Install dependency"],
        "files_to_change": ["package.json"],
        "confidence": 1.0,
        "verified": True,
    }


def test_gold_tool_stratified_sampling_and_resume(tmp_path: Path) -> None:
    """Verify stratified sampling includes all categories, real logs, and resumes cleanly."""
    test_records = [
        _make_record("syn_1", "org/r1", category="type_error", source="synthetic_fault_injector"),
        _make_record("syn_2", "org/r2", category="type_error", source="synthetic_fault_injector"),
        _make_record("real_1", "org/r3", category="type_error", source="github_actions"),
        _make_record("syn_3", "org/r4", category="oom", source="synthetic_fault_injector"),
        _make_record("real_2", "org/r5", category="lint_error", source="stackexchange"),
    ]
    sample = sample_stratified(test_records, sample_size=3, seed=42)
    assert len(sample) == 3
    sampled_cats = {r["error_category"] for r in sample}
    assert sampled_cats == {"type_error", "oom", "lint_error"}
    assert any(r["id"] == "real_1" for r in sample)
    assert any(r["id"] == "real_2" for r in sample)

    gold_path = tmp_path / "gold.jsonl"
    append_gold_record(gold_path, DatasetRecord.model_validate({**sample[0], "verified": True}))
    resumed = sample_stratified(test_records, sample_size=3, seed=42, exclude_ids={sample[0]["id"]})
    assert all(r["id"] != sample[0]["id"] for r in resumed)


def test_validate_splits_passes_on_clean_balanced_data() -> None:
    """Clean, balanced, zero-overlap splits pass all quality gates."""
    cats = ["dependency_missing", "type_error", "lint_error", "oom", "syntax_error"]
    train = [_make_record(f"tr_{i}", f"org/tr-{i}", category=cats[i % len(cats)]) for i in range(10)]
    val = [_make_record("va_1", "org/va-1", category="type_error")]
    test = [_make_record("te_1", "org/te-1", category="oom")]
    gold = [_make_record("te_1", "org/te-1", category="oom")]

    summary = validate_splits(train, val, test, gold=gold, max_class_share=0.20, max_tokens=1500)
    assert summary["status"] == "PASS"


def test_validate_splits_fails_loudly_on_each_gate() -> None:
    """Verify every hard quality gate raises DatasetValidationError."""
    cats = ["dependency_missing", "type_error", "lint_error", "oom", "syntax_error"]
    base_train = [_make_record(f"tr_{i}", f"org/tr-{i}", category=cats[i]) for i in range(5)]

    # 1. Repo overlap
    with pytest.raises(DatasetValidationError, match="Repo overlap"):
        validate_splits(base_train, [_make_record("va_1", "org/tr-0")], [])

    # 2. Class > 20% of train
    imbalanced = [_make_record(f"tr_{i}", f"org/tr-{i}", category="type_error") for i in range(5)]
    with pytest.raises(DatasetValidationError, match="exceeding max_class_share"):
        validate_splits(imbalanced, [], [])

    # 3. Schema invalid
    bad_schema = [{**base_train[0], "confidence": 2.5}] + base_train[1:]
    with pytest.raises(DatasetValidationError, match="Schema validation failed"):
        validate_splits(bad_schema, [], [])

    # 4. Secret leaked
    leaked_secret = [
        _make_record(
            "tr_0",
            "org/tr-0",
            category=cats[0],
            log_excerpt="Token=ghp_1234567890abcdefghijklmnopqrstuv",
        )
    ] + base_train[1:]
    with pytest.raises(DatasetValidationError, match="Potential secret detected"):
        validate_splits(leaked_secret, [], [])

    # 5. Gold contains train ID
    with pytest.raises(DatasetValidationError, match="Gold set contains train-split"):
        validate_splits(base_train, [], [], gold=[base_train[0]])

    # 6. Token-length cap exceeded
    long_log = " ".join(["token"] * 50)
    with pytest.raises(DatasetValidationError, match="max_tokens"):
        validate_splits(base_train, [], [_make_record("te_1", "org/te-1", log_excerpt=long_log)], max_tokens=20)


def test_to_chat_produces_strict_model_output_json(tmp_path: Path) -> None:
    """Verify to_chat.py produces system/user/assistant messages with valid ModelOutput JSON."""
    in_jsonl = tmp_path / "train.jsonl"
    out_jsonl = tmp_path / "train_chat.jsonl"
    rec = _make_record("tr_1", "org/r1", category=ErrorCategory.DEPENDENCY_MISSING.value)
    in_jsonl.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    cnt = convert_jsonl_to_chat(in_jsonl, out_jsonl, PROMPTS_PATH)
    assert cnt == 1

    chat = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "vite-react" in chat["messages"][1]["content"]

    parsed_target = ModelOutput.model_validate_json(chat["messages"][2]["content"])
    assert parsed_target.error_category == ErrorCategory.DEPENDENCY_MISSING
