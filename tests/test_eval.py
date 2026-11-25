"""Tests for Phase 7 evaluation metrics and runner with MockBackend."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.eval.backends import MockBackend
from src.eval.metrics import compute_metrics
from src.eval.run_eval import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compute_metrics_hand_built_toy_cases() -> None:
    """Verify accuracy, macro-F1, JSON validity, entity match, hallucination, calibration, latency."""
    gold = [
        {
            "id": "g1",
            "error_category": "dependency_missing",
            "log_excerpt": "Error: Cannot find module 'lucide-react'",
            "root_cause": "Missing package 'lucide-react' in package.json.",
            "files_to_change": ["package.json"],
        },
        {
            "id": "g2",
            "error_category": "type_error",
            "log_excerpt": "src/App.tsx:12:5 - error TS2304: Cannot find name 'Foo'.",
            "root_cause": "TypeScript TS2304 in src/App.tsx.",
            "files_to_change": ["src/App.tsx"],
        },
        {
            "id": "g3",
            "error_category": "oom",
            "log_excerpt": "FATAL ERROR: Reached heap limit Allocation failed",
            "root_cause": "Node.js heap memory exhausted.",
            "files_to_change": [],
        },
        {
            "id": "g4",
            "error_category": "lint_error",
            "log_excerpt": "ESLint: 1:1 error Unexpected var",
            "root_cause": "ESLint rule violation.",
            "files_to_change": [],
        },
    ]

    preds = [
        # g1: exact match, grounded evidence, entity matched ('lucide-react')
        json.dumps(
            {
                "error_category": "dependency_missing",
                "root_cause": "Install 'lucide-react' in package.json.",
                "evidence": ["Cannot find module 'lucide-react'"],
                "fix_steps": ["npm install lucide-react"],
                "confidence": 0.90,
            }
        ),
        # g2: wrong category ('unknown' when wrong), hallucinated evidence line, entity matched ('TS2304')
        json.dumps(
            {
                "error_category": "unknown",
                "root_cause": "Saw TS2304 in logs.",
                "evidence": ["This line does not exist in log"],
                "fix_steps": ["Check src/App.tsx"],
                "confidence": 0.20,
            }
        ),
        # g3: invalid JSON
        "NOT VALID JSON {{{",
        # g4: exact match, no hallucination
        json.dumps(
            {
                "error_category": "lint_error",
                "root_cause": "Fix ESLint error.",
                "evidence": ["ESLint: 1:1 error Unexpected var"],
                "fix_steps": ["Use const instead of var"],
                "confidence": 0.80,
            }
        ),
    ]

    metrics = compute_metrics(gold, preds, latencies_sec=[0.1, 0.2, 0.3, 0.4])

    assert metrics["total"] == 4
    assert metrics["category_accuracy"] == 0.5
    assert metrics["json_validity_rate"] == 0.75
    assert metrics["hallucination_rate"] == round(1 / 3, 4)
    assert metrics["root_cause_entity_match"] == 1.0
    assert metrics["unknown_when_wrong_rate"] == 0.5
    assert metrics["calibration"]["0.75-1.00"]["count"] == 2
    assert metrics["calibration"]["0.75-1.00"]["accuracy"] == 1.0
    assert metrics["calibration"]["0.00-0.25"]["count"] == 1
    assert metrics["calibration"]["0.00-0.25"]["accuracy"] == 0.0
    assert metrics["latency"]["mean_sec"] == 0.25
    assert metrics["latency"]["max_sec"] == 0.4
    assert set(metrics["per_class"].keys()) == {"dependency_missing", "type_error", "oom", "lint_error"}


def test_run_evaluation_with_mock_backend(tmp_path: Path) -> None:
    """Verify run_evaluation writes predictions JSONL, metrics JSON, and markdown table."""
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "id": "gold_1",
                "source": "synthetic_fault_injector",
                "framework": "nextjs",
                "package_manager": "npm",
                "log_excerpt": "Error: Cannot find module 'zod'",
                "error_category": "dependency_missing",
                "root_cause": "Missing zod in package.json.",
                "fix_steps": ["npm install zod"],
                "files_to_change": ["package.json"],
                "confidence": 1.0,
                "verified": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "prompts_path": str(PROJECT_ROOT / "configs" / "prompts.yaml"),
                "schema_path": str(PROJECT_ROOT / "schema.json"),
                "reports_dir": str(tmp_path / "reports"),
                "splits": {"gold": str(gold_path)},
                "inference": {"batch_size": 1, "max_new_tokens": 256, "temperature": 0.0, "rag_enabled": False},
                "baselines": {"mock": {"backend": "hf_4bit", "model_id": "mock"}},
            }
        ),
        encoding="utf-8",
    )

    mock = MockBackend(
        json.dumps(
            {
                "error_category": "dependency_missing",
                "root_cause": "Missing 'zod' in package.json.",
                "evidence": ["Error: Cannot find module 'zod'"],
                "fix_steps": ["Run npm install zod"],
                "confidence": 0.92,
            }
        )
    )

    res = run_evaluation(
        config_path=cfg_path,
        model_key="mock",
        split="gold",
        rag=False,
        backend_override=mock,
    )
    assert res["metrics"]["category_accuracy"] == 1.0
    assert res["metrics"]["hallucination_rate"] == 0.0
    assert Path(res["predictions_path"]).exists()
    assert Path(res["metrics_path"]).exists()
    assert "| Category Accuracy | 1.0000 |" in res["markdown_table"]
