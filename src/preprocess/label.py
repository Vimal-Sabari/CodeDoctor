"""Heuristic taxonomy pre-labeling for non-synthetic logs and review CSV export."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from src.schema import DEFAULT_TAXONOMY_PATH, ErrorCategory, TaxonomyEntry, load_taxonomy


def compile_taxonomy_patterns(
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> list[tuple[ErrorCategory, list[re.Pattern[str]], str]]:
    """Compile regex signatures from configs/taxonomy.yaml."""
    entries: list[TaxonomyEntry] = load_taxonomy(taxonomy_path)
    compiled: list[tuple[ErrorCategory, list[re.Pattern[str]], str]] = []
    for entry in entries:
        if entry.id == ErrorCategory.UNKNOWN:
            continue
        patterns = [re.compile(pat, re.IGNORECASE) for pat in entry.typical_log_signatures]
        compiled.append((entry.id, patterns, entry.example_fix))
    return compiled


def propose_category(
    log_text: str,
    compiled_taxonomy: list[tuple[ErrorCategory, list[re.Pattern[str]], str]],
) -> tuple[ErrorCategory, float, str]:
    """Propose an error_category using regex match counts against the taxonomy."""
    best_cat = ErrorCategory.UNKNOWN
    best_hits = 0
    best_fix = "Inspect the full CI build log to identify the underlying error."

    for cat, patterns, example_fix in compiled_taxonomy:
        hits = sum(1 for pat in patterns if pat.search(log_text))
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
            best_fix = example_fix

    confidence = min(0.75, 0.45 + 0.15 * best_hits) if best_hits > 0 else 0.2
    return best_cat, confidence, best_fix


def label_unverified_records(
    records: list[dict[str, Any]],
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply heuristic labels to non-synthetic records (verified=False)."""
    compiled = compile_taxonomy_patterns(taxonomy_path)
    all_records: list[dict[str, Any]] = []
    unverified_records: list[dict[str, Any]] = []

    for rec in records:
        if rec.get("source") == "synthetic_fault_injector":
            rec.setdefault("verified", True)
            all_records.append(rec)
            continue

        log_text = rec.get("log_excerpt") or rec.get("raw_log") or ""
        proposed_cat, conf, example_fix = propose_category(log_text, compiled)
        labeled = {
            **rec,
            "error_category": rec.get("error_category") or proposed_cat.value,
            "root_cause": rec.get("root_cause")
            or f"Heuristic match ({proposed_cat.value}) — requires human verification.",
            "fix_steps": rec.get("fix_steps") or [example_fix],
            "confidence": rec.get("confidence", conf),
            "verified": False,
        }
        all_records.append(labeled)
        unverified_records.append(labeled)

    return all_records, unverified_records


def export_unverified_csv(unverified_records: list[dict[str, Any]], csv_path: Path) -> Path:
    """Export unverified non-synthetic records to CSV for manual review."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "repo",
        "framework",
        "proposed_category",
        "confidence",
        "signature",
        "log_excerpt",
        "answer_text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in unverified_records:
            writer.writerow(
                {
                    "id": rec.get("id", ""),
                    "source": rec.get("source", ""),
                    "repo": rec.get("repo", ""),
                    "framework": rec.get("framework", ""),
                    "proposed_category": rec.get("error_category", "unknown"),
                    "confidence": rec.get("confidence", 0.2),
                    "signature": rec.get("signature", ""),
                    "log_excerpt": rec.get("log_excerpt", ""),
                    "answer_text": rec.get("answer_text", ""),
                }
            )
    return csv_path
