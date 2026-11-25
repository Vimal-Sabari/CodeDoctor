"""Evaluation metrics for frontend build-failure root-cause analysis."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from pydantic import ValidationError

from src.schema import ErrorCategory, ModelOutput

ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:TS\d{4}|ERR_[A-Z0-9_]+|ERESOLVE|ENOENT|E404|ETIMEDOUT|SIGKILL)\b"),
    re.compile(r"\b[A-Za-z0-9_.-]+\.(?:tsx?|jsx?|json|ya?ml|css|scss|mjs|cjs)\b"),
    re.compile(r"['\"`](@?[a-z0-9-]+(?:/[a-z0-9._-]+)?)['\"`]"),
]

CONFIDENCE_BUCKETS: list[tuple[str, float, float]] = [
    ("0.00-0.25", 0.0, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-0.75", 0.50, 0.75),
    ("0.75-1.00", 0.75, 1.01),
]


def parse_prediction(raw_text: str) -> ModelOutput | None:
    """Parse raw model output string into ModelOutput, stripping markdown fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        return ModelOutput.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        return None


def extract_entities(text: str) -> set[str]:
    """Extract error codes, filenames, and quoted package names from text."""
    entities: set[str] = set()
    for pat in ENTITY_PATTERNS:
        for match in pat.findall(text):
            token = match.strip()
            if len(token) >= 2:
                entities.add(token.lower())
    return entities


def _is_evidence_grounded(evidence_line: str, log_excerpt: str) -> bool:
    ev_norm = " ".join(evidence_line.split())
    log_norm = " ".join(log_excerpt.split())
    return bool(ev_norm) and ev_norm in log_norm


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(pct * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def compute_metrics(
    gold_records: list[dict[str, Any]],
    raw_predictions: list[str],
    latencies_sec: list[float],
) -> dict[str, Any]:
    """Compute classification, JSON validity, entity match, hallucination, calibration, and latency."""
    if len(gold_records) != len(raw_predictions):
        raise ValueError("gold_records and raw_predictions must have identical length.")

    total = len(gold_records)
    if total == 0:
        return {"total": 0}

    valid_count = 0
    correct_count = 0
    entity_evaluated = 0
    entity_matched = 0
    hallucinated_preds = 0

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()

    bucket_stats: dict[str, dict[str, int]] = {
        label: {"count": 0, "correct": 0} for label, _, _ in CONFIDENCE_BUCKETS
    }
    wrong_count = 0
    unknown_when_wrong = 0
    unknown_total = 0

    for gold, raw_pred in zip(gold_records, raw_predictions):
        gold_cat = str(gold["error_category"])
        support[gold_cat] += 1
        parsed = parse_prediction(raw_pred)

        if parsed is None:
            fn[gold_cat] += 1
            wrong_count += 1
            continue

        valid_count += 1
        pred_cat = parsed.error_category.value
        is_correct = pred_cat == gold_cat

        if pred_cat == ErrorCategory.UNKNOWN.value:
            unknown_total += 1

        if is_correct:
            correct_count += 1
            tp[gold_cat] += 1
        else:
            fp[pred_cat] += 1
            fn[gold_cat] += 1
            wrong_count += 1
            if pred_cat == ErrorCategory.UNKNOWN.value:
                unknown_when_wrong += 1

        # Calibration bucket
        for label, lo, hi in CONFIDENCE_BUCKETS:
            if lo <= parsed.confidence < hi:
                bucket_stats[label]["count"] += 1
                if is_correct:
                    bucket_stats[label]["correct"] += 1
                break

        # Root-cause entity match
        gold_entity_source = " ".join(
            [str(gold.get("root_cause", ""))] + [str(f) for f in gold.get("files_to_change", [])]
        )
        gold_entities = extract_entities(gold_entity_source)
        if gold_entities:
            entity_evaluated += 1
            pred_text = " ".join([parsed.root_cause] + parsed.fix_steps).lower()
            if any(ent in pred_text for ent in gold_entities):
                entity_matched += 1

        # Hallucination rate: any cited evidence line missing from log_excerpt
        log_excerpt = str(gold.get("log_excerpt", ""))
        if parsed.evidence and any(
            not _is_evidence_grounded(ev, log_excerpt) for ev in parsed.evidence
        ):
            hallucinated_preds += 1

    # Per-class report and macro-F1 over classes present in gold
    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []
    for cat in sorted(support.keys()):
        c_tp = tp[cat]
        c_fp = fp[cat]
        c_fn = fn[cat]
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[cat],
        }

    calibration = {
        label: {
            "count": vals["count"],
            "accuracy": round(vals["correct"] / vals["count"], 4) if vals["count"] else 0.0,
        }
        for label, vals in bucket_stats.items()
    }

    sorted_lat = sorted(latencies_sec)
    latency_summary = {
        "mean_sec": round(sum(sorted_lat) / len(sorted_lat), 4) if sorted_lat else 0.0,
        "p50_sec": round(_percentile(sorted_lat, 0.50), 4),
        "p95_sec": round(_percentile(sorted_lat, 0.95), 4),
        "max_sec": round(sorted_lat[-1], 4) if sorted_lat else 0.0,
    }

    return {
        "total": total,
        "category_accuracy": round(correct_count / total, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "json_validity_rate": round(valid_count / total, 4),
        "root_cause_entity_match": (
            round(entity_matched / entity_evaluated, 4) if entity_evaluated else 0.0
        ),
        "hallucination_rate": round(hallucinated_preds / valid_count, 4) if valid_count else 0.0,
        "unknown_rate": round(unknown_total / valid_count, 4) if valid_count else 0.0,
        "unknown_when_wrong_rate": (
            round(unknown_when_wrong / wrong_count, 4) if wrong_count else 0.0
        ),
        "calibration": calibration,
        "latency": latency_summary,
        "per_class": per_class,
    }
