"""Dataset statistics and markdown report generator (reports/data_stats.md)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.preprocess.window import default_token_counter

TOKEN_BUCKETS: list[tuple[str, int, int]] = [
    ("0-250", 0, 250),
    ("251-500", 251, 500),
    ("501-750", 501, 750),
    ("751-1000", 751, 1000),
    ("1001-1250", 1001, 1250),
    ("1251-1500", 1251, 1500),
    (">1500", 1501, 10**9),
]


def write_data_stats_report(
    report_path: Path,
    records: list[dict[str, Any]],
    token_counter: Callable[[str], int] | None = None,
) -> Path:
    """Generate reports/data_stats.md with class, token-length, source, and framework stats."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    count_tokens = token_counter or default_token_counter

    cat_counts: Counter[str] = Counter(str(r.get("error_category", "unknown")) for r in records)
    source_counts: Counter[str] = Counter(str(r.get("source", "unknown")) for r in records)
    fw_counts: Counter[str] = Counter(str(r.get("framework", "unknown")) for r in records)

    token_lengths = [count_tokens(str(r.get("log_excerpt", ""))) for r in records]
    bucket_counts: dict[str, int] = {label: 0 for label, _, _ in TOKEN_BUCKETS}
    for length in token_lengths:
        for label, lo, hi in TOKEN_BUCKETS:
            if lo <= length <= hi:
                bucket_counts[label] += 1
                break

    total = len(records)
    lines = [
        "# LaunchForge AI — Preprocessed Dataset Statistics",
        "",
        f"- **Total records:** {total}",
        "",
        "## Class Distribution (`error_category`)",
        "",
        "| Category | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for cat, cnt in cat_counts.most_common():
        pct = (cnt / total * 100.0) if total else 0.0
        lines.append(f"| `{cat}` | {cnt} | {pct:.1f}% |")

    lines.extend(
        [
            "",
            "## Token-Length Histogram",
            "",
            "| Bucket | Count | Bar |",
            "| --- | ---: | --- |",
        ]
    )
    max_b = max(bucket_counts.values(), default=1) or 1
    for label, _, _ in TOKEN_BUCKETS:
        cnt = bucket_counts[label]
        bar = "#" * int(cnt / max_b * 20) if cnt else ""
        lines.append(f"| `{label}` | {cnt} | `{bar}` |")

    lines.extend(["", "## Per-Source Counts", "", "| Source | Count |", "| --- | ---: |"])
    for src, cnt in source_counts.most_common():
        lines.append(f"| `{src}` | {cnt} |")

    lines.extend(["", "## Per-Framework Counts", "", "| Framework | Count |", "| --- | ---: |"])
    for fw, cnt in fw_counts.most_common():
        lines.append(f"| `{fw}` | {cnt} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
