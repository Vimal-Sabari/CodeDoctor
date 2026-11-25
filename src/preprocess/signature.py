"""Normalized error signature extraction for deduplication and RAG indexing."""

from __future__ import annotations

import hashlib
import re

from src.preprocess.clean import normalize_for_signature
from src.preprocess.window import find_first_error_index

QUOTED_PKG_RE = re.compile(r"Cannot find module '[^']+'")
TS_CODE_RE = re.compile(r"error TS\d{4}:")
QUOTED_SINGLE_RE = re.compile(r"'[^'\n]{2,80}'")
QUOTED_DOUBLE_RE = re.compile(r'"[^"\n]{2,80}"')
LINE_COL_RE = re.compile(r":\d+:\d+")


def compute_error_signature(log_text: str) -> str:
    """Compute a normalized error signature from the primary error line(s)."""
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return "EMPTY_LOG"

    err_idx = find_first_error_index(lines)
    anchor_slice = " | ".join(lines[err_idx : min(len(lines), err_idx + 2)])
    norm = normalize_for_signature(anchor_slice)
    norm = LINE_COL_RE.sub(":<LINE>:<COL>", norm)
    norm = QUOTED_SINGLE_RE.sub("'<STR>'", norm)
    norm = QUOTED_DOUBLE_RE.sub('"<STR>"', norm)
    norm = QUOTED_PKG_RE.sub("Cannot find module '<PKG>'", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm[:240]


def signature_hash(signature: str) -> str:
    """Return a short deterministic hex digest for a normalized signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
