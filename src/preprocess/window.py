"""Token-budgeted log window extraction around the first error line + tail."""

from __future__ import annotations

import re
from typing import Callable

ERROR_ANCHOR_RE = re.compile(
    r"(?:error TS\d{4}:|Error:|ERR!|FATAL ERROR|Failed to compile|"
    r"SyntaxError:|ReferenceError:|TypeError:|Cannot find module|ERESOLVE|"
    r"Module not found|ERR_PNPM_|Exit code \d+)",
    re.IGNORECASE,
)


def default_token_counter(text: str) -> int:
    """Lightweight deterministic fallback token counter when HF tokenizer is unavailable."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def load_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Load the configured Qwen tokenizer (local cache only) or fall back cleanly."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception:  # noqa: BLE001
        return default_token_counter


def find_first_error_index(lines: list[str]) -> int:
    """Return index of the first line matching a build error anchor, or 0 if none found."""
    for idx, line in enumerate(lines):
        if ERROR_ANCHOR_RE.search(line):
            return idx
    return 0


def _build_window_text(
    lines: list[str],
    err_idx: int,
    before: int,
    after: int,
    tail: int,
) -> str:
    n = len(lines)
    if n == 0:
        return ""

    head_start = max(0, err_idx - before)
    head_end = min(n, err_idx + after + 1)
    tail_start = max(0, n - tail)

    selected_indices: set[int] = set(range(head_start, head_end)) | set(range(tail_start, n))
    sorted_indices = sorted(selected_indices)

    out_lines: list[str] = []
    if sorted_indices[0] > 0:
        out_lines.append(f"[... {sorted_indices[0]} lines omitted ...]")

    prev_idx = sorted_indices[0] - 1
    for idx in sorted_indices:
        gap = idx - prev_idx - 1
        if gap > 0:
            out_lines.append(f"[... {gap} lines omitted ...]")
        out_lines.append(lines[idx])
        prev_idx = idx

    return "\n".join(out_lines)


def extract_log_window(
    text: str,
    context_before: int = 40,
    context_after: int = 40,
    tail_lines: int = 60,
    max_tokens: int = 1500,
    token_counter: Callable[[str], int] | None = None,
) -> str:
    """Extract an ordered log window capped at `max_tokens` with omission markers."""
    count_tokens = token_counter or default_token_counter
    lines = text.splitlines()
    if not lines:
        return ""

    err_idx = find_first_error_index(lines)
    window = _build_window_text(lines, err_idx, context_before, context_after, tail_lines)
    if count_tokens(window) <= max_tokens:
        return window

    # Progressively shrink context around the anchor and tail until within max_tokens
    cur_before, cur_after, cur_tail = context_before, context_after, tail_lines
    while count_tokens(window) > max_tokens and (cur_before > 2 or cur_after > 5 or cur_tail > 10):
        cur_before = max(2, cur_before // 2)
        cur_after = max(5, cur_after // 2)
        cur_tail = max(10, cur_tail // 2)
        window = _build_window_text(lines, err_idx, cur_before, cur_after, cur_tail)

    if count_tokens(window) <= max_tokens:
        return window

    # Final hard line-by-line trim from the end if a single huge block still exceeds max_tokens
    win_lines = window.splitlines()
    while len(win_lines) > 1 and count_tokens("\n".join(win_lines)) > max_tokens:
        win_lines.pop()

    result = "\n".join(win_lines)
    if count_tokens(result) > max_tokens:
        # Character-level safety clamp for pathological single-line logs
        words = result.split()
        while words and count_tokens(" ".join(words)) > max_tokens:
            words.pop()
        result = " ".join(words)
    return result
