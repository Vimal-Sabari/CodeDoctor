"""Log cleaning and normalization utilities."""

from __future__ import annotations

import re

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\]|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+"
)
PROGRESS_BAR_RE = re.compile(r"^\s*(?:[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]|\[[=#>\-\s]{4,}\]|\d{1,3}%\|).+$")

UNIX_PATH_RE = re.compile(r"/(?:Users|home|workspace|tmp|var|opt|mnt)/[^\s:'\"]+")
WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s:'\"]+")
HEX_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
SEMVER_RE = re.compile(r"\bv?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?\b")


def strip_ansi_and_timestamps(text: str) -> str:
    """Remove ANSI escape sequences and leading timestamps, resolving carriage returns."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        # Keep only the final segment after '\r' to collapse progress-bar overwrites
        if "\r" in raw_line:
            raw_line = raw_line.rsplit("\r", 1)[-1]
        cleaned = ANSI_ESCAPE_RE.sub("", raw_line)
        cleaned = TIMESTAMP_PREFIX_RE.sub("", cleaned).rstrip()
        if PROGRESS_BAR_RE.match(cleaned):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def collapse_repeated_lines(text: str, min_repeat: int = 3) -> str:
    """Collapse consecutive identical lines into a single line with a repeat count."""
    lines = text.splitlines()
    if not lines:
        return ""

    out: list[str] = []
    prev = lines[0]
    count = 1

    def _flush(line: str, n: int) -> None:
        out.append(line)
        if n >= min_repeat:
            out.append(f"[... repeated {n - 1} more times ...]")
        elif n > 1:
            for _ in range(n - 1):
                out.append(line)

    for line in lines[1:]:
        if line == prev:
            count += 1
        else:
            _flush(prev, count)
            prev = line
            count = 1
    _flush(prev, count)
    return "\n".join(out)


def clean_log(text: str) -> str:
    """Clean a raw build log while keeping the log text intact (no path/version normalization)."""
    stripped = strip_ansi_and_timestamps(text)
    return collapse_repeated_lines(stripped)


def normalize_for_signature(text: str) -> str:
    """Normalize absolute paths, hex hashes, and semver versions for signature generation."""
    norm = UNIX_PATH_RE.sub("<PATH>", text)
    norm = WIN_PATH_RE.sub("<PATH>", norm)
    norm = SEMVER_RE.sub("<VER>", norm)
    norm = HEX_HASH_RE.sub("<HASH>", norm)
    return norm
