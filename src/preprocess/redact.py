"""Secret and PII redaction using detect-secrets (optional) + regex patterns."""

from __future__ import annotations

import re

REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"), "<REDACTED_NPM_TOKEN>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS_KEY>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "<REDACTED_JWT>"),
    (
        re.compile(r"(?i)\b(authorization|bearer|token|api[_-]?key|secret|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{12,})['\"]?"),
        r"\1=<REDACTED_SECRET>",
    ),
    (re.compile(r"https?://[^:\s/@]+:[^@\s/]+@[^\s]+"), "<REDACTED_AUTH_URL>"),
    (
        re.compile(r"https?://(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|[^\s/]*\.(?:internal|local|corp|lan|intranet))(?::\d+)?[^\s'\"]*"),
        "<REDACTED_INTERNAL_URL>",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<REDACTED_EMAIL>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<REDACTED_IP>"),
]


def _redact_with_detect_secrets(text: str) -> str:
    """Run detect-secrets line-by-line if installed, replacing flagged secret values."""
    try:
        from detect_secrets.core.scan import scan_line  # type: ignore[import-not-found]
    except ImportError:
        return text

    out_lines: list[str] = []
    for line in text.splitlines():
        secrets = list(scan_line(line))
        redacted_line = line
        for secret in secrets:
            val = getattr(secret, "secret_value", None)
            if val and isinstance(val, str):
                redacted_line = redacted_line.replace(val, "<REDACTED_SECRET>")
        out_lines.append(redacted_line)
    return "\n".join(out_lines)


def redact_secrets(text: str) -> str:
    """Redact tokens, emails, IPs, internal URLs, and detect-secrets matches."""
    if not text:
        return ""
    redacted = _redact_with_detect_secrets(text)
    for pattern, replacement in REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
