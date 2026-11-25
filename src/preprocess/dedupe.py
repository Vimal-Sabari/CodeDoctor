"""MinHash near-duplicate removal and per-signature capping."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


def _char_shingles(text: str, k: int = 3) -> set[str]:
    norm = " ".join(text.lower().split())
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _minhash_signature(shingles: set[str], num_perm: int) -> list[int]:
    """Compute a deterministic MinHash signature (pure-Python fallback)."""
    if not shingles:
        return [0] * num_perm
    sig: list[int] = []
    for perm_idx in range(num_perm):
        salt = f"{perm_idx}:".encode("utf-8")
        min_val = min(
            int.from_bytes(hashlib.blake2b(salt + sh.encode("utf-8"), digest_size=8).digest(), "big")
            for sh in shingles
        )
        sig.append(min_val)
    return sig


def _estimated_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def dedupe_records(
    records: list[dict[str, Any]],
    num_perm: int = 128,
    jaccard_threshold: float = 0.85,
    max_per_signature: int = 15,
    shingle_k: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate records using MinHash on signatures + per-signature count cap."""
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]

        use_datasketch = True
    except ImportError:
        use_datasketch = False

    kept: list[dict[str, Any]] = []
    sig_counts: Counter[str] = Counter()
    dropped_cap = 0
    dropped_near_dup = 0

    if use_datasketch:
        lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        for idx, rec in enumerate(records):
            sig_str = rec.get("signature", "")
            if sig_counts[sig_str] >= max_per_signature:
                dropped_cap += 1
                continue

            shingles = _char_shingles(sig_str, k=shingle_k)
            mh = MinHash(num_perm=num_perm)
            for sh in shingles:
                mh.update(sh.encode("utf-8"))

            if sig_counts[sig_str] == 0 and lsh.query(mh):
                dropped_near_dup += 1
                continue

            if sig_counts[sig_str] == 0:
                lsh.insert(f"rec_{idx}", mh)
            sig_counts[sig_str] += 1
            kept.append(rec)
    else:
        seen_minhashes: list[tuple[str, list[int]]] = []
        for rec in records:
            sig_str = rec.get("signature", "")
            if sig_counts[sig_str] >= max_per_signature:
                dropped_cap += 1
                continue

            shingles = _char_shingles(sig_str, k=shingle_k)
            mh_sig = _minhash_signature(shingles, num_perm=num_perm)

            is_near_dup = False
            if sig_counts[sig_str] == 0:
                for seen_str, seen_mh in seen_minhashes:
                    if seen_str != sig_str and _estimated_jaccard(mh_sig, seen_mh) >= jaccard_threshold:
                        is_near_dup = True
                        break

            if is_near_dup:
                dropped_near_dup += 1
                continue

            if sig_counts[sig_str] == 0:
                seen_minhashes.append((sig_str, mh_sig))
            sig_counts[sig_str] += 1
            kept.append(rec)

    stats = {
        "input_count": len(records),
        "kept_count": len(kept),
        "dropped_cap": dropped_cap,
        "dropped_near_dup": dropped_near_dup,
    }
    return kept, stats
