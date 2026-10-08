"""Hashing helpers for manifests, weights, and lockfiles."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
