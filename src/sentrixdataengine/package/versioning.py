"""Dataset identity + reproducibility hashing.

Same Session + same profile + same engine version → identical content hash, so a
build is byte-reproducible and auditable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def derive_dataset_id(session_id: str) -> str:
    """Stable dataset id derived from the originating session id."""
    h = hashlib.sha256(f"sentrix-dataset::{session_id}".encode()).hexdigest()[:12]
    return f"ds_{session_id}_{h}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(files: list[Path]) -> str:
    """Order-independent content hash over a set of files (by their digests)."""
    digests = sorted(sha256_file(p) for p in files if Path(p).is_file())
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode())
    return h.hexdigest()
