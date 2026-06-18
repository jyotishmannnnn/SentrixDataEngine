"""Provenance & integrity (manual Phase 6.3a).

SHA-256 every output file → Merkle tree → sign the root (Ed25519 when the
optional ``cryptography`` dependency is present; otherwise an HMAC-SHA256 local
signature, flagged as unsigned-grade so the QA gate can react).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path

from .versioning import sha256_file


@dataclass
class ProvenanceResult:
    merkle_root: str
    signature: str
    algorithm: str          # "ed25519" | "hmac-sha256"
    signed: bool            # True only for ed25519 (cryptographic signature)
    file_hashes: dict[str, str]
    sidecar_path: Path


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    layer = [bytes.fromhex(h) for h in hashes]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest()
                 for i in range(0, len(layer), 2)]
    return layer[0].hex()


def _sign(root_hex: str) -> tuple[str, str, bool]:
    """Return (signature_hex, algorithm, signed)."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key = Ed25519PrivateKey.generate()
        sig = key.sign(bytes.fromhex(root_hex))
        return sig.hex(), "ed25519", True
    except Exception:
        # deterministic local fallback (NOT a cryptographic signature)
        sig = hmac.new(b"sentrixdataengine-local", bytes.fromhex(root_hex),
                       hashlib.sha256).hexdigest()
        return sig, "hmac-sha256", False


def stamp_provenance(files: list[Path], sidecar_path: Path, *, dataset_id: str,
                     version: str, session_id: str, schema_version: str,
                     source_episode_hashes: list[str],
                     customer_id: str | None = None) -> ProvenanceResult:
    file_hashes = {str(p): sha256_file(p) for p in files if Path(p).is_file()}
    root = merkle_root(sorted(file_hashes.values()))
    signature, algorithm, signed = _sign(root)
    sidecar = {
        "dataset_id": dataset_id, "version": version, "session_id": session_id,
        "schema_version": schema_version, "customer_id": customer_id,
        "source_episode_hashes": source_episode_hashes,
        "merkle_root": root, "signature": signature, "algorithm": algorithm,
        "signed": signed, "file_hashes": file_hashes,
    }
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2, sort_keys=True),
                                  encoding="utf-8")
    return ProvenanceResult(merkle_root=root, signature=signature,
                            algorithm=algorithm, signed=signed,
                            file_hashes=file_hashes, sidecar_path=Path(sidecar_path))
