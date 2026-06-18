"""Watermark seam (manual Phase 6.3b/c). Local: identity (records intent only)."""
from __future__ import annotations


def watermark(*, customer_id: str | None, dataset_id: str) -> dict:
    """Return a watermark descriptor. V1 local: no embedding, integrity-only.

    A real implementation injects a per-licensee traitor-tracing fingerprint and
    radioactive-data marking keyed by (customer_id, dataset_id)."""
    return {"applied": False, "kind": "none",
            "note": "integrity-only (Merkle+signature); per-licensee mark deferred to Phase 4"}
