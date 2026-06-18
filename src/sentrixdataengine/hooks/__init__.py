"""Phase-4 seams. Local no-op implementations now; real catalog later.

These exist so the pipeline shape matches the manual's Phase 5.0 (authorize →
materialize → stamp) without building a commerce layer that has no consumer yet.
"""
from __future__ import annotations

from .authorize import authorize
from .watermark import watermark

__all__ = ["authorize", "watermark"]
