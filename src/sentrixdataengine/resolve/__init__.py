"""Payload resolution — SentrixSync deferred item #1.

SentrixSync carries payloads by reference (``parquet://...#stream=..&row=..``)
and explicitly does not resolve them. This package turns those references back
into arrays, honoring the boundary: it reads the *artifacts SentrixSim wrote*,
never imports SentrixSim.
"""
from __future__ import annotations

from .mcap_resolver import McapPayloadResolver
from .parquet_resolver import ParquetPayloadResolver
from .resolver import ResolverRegistry, default_registry

__all__ = ["ParquetPayloadResolver", "McapPayloadResolver", "ResolverRegistry",
           "default_registry"]
