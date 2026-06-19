"""Gold exporters — pure projections of the canonical (Silver) table.

Each format reads the one canonical representation; none reads the timeline or
resolves payloads directly. Register new formats with ``@register_exporter``.
"""
from __future__ import annotations

from .base import Exporter, get_exporter, register_exporter, registered_exporters

# Import side-effect: populate the registry.
from . import derived, hdf5, lerobot, mcap, parquet, rlds  # noqa: E402,F401

__all__ = ["Exporter", "get_exporter", "register_exporter", "registered_exporters"]
