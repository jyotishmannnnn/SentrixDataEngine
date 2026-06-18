"""Silver materialization — SentrixSync deferred items #2, #3, #4.

Turns ``SyncResult.timeline`` (grid + per-stream join indices) plus resolved
payloads into ONE canonical aligned columnar table. Every Gold exporter is a
pure projection of this single representation.
"""
from __future__ import annotations

from .canonical import CanonicalStream, CanonicalTable
from .projector import project
from .silver_writer import write_silver
from .subframe import SubframeTensor, materialize_subframe

__all__ = [
    "CanonicalStream",
    "CanonicalTable",
    "project",
    "write_silver",
    "SubframeTensor",
    "materialize_subframe",
]
