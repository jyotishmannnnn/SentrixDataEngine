"""Packaging — Gold layout, dataset manifest, provenance, data card, versioning."""
from __future__ import annotations

from .datacard import write_datacard
from .layout import GoldLayout
from .manifest import append_export_record, write_manifest
from .provenance import ProvenanceResult, stamp_provenance
from .versioning import content_hash, derive_dataset_id

__all__ = [
    "GoldLayout", "write_manifest", "append_export_record", "write_datacard",
    "ProvenanceResult", "stamp_provenance", "content_hash", "derive_dataset_id",
]
