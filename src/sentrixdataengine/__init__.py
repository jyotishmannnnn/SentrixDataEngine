"""SentrixDataEngine — materialize synchronized Sentrix sessions into datasets.

Consumes SentrixSync's ``SyncResult`` / ``Session`` (immutable inputs), resolves
the payload references SentrixSync recorded, materializes one canonical aligned
columnar representation (Silver), then projects it to ML-ready formats (Gold),
validates each, and packages it with manifests, provenance and a data card.

Repository boundary (non-negotiable):
  * does NOT synchronize (SentrixSync owns that),
  * does NOT simulate (SentrixSim owns that),
  * does NOT import SentrixSim — raw bytes are reached only via payload_ref URIs,
  * never mutates a SyncResult; the only write-back is appending an ExportRecord.
"""
from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"  # canonical (Silver) schema version — see docs/CANONICAL_SCHEMA.md

from .contracts import (  # noqa: E402
    DatasetSpec,
    ExportResult,
    MaterializationRequest,
    QAReport,
)
from .pipeline import Pipeline, PipelineResult  # noqa: E402

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "MaterializationRequest",
    "DatasetSpec",
    "ExportResult",
    "QAReport",
    "Pipeline",
    "PipelineResult",
]
