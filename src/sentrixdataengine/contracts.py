"""Frozen data contracts for SentrixDataEngine.

These are the engine's public shapes. They reference SentrixSync types by name
in annotations only (lazy import) so this module is importable without a heavy
dependency chain, and so the boundary stays one-directional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:  # avoid a hard import cycle; sentrixsync is a runtime dep
    from sentrixsync.sync.engine import SyncResult


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterializationRequest:
    """A single materialization job.

    `sync_result` is the in-memory result of ``sentrixsync.sync.engine.synchronize``.
    `session` is the (optional) SentrixSync Session manifest object; when given,
    its device `stream_refs` supply payload base URIs and the engine appends an
    ExportRecord back into it. `payload_sources` overrides/augments the per-stream
    payload base URIs (key = "device::stream" or bare "stream").
    """
    sync_result: "SyncResult"
    out_root: Path
    formats: tuple[str, ...] = ("parquet",)
    session: Any | None = None
    payload_sources: dict[str, str] = field(default_factory=dict)
    dataset_id: str | None = None
    version: str = "0.1.0"
    profile: str = "default"
    subframe_anchor_fps: float | None = None  # enable sub-frame tactile bucketing
    customer_id: str | None = None            # passed to authorize/watermark hooks
    format_options: dict[str, dict] = field(default_factory=dict)  # per-format export options


# --------------------------------------------------------------------------- #
# Resolver protocol
# --------------------------------------------------------------------------- #
class PayloadResolver(Protocol):
    def supports(self, scheme: str) -> bool: ...

    def resolve_stream(self, base_uri: str, payload_kind: str,
                       payload_shape: tuple[int, ...] | None) -> np.ndarray:
        """Return the full per-stream payload array, shape ``[N, *payload_shape]``."""
        ...


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    """Resolved identity + layout for a materialized dataset."""
    dataset_id: str
    version: str
    session_id: str
    reference_clock_id: str
    grid_rate_hz: float
    schema_version: str
    profile: str
    engine_version: str


@dataclass
class ExportResult:
    """One produced format artifact."""
    format: str
    out_dir: Path
    files: list[Path]
    frame_count: int
    sample_count: int


@dataclass
class QAReport:
    """Dataset-level QA outcome (mirrors manual Phase 7.3 release gate)."""
    gate_verdict: str                       # CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED
    integrity: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, str] = field(default_factory=dict)   # name -> pass|fail
    detail: str = ""

    def ok_to_publish(self) -> bool:
        return self.gate_verdict in ("RELEASE", "CERTIFIED")
