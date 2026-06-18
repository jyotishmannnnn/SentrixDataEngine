"""The canonical (Silver) representation — the single internal source of truth.

See docs/CANONICAL_SCHEMA.md. Held in memory as per-stream value arrays plus
validity + confidence; flattened to columns only when written to Parquet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CanonicalStream:
    """One synchronized stream resampled onto the reference grid."""
    key: str                       # "device::stream"
    device_id: str
    stream_id: str
    payload_kind: str
    units: str
    kernel: str                    # "continuous" | "hold"
    shape: tuple[int, ...]         # per-frame payload shape, e.g. (21, 3)
    values: np.ndarray             # [n_grid, *shape] float32, NaN at gaps
    valid: np.ndarray              # [n_grid] bool
    confidence: np.ndarray         # [n_grid] float, derived scalar (source*clock*interp)
    conf_source: np.ndarray        # [n_grid] float
    conf_clock: np.ndarray         # [n_grid] float
    conf_interp: np.ndarray        # [n_grid] float

    @property
    def n_grid(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        w = 1
        for d in self.shape:
            w *= d
        return int(w)

    def flat_values(self) -> np.ndarray:
        return self.values.reshape(self.n_grid, self.width)

    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


@dataclass
class CanonicalTable:
    """Reference grid + per-stream aligned payloads + carried metadata."""
    grid_us: np.ndarray                    # [n_grid] int64, reference time
    frame_index: np.ndarray                # [n_grid] int64
    streams: dict[str, CanonicalStream]    # key -> CanonicalStream
    reference_clock_id: str
    grid_rate_hz: float
    session_id: str
    schema_version: str
    extra: dict = field(default_factory=dict)   # passthrough labels, source hashes, ...

    @property
    def n_grid(self) -> int:
        return int(self.grid_us.shape[0])

    def feature_names(self) -> dict[str, str]:
        """Map each stream key -> a unique, export-safe feature name.

        Uses the bare ``stream_id`` when it is unique across the table (so
        single-device output is unchanged), and disambiguates with the device id
        (``<device>.<stream>``) only when two devices expose the same stream id.
        """
        from collections import Counter
        counts = Counter(s.stream_id for s in self.streams.values())
        return {key: (s.stream_id if counts[s.stream_id] == 1
                      else f"{s.device_id}.{s.stream_id}")
                for key, s in self.streams.items()}

    def coverage_min(self) -> float:
        if not self.streams:
            return 1.0
        return min(s.coverage() for s in self.streams.values())
