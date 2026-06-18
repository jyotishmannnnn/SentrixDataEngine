"""Apply SentrixSync's as-of join indices to resolved payloads → canonical table.

The timeline gives, per stream, a StreamAlignment of integer indices
(`source_index`, `next_index`, `weight`, `valid`). We never re-derive the join;
we apply it:
  * HOLD       → value at source_index (zero-order hold)
  * CONTINUOUS → linear blend of source_index and next_index by weight
Gaps (`valid == False`) are filled with NaN and flagged — never fabricated.
"""
from __future__ import annotations

import numpy as np

from ..contracts import PayloadResolver
from .canonical import CanonicalStream, CanonicalTable
from .confidence import fold


def _apply_alignment(payload: np.ndarray, alignment) -> np.ndarray:
    """payload: [N, *shape] → out: [n_grid, *shape] float32, NaN at gaps."""
    n_grid = alignment.valid.shape[0]
    shape = payload.shape[1:]
    out = np.full((n_grid, *shape), np.nan, dtype=np.float32)

    src = alignment.source_index
    valid = alignment.valid
    kernel = getattr(alignment.kernel, "value", alignment.kernel)

    if kernel == "hold":
        ok = valid & (src >= 0)
        out[ok] = payload[src[ok]]
        return out

    # continuous: blend source and next where available
    nxt = alignment.next_index
    w = alignment.weight.astype(np.float32)
    has_src = valid & (src >= 0)
    has_both = has_src & (nxt >= 0)
    only_src = has_src & (nxt < 0)

    # broadcast weight over payload dims
    wb = w.reshape((n_grid,) + (1,) * len(shape))
    if np.any(has_both):
        a = payload[src[has_both]]
        b = payload[nxt[has_both]]
        out[has_both] = (a * (1.0 - wb[has_both]) + b * wb[has_both]).astype(np.float32)
    if np.any(only_src):
        out[only_src] = payload[src[only_src]]
    return out


def project(sync_result, descriptors: dict, registry, payload_sources: dict[str, str],
            *, session_id: str, schema_version: str) -> CanonicalTable:
    """Build the canonical table from a SyncResult.

    `descriptors`: device_id -> sentrixsync DeviceDescriptor (for payload_kind/shape).
    `payload_sources`: stream key ("device::stream" or bare "stream") -> base URI.
    `registry`: a ResolverRegistry.
    """
    timeline = sync_result.timeline
    grid_us = np.asarray(timeline.grid_us, dtype=np.int64)
    n_grid = grid_us.shape[0]
    frame_index = np.arange(n_grid, dtype=np.int64)

    streams: dict[str, CanonicalStream] = {}
    for key, alignment in timeline.per_stream.items():
        device_id, stream_id = key.split("::", 1)
        sd = _stream_descriptor(descriptors, device_id, stream_id)
        base_uri = payload_sources.get(key) or payload_sources.get(stream_id)
        if base_uri is None:
            raise KeyError(
                f"no payload source for stream {key!r}; provide it via "
                "MaterializationRequest.payload_sources or Session stream_refs")
        shape = tuple(sd.payload_shape) if sd.payload_shape else ()
        payload = registry.resolve_stream(base_uri, sd.payload_kind, shape or None)
        values = _apply_alignment(payload, alignment)

        conf = fold(sync_result.confidence.get(key), n_grid)
        streams[key] = CanonicalStream(
            key=key, device_id=device_id, stream_id=stream_id,
            payload_kind=sd.payload_kind, units=sd.units,
            kernel=getattr(alignment.kernel, "value", str(alignment.kernel)),
            shape=shape, values=values,
            valid=np.asarray(alignment.valid, dtype=bool),
            confidence=conf["scalar"], conf_source=conf["source"],
            conf_clock=conf["clock"], conf_interp=conf["interp"])

    return CanonicalTable(
        grid_us=grid_us, frame_index=frame_index, streams=streams,
        reference_clock_id=sync_result.reference_clock_id,
        grid_rate_hz=float(getattr(timeline, "grid_rate_hz", 0.0) or _infer_rate(grid_us)),
        session_id=session_id, schema_version=schema_version)


def _infer_rate(grid_us: np.ndarray) -> float:
    if grid_us.size < 2:
        return 0.0
    dt = float(np.median(np.diff(grid_us)))
    return 1e6 / dt if dt > 0 else 0.0


def _stream_descriptor(descriptors: dict, device_id: str, stream_id: str):
    desc = descriptors.get(device_id)
    if desc is None:
        raise KeyError(f"no descriptor for device {device_id!r}")
    for s in desc.streams:
        if s.stream_id == stream_id:
            return s
    raise KeyError(f"device {device_id!r} has no stream {stream_id!r}")
