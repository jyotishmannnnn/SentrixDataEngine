"""Sub-frame tactile bucketing — the manual's single most important fidelity rule.

LeRobot/RLDS/HDF5 are frame-indexed at the slow (anchor) rate. Decimating the
high-rate tactile to that rate throws away the premium signal. Instead, gather
the high-rate samples falling inside each anchor-frame interval into a fixed
``[R, *payload_shape]`` tensor.

We REUSE SentrixSync's approved index logic (``sentrixsync.sync.join``):
``compute_R`` and ``subframe_buckets`` produce the per-frame index sets and the
fixed-R repeat-pad / validity rule. We only *materialize* those indices into
values — exactly the step SentrixSync defers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubframeTensor:
    """Materialized per-frame high-rate burst."""
    R: int
    anchor_fps: float
    tensor: np.ndarray   # [n_frames, R, *payload_shape] float32 (repeat-padded)
    m_k: np.ndarray      # [n_frames] int   real sample count per frame (capped at R)
    valid: np.ndarray    # [n_frames] bool  False = empty frame (gap, not padded)

    @property
    def n_frames(self) -> int:
        return int(self.tensor.shape[0])


def materialize_subframe(anchor_times_us: np.ndarray, highrate_times_us: np.ndarray,
                         highrate_values: np.ndarray, grid_rate_hz: float,
                         anchor_fps: float) -> SubframeTensor:
    """Bucket `highrate_values` ([N, *shape]) into anchor-frame intervals.

    `highrate_times_us` and `highrate_values` must be index-aligned and sorted by
    time. Returns a tensor of shape [n_frames, R, *shape]; empty frames are zero
    and flagged invalid (never repeat-padded from a neighbour).
    """
    from sentrixsync.sync.join import compute_R, subframe_buckets

    R = compute_R(grid_rate_hz, anchor_fps)
    buckets = subframe_buckets(np.asarray(anchor_times_us, dtype=np.int64),
                               np.asarray(highrate_times_us, dtype=np.int64), R)
    shape = highrate_values.shape[1:]
    n_frames = buckets.index.shape[0]
    tensor = np.zeros((n_frames, R, *shape), dtype=np.float32)
    for k in range(n_frames):
        if not buckets.valid[k]:
            continue
        tensor[k] = highrate_values[buckets.index[k]]
    return SubframeTensor(R=R, anchor_fps=float(anchor_fps), tensor=tensor,
                          m_k=np.asarray(buckets.m_k, dtype=int),
                          valid=np.asarray(buckets.valid, dtype=bool))
