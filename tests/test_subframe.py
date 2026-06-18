from __future__ import annotations

import numpy as np

from sentrixdataengine.materialize.subframe import materialize_subframe


def test_subframe_buckets_highrate_into_frames():
    # 1600 Hz high-rate, anchor at 400 Hz -> R = 4 samples per frame
    grid_rate = 1600.0
    anchor_fps = 400.0
    highrate_t = np.arange(16, dtype=np.int64) * 625        # 1600 Hz
    anchor_t = np.arange(5, dtype=np.int64) * 2500          # 400 Hz frame edges
    values = np.arange(16 * 6, dtype=np.float32).reshape(16, 2, 3)

    sub = materialize_subframe(anchor_t, highrate_t, values, grid_rate, anchor_fps)
    assert sub.R == 4
    # 5 anchor edges -> 4 frames
    assert sub.tensor.shape == (4, 4, 2, 3)
    assert sub.valid.all()
    assert (sub.m_k == 4).all()
    # first frame holds the first 4 high-rate samples
    assert np.array_equal(sub.tensor[0, 0], values[0])
    assert np.array_equal(sub.tensor[0, 3], values[3])


def test_subframe_empty_frame_flagged_not_padded():
    grid_rate = 1600.0
    anchor_fps = 800.0       # R = 2
    # high-rate samples only in the first interval; second interval empty
    highrate_t = np.array([0, 600], dtype=np.int64)
    anchor_t = np.array([0, 1250, 2500], dtype=np.int64)
    values = np.ones((2, 1), dtype=np.float32)
    sub = materialize_subframe(anchor_t, highrate_t, values, grid_rate, anchor_fps)
    assert sub.valid[0]
    assert not sub.valid[1]                 # gap, not repeat-padded
    assert np.all(sub.tensor[1] == 0.0)
