from __future__ import annotations

import numpy as np

from sentrixdataengine.resolve import default_registry


def test_resolve_tactile_stream_shape(sim_parquet):
    reg = default_registry()
    uri = "parquet://" + str(sim_parquet).replace("\\", "/")
    arr = reg.resolve_stream(uri, "bmm350_cluster_uT", (21, 3))
    assert arr.shape == (16, 21, 3)
    assert arr.dtype == np.float32


def test_unknown_payload_kind_raises(sim_parquet):
    reg = default_registry()
    uri = "parquet://" + str(sim_parquet).replace("\\", "/")
    try:
        reg.resolve_stream(uri, "mystery_kind", (1,))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_strip_fragment_used_for_row_addressed_uri(sim_parquet):
    reg = default_registry()
    uri = "parquet://" + str(sim_parquet).replace("\\", "/") + "#stream=tactile_field&row=3"
    arr = reg.resolve_stream(uri, "bmm350_cluster_uT", (21, 3))
    assert arr.shape == (16, 21, 3)
