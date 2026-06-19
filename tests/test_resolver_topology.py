"""Migration Phase 3: the parquet resolver is topology-driven (sensor_id-keyed),
reading per-modality sensor order from the parquet KV metadata. Legacy Layout-B
files still resolve via fallback."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sentrixdataengine.resolve import default_registry

N = 8
BMM = ["bmm_thumb_0", "bmm_thumb_1", "bmm_index_0"]   # arbitrary count/order
LIS = ["lis_thumb", "lis_index"]


def _write_sensor_id_parquet(path, with_meta=True):
    cols = {"t_master_us": pa.array(np.arange(N, dtype=np.int64) * 625)}
    # distinguishable constants: sensor k, axis a -> value k*10 + a
    for k, sid in enumerate(BMM):
        for a, ax in enumerate(("bx", "by", "bz")):
            cols[f"mag.{sid}.{ax}_uT"] = pa.array(np.full(N, k * 10 + a, np.float32))
    for k, sid in enumerate(LIS):
        for a, ax in enumerate(("ax", "ay", "az")):
            cols[f"dyn.{sid}.{ax}_g"] = pa.array(np.full(N, 100 + k * 10 + a, np.float32))
        cols[f"dyn.{sid}.temp_c"] = pa.array(np.full(N, 25.0 + k, np.float32))
    table = pa.table(cols)
    if with_meta:
        meta = {b"sentrixsim_meta": json.dumps(
            {"bmm_sensor_ids": BMM, "lis_sensor_ids": LIS}).encode()}
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, path, compression="zstd")


def _uri(p):
    return "parquet://" + str(p).replace("\\", "/")


def test_resolve_sensor_id_tactile(tmp_path):
    p = tmp_path / "ep_sid.parquet"
    _write_sensor_id_parquet(p)
    arr = default_registry().resolve_stream(_uri(p), "bmm350_cluster_uT", (len(BMM), 3))
    assert arr.shape == (N, 3, 3)
    # order follows meta bmm_sensor_ids; values reconstruct k*10 + axis
    for k in range(len(BMM)):
        assert list(arr[0, k]) == [k * 10, k * 10 + 1, k * 10 + 2]


def test_resolve_sensor_id_accel_and_temp(tmp_path):
    p = tmp_path / "ep_sid.parquet"
    _write_sensor_id_parquet(p)
    reg = default_registry()
    acc = reg.resolve_stream(_uri(p), "lis2dtw12_accel_g", (len(LIS), 3))
    assert acc.shape == (N, 2, 3)
    assert list(acc[0, 1]) == [110, 111, 112]
    temp = reg.resolve_stream(_uri(p), "lis2dtw12_temp_degC", None)
    assert temp.shape == (N, len(LIS))
    assert list(temp[0]) == [25.0, 26.0]


def test_arbitrary_count_no_code_change(tmp_path):
    """The resolver adapts to whatever sensor count the file declares."""
    p = tmp_path / "ep_sid.parquet"
    _write_sensor_id_parquet(p)
    arr = default_registry().resolve_stream(_uri(p), "bmm350_cluster_uT", (3, 3))
    assert arr.shape[1] == len(BMM) == 3


def test_legacy_layout_b_fallback(tmp_path):
    """A file with no sensor-id metadata (pre-Phase-1) still resolves via the
    legacy tactile.bNN scheme."""
    cols = {"t_master_us": pa.array(np.arange(N, dtype=np.int64) * 625)}
    rng = np.random.default_rng(0)
    for i in range(21):
        for ax in ("bx", "by", "bz"):
            cols[f"tactile.b{i:02d}.{ax}_uT"] = pa.array(rng.normal(0, 1, N).astype(np.float32))
    p = tmp_path / "ep_legacy.parquet"
    pq.write_table(pa.table(cols), p, compression="zstd")
    arr = default_registry().resolve_stream(_uri(p), "bmm350_cluster_uT", (21, 3))
    assert arr.shape == (N, 21, 3)
