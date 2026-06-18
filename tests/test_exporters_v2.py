from __future__ import annotations

import json

import numpy as np

from sentrixdataengine import SCHEMA_VERSION
from sentrixdataengine.contracts import DatasetSpec
from sentrixdataengine.exporters import get_exporter, registered_exporters
from sentrixdataengine.materialize.projector import project
from sentrixdataengine.resolve import default_registry


def _canonical(sync_result, descriptor, sim_parquet):
    sources = {"glove_L::tactile_field":
               "parquet://" + str(sim_parquet).replace("\\", "/")}
    return project(sync_result, {"glove_L": descriptor}, default_registry(), sources,
                   session_id="S1", schema_version=SCHEMA_VERSION)


def _spec(table):
    return DatasetSpec(dataset_id="ds1", version="0.2.0", session_id="S1",
                       reference_clock_id=table.reference_clock_id,
                       grid_rate_hz=table.grid_rate_hz, schema_version=SCHEMA_VERSION,
                       profile="default", engine_version="0.2.0")


def test_registry_has_v2_formats():
    assert {"lerobot", "mcap", "parquet", "hdf5", "rlds"}.issubset(set(registered_exporters()))


def test_hdf5_export(sync_result, descriptor, sim_parquet, tmp_path):
    import h5py
    table = _canonical(sync_result, descriptor, sim_parquet)
    res = get_exporter("hdf5").export(table, _spec(table), tmp_path / "h5")
    assert res.files[0].exists()
    with h5py.File(res.files[0], "r") as f:
        obs = f["data"]["demo_0"]["obs"]
        assert obs["tactile_field"].shape == (table.n_grid, 63)
        assert f["data"]["demo_0"].attrs["num_samples"] == table.n_grid


def test_rlds_export_step_fields(sync_result, descriptor, sim_parquet, tmp_path):
    import pyarrow.parquet as pq
    table = _canonical(sync_result, descriptor, sim_parquet)
    res = get_exporter("rlds").export(table, _spec(table), tmp_path / "rlds")
    steps = pq.read_table(tmp_path / "rlds" / "steps.parquet")
    names = set(steps.column_names)
    assert {"reward", "discount", "is_first", "is_last", "is_terminal"}.issubset(names)
    is_first = np.asarray(steps.column("is_first").to_numpy())
    is_last = np.asarray(steps.column("is_last").to_numpy())
    assert is_first[0] and is_last[-1]
    meta = json.loads((tmp_path / "rlds" / "episode_metadata.json").read_text())
    assert meta["num_steps"] == table.n_grid


def test_lerobot_v2_vs_v3_layout(sync_result, descriptor, sim_parquet, tmp_path):
    table = _canonical(sync_result, descriptor, sim_parquet)
    spec = _spec(table)
    v3 = get_exporter("lerobot").export(table, spec, tmp_path / "v3", {"layout": "v3"})
    v2 = get_exporter("lerobot").export(table, spec, tmp_path / "v2", {"layout": "v2"})
    assert (tmp_path / "v3" / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "v2" / "data" / "chunk-000" / "episode_000000.parquet").exists()
    info_v2 = json.loads((tmp_path / "v2" / "meta" / "info.json").read_text())
    assert info_v2["layout"] == "v2"
    assert "lerobot_validation" in info_v2
