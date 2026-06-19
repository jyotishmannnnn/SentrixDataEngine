"""Derived-feature exporter: topology-dependent proxies (normal/shear/centroid)
materialized per cluster from raw B, using the descriptor's spatial layout.
Opt-in; never touches Silver; records formula+version+descriptor hash."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from sentrixdataengine import MaterializationRequest, Pipeline

REF = "Mark2_v1"
HASH = "sha256:6a67490b1bbb8cb1992500920f1fc313135e6a889b717cf38c6c59c19419cde4"


def _run_derived(session, descriptor, sync_result, out):
    descriptor.topology_ref = REF
    descriptor.topology_hash = HASH
    return Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=out, formats=("derived",)))


def test_derived_features_and_metadata(session, descriptor, sync_result, tmp_path):
    result = _run_derived(session, descriptor, sync_result, tmp_path / "gold")
    path = result.layout.base / "format=derived" / "part-000.parquet"
    assert path.exists()

    t = pq.read_table(path)
    cols = set(t.column_names)
    # topology-dependent features, per descriptor cluster (thumb..palm from Mark2_v1)
    for c in ("derived.thumb.normal_proxy", "derived.thumb.shear_x",
              "derived.thumb.shear_y", "derived.thumb.shear_mag",
              "derived.thumb.centroid_x_m", "derived.thumb.centroid_y_m",
              "derived.palm.normal_proxy"):
        assert c in cols
    assert t.num_rows == result.canonical.n_grid

    meta = t.schema.metadata
    assert meta[b"descriptor_version"] == REF.encode()
    assert meta[b"descriptor_hash"] == HASH.encode()
    assert meta[b"sentrixdataengine_derived_version"] == b"1.0"
    assert meta[b"baseline_method"] == b"per_sensor_median_over_valid"
    formulas = json.loads(meta[b"formulas"].decode())
    assert "normal_proxy" in formulas and "centroid_x_m/centroid_y_m" in formulas
    # cluster grouping came from the descriptor (thumb quad = 4 sensors)
    clusters = json.loads(meta[b"clusters"].decode())
    assert len(clusters["thumb"]) == 4
    assert clusters["thumb"][0] == "bmm_thumb_0"


def test_centroid_within_cluster_bounds(session, descriptor, sync_result, tmp_path):
    """Response-weighted centroid must lie inside the cluster's xy extent (it's a
    convex combination of member positions) — a real topology-dependence check."""
    result = _run_derived(session, descriptor, sync_result, tmp_path / "gold")
    t = pq.read_table(result.layout.base / "format=derived" / "part-000.parquet")
    from sentrix_contracts import bundled_descriptor_path, load_descriptor
    d = load_descriptor(bundled_descriptor_path(REF))
    thumb = [s for s in d.sensors.values() if s.cluster_id == "thumb" and s.modality == "magnetic"]
    xs = [s.position_m[0] for s in thumb]
    cx = np.asarray(t.column("derived.thumb.centroid_x_m").to_pylist(), float)
    cx = cx[~np.isnan(cx)]
    assert cx.size > 0
    assert cx.min() >= min(xs) - 1e-6 and cx.max() <= max(xs) + 1e-6


def test_derived_requires_topology(session, sync_result, tmp_path):
    """Without topology provenance the exporter refuses (it needs the descriptor)."""
    with pytest.raises(ValueError, match="topology"):
        Pipeline().run(MaterializationRequest(
            sync_result=sync_result, session=session, out_root=tmp_path / "g",
            formats=("derived",)))


def test_silver_has_no_derived_columns(session, descriptor, sync_result, tmp_path):
    result = _run_derived(session, descriptor, sync_result, tmp_path / "gold")
    silver = pq.read_table(result.layout.base / "silver" / "aligned" / "part-000.parquet")
    assert not any(c.startswith("derived.") for c in silver.column_names)
