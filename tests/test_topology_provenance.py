"""The topology descriptor (version + hash) carried by SentrixSync flows into
DataEngine's Silver KV metadata, manifest, provenance sidecar, and datacard —
closing the loop dataset -> descriptor_hash -> exact hardware revision."""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from sentrixdataengine import MaterializationRequest, Pipeline

REF = "Mark2_v1"
HASH = "sha256:6a67490b1bbb8cb1992500920f1fc313135e6a889b717cf38c6c59c19419cde4"


def test_topology_packaged_everywhere(session, descriptor, sync_result, tmp_path):
    # producer-set opaque provenance (same object the session fixture registered)
    descriptor.topology_ref = REF
    descriptor.topology_hash = HASH

    result = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=tmp_path / "gold",
        formats=("parquet",)))
    base = result.layout.base

    # manifest
    man = json.loads(result.layout.manifest_path.read_text())
    assert man["topology"] == [{"device_id": "glove_L",
                                "topology_ref": REF, "topology_hash": HASH}]

    # provenance sidecar
    prov = json.loads(result.layout.provenance_path.read_text())
    assert prov["topology"][0]["topology_hash"] == HASH

    # datacard
    card = result.layout.datacard_path.read_text()
    assert "## Topology" in card and REF in card and HASH in card

    # Silver KV metadata
    meta = pq.read_schema(base / "silver" / "aligned" / "part-000.parquet").metadata
    topo = json.loads(meta[b"topology"].decode())
    assert topo[0]["topology_ref"] == REF


def test_topology_absent_is_empty(session, sync_result, tmp_path):
    """No topology declared -> empty lists, no crash, no spurious datacard section."""
    result = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=tmp_path / "gold",
        formats=("parquet",)))
    man = json.loads(result.layout.manifest_path.read_text())
    assert man["topology"] == []
    assert "## Topology" not in result.layout.datacard_path.read_text()
