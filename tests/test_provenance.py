from __future__ import annotations

import json

from sentrixdataengine.package.provenance import merkle_root, stamp_provenance


def test_merkle_root_deterministic():
    h = ["aa" * 32, "bb" * 32, "cc" * 32]
    assert merkle_root(h) == merkle_root(h)
    assert merkle_root(h) != merkle_root(list(reversed(h)))


def test_stamp_writes_sidecar(tmp_path):
    f1 = tmp_path / "a.bin"; f1.write_bytes(b"hello")
    f2 = tmp_path / "b.bin"; f2.write_bytes(b"world")
    side = tmp_path / "prov.json"
    res = stamp_provenance([f1, f2], side, dataset_id="ds1", version="0.1.0",
                           session_id="s1", schema_version="1.0",
                           source_episode_hashes=["deadbeef"])
    assert side.exists()
    data = json.loads(side.read_text())
    assert data["merkle_root"] == res.merkle_root
    assert set(data["file_hashes"]) == {str(f1), str(f2)}
    assert data["algorithm"] in ("ed25519", "hmac-sha256")
