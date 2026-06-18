from __future__ import annotations

import json

from sentrixdataengine import MaterializationRequest, Pipeline


def test_end_to_end_materialize(session, sync_result, tmp_path):
    out = tmp_path / "gold"
    result = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=out,
        formats=("parquet", "lerobot")))

    # verdict (CERTIFIED requires cryptographic signature; otherwise BLOCKED)
    assert result.qa.gate_verdict in ("CERTIFIED", "RELEASE", "NEEDS_REVIEW", "BLOCKED")

    base = result.layout.base
    assert (base / "silver" / "aligned" / "part-000.parquet").exists()
    assert (base / "format=parquet" / "part-000.parquet").exists()
    assert (base / "format=lerobot" / "meta" / "info.json").exists()
    assert (base / "format=lerobot" / "data" / "chunk-000" / "file-000.parquet").exists()
    assert result.layout.manifest_path.exists()
    assert result.layout.provenance_path.exists()
    assert result.layout.datacard_path.exists()
    assert result.layout.qa_path.exists()

    # ExportRecord appended back into the Session (the only write-back)
    assert len(session.exports) == 2
    formats = {e.format for e in session.exports}
    assert formats == {"parquet", "lerobot"}

    # manifest links back to the source session, not duplicates it
    manifest = json.loads(result.layout.manifest_path.read_text())
    assert manifest["source_session_id"] == "01J9SYNTH0001"
    assert manifest["content_hash"]


def test_reproducible_content_hash(session, sync_result, tmp_path):
    r1 = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=tmp_path / "a",
        formats=("parquet",)))
    # rebuild session.exports list to avoid carryover
    session.exports.clear()
    r2 = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=tmp_path / "b",
        formats=("parquet",)))
    assert r1.content_hash == r2.content_hash


def test_lerobot_info_features(session, sync_result, tmp_path):
    result = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=tmp_path / "g",
        formats=("lerobot",)))
    info = json.loads((result.layout.base / "format=lerobot" / "meta" / "info.json").read_text())
    assert "observation.tactile_field" in info["features"]
    assert info["features"]["observation.tactile_field"]["shape"] == [21, 3]
    assert info["total_frames"] == result.canonical.n_grid
