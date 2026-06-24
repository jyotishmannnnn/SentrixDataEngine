"""DE-CLI-1 — `sentrixdataengine materialize` from a persisted SyncResult bundle."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sentrixsync import save_sync_result

from sentrixdataengine.cli import app

runner = CliRunner()


def test_materialize_from_persisted_bundle(session, sync_result, tmp_path):
    bundle = tmp_path / "bundle"
    save_sync_result(sync_result, bundle, session=session)   # SYNC-1 bundle on disk

    out = tmp_path / "gold"
    res = runner.invoke(app, ["materialize", "--bundle", str(bundle),
                              "--out", str(out), "--formats", "parquet,lerobot"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)

    assert data["verdict"] in ("CERTIFIED", "RELEASE", "NEEDS_REVIEW", "BLOCKED")
    assert data["content_hash"]
    assert Path(data["silver"]).exists()
    assert Path(data["manifest"]).exists()
    assert Path(data["provenance"]).exists()
    base = Path(data["dataset"])
    assert (base / "format=parquet" / "part-000.parquet").exists()
    assert (base / "format=lerobot" / "meta" / "info.json").exists()


def test_materialize_is_deterministic(session, sync_result, tmp_path):
    bundle = tmp_path / "bundle"
    save_sync_result(sync_result, bundle, session=session)
    h = []
    for name in ("a", "b"):
        session.exports.clear()  # avoid export-record carryover between runs
        res = runner.invoke(app, ["materialize", "--bundle", str(bundle),
                                  "--out", str(tmp_path / name), "--formats", "parquet"])
        assert res.exit_code == 0, res.output
        h.append(json.loads(res.output)["content_hash"])
    assert h[0] == h[1]


def test_materialize_requires_session(sync_result, tmp_path):
    bundle = tmp_path / "no_session"
    save_sync_result(sync_result, bundle)            # session omitted
    res = runner.invoke(app, ["materialize", "--bundle", str(bundle),
                              "--out", str(tmp_path / "o"), "--formats", "parquet"])
    assert res.exit_code == 2
    assert "session.json" in res.output


def test_materialize_missing_bundle(tmp_path):
    res = runner.invoke(app, ["materialize", "--bundle", str(tmp_path / "nope"),
                              "--out", str(tmp_path / "o")])
    assert res.exit_code == 2
