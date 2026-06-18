from __future__ import annotations

from sentrixdataengine import MaterializationRequest, Pipeline
from sentrixdataengine.inspect import (
    diff_datasets,
    summarize_canonical,
    summarize_dataset,
)


def _run(session, sync_result, out):
    return Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=out, formats=("parquet",)))


def test_summarize_dataset(session, sync_result, tmp_path):
    r = _run(session, sync_result, tmp_path / "g")
    summ = summarize_dataset(r.layout.base)
    assert summ["session_id"] == "01J9SYNTH0001"
    assert summ["n_grid"] == r.canonical.n_grid
    assert "tactile_field" in summ["streams"]
    assert summ["streams"]["tactile_field"]["coverage"] == 1.0
    assert summ["qa_verdict"] == r.qa.gate_verdict
    assert summ["content_hash"] == r.content_hash


def test_summarize_canonical(session, sync_result, tmp_path):
    r = _run(session, sync_result, tmp_path / "g")
    summ = summarize_canonical(r.canonical)
    assert summ["streams"]["tactile_field"]["shape"] == [21, 3]


def test_diff_identical(session, sync_result, tmp_path):
    r1 = _run(session, sync_result, tmp_path / "a")
    session.exports.clear()
    r2 = _run(session, sync_result, tmp_path / "b")
    d = diff_datasets(r1.layout.base, r2.layout.base)
    assert d["content_hash_identical"] is True
    assert d["n_grid_delta"] == 0
    assert d["streams"]["tactile_field"]["status"] == "same"
