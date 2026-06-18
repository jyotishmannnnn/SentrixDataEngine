"""Multi-device synchronization + materialization validation.

Runs the deterministic (seeded) benchmark and asserts the architecture behaves
correctly under realistic multi-device conditions: graph reconciliation, clock
recovery, confidence propagation, gap preservation, sub-frame activation, and a
full end-to-end export through SentrixDataEngine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmarks/ is a sibling of tests/ — make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from multi_device_benchmark import run  # noqa: E402


@pytest.fixture(scope="module")
def report() -> dict:
    return run()


def test_graph_reconciliation_transitive(report):
    g = report["graph"]
    assert set(g["reachable"]) == {"A", "B", "C", "D"}
    assert g["unreachable"] == []
    # A is the anchor; B,C attach directly; D is reconciled transitively (never
    # co-observes A) -> exactly 2 hops.
    assert g["hops"] == {"A": 0, "B": 1, "C": 1, "D": 2}


def test_clock_recovery_within_tolerance(report):
    rows = report["clock_recovery"]
    assert rows["A"]["alpha_err"] == 0.0 and rows["A"]["beta_err_us"] == 0.0
    for dev in ("B", "C", "D"):
        assert rows[dev]["alpha_err"] < 1e-4, dev          # < 100 ppm
        assert rows[dev]["beta_err_us"] < 100.0, dev       # < 100 us
        assert rows[dev]["alignment_rmse_us"] < 50.0, dev
    # the 2-hop device accumulates the most error (compounding along the path)
    assert rows["D"]["alignment_rmse_us"] >= rows["B"]["alignment_rmse_us"]


def test_residual_small(report):
    assert report["coverage"]["sync_resid_us"] < 200.0


def test_confidence_behavior(report):
    conf = report["confidence"]
    # clock confidence decays away from sync events (not flat)
    assert conf["B::tactile_field"]["clock_min"] < conf["B::tactile_field"]["clock_max"]
    # device D dropped samples -> source confidence below 1
    assert conf["D::tactile_field"]["source_mean"] < 1.0
    # all components stay within [0,1]
    for c in conf.values():
        assert 0.0 <= c["clock_min"] <= c["clock_max"] <= 1.0
        assert 0.0 <= c["scalar_mean"] <= 1.0


def test_gaps_preserved_not_fabricated(report):
    # D's dropout leaves real gaps -> coverage strictly below 1 (never filled)
    assert report["coverage"]["coverage_min"] < 1.0
    assert report["coverage"]["dropout_max"] > 0.0


def test_subframe_activated_and_materialized(report):
    s = report["subframe"]
    assert s["R_expected"] == 5 and s["R_materialized"] == 5
    assert s["tensor_shape"] == [401, 5, 21, 3]
    assert s["matches_sync_indices"] is True
    assert s["all_frames_valid"] is True
    # repeat-pad / ordering sanity: first frame holds grid samples 0..4
    assert s["first_frame_first_sample"] == 0.0
    assert s["first_frame_last_sample"] == 4.0


def test_end_to_end_export(report):
    e = report["export"]
    assert set(e["silver_streams"]) == {
        "A::tactile_field", "B::tactile_field", "C::dynamics", "D::tactile_field"}
    assert set(e["formats"]) == {"parquet", "lerobot", "hdf5", "mcap"}
    assert e["session_export_records"] == 4
    assert e["qa_verdict"] in ("CERTIFIED", "RELEASE", "NEEDS_REVIEW", "BLOCKED")
    assert e["content_hash"]
