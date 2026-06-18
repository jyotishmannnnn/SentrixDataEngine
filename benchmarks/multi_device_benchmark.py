"""Multi-device synchronization + materialization benchmark.

Validation artifact only — builds NO new infrastructure. It exercises the existing
SentrixSync + SentrixDataEngine stack under realistic multi-device conditions:

  Device A : tactile reference, 1000 Hz, identity clock (the anchor)
  Device B : tactile, 1000 Hz, +offset +drift
  Device C : IMU-like accel, 200 Hz, -offset -drift (different rate)
  Device D : tactile, 500 Hz, +offset +drift, partial observability + dropout
             (D NEVER co-observes A directly -> must reconcile transitively)

Shared events are injected across overlapping device subsets. The script:
  1. injects known clock corruption (ForwardClock) and recovers it via synchronize()
  2. reports expected vs recovered clock params + residual/error
  3. analyses confidence behaviour (three components + decay away from events)
  4. materializes sub-frame tactile buckets and verifies activation
  5. runs the full SentrixDataEngine export end to end

Run:  python benchmarks/multi_device_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sentrixsync.clock.forward import ForwardClock, enforce_monotonic_int_us
from sentrixsync.core import (
    ClockDescriptor, DeviceDescriptor, DeviceRegistration, DeviceRole,
    EvidenceTier, Kernel, Origin, Session, SessionMetadata, StreamDescriptor,
)
from sentrixsync.core.events import SyncEvent
from sentrixsync.sync.engine import synchronize
from sentrixsync.sync.join import compute_R, subframe_buckets

from sentrixdataengine import MaterializationRequest, Pipeline
from sentrixdataengine.materialize.subframe import materialize_subframe

T_REF_US = 2_000_000         # 2 s reference window — long enough to observe skew
GRID_RATE_HZ = 1000.0
TOLERANCE_US = 6000          # covers the 200 Hz device's 5 ms spacing on a 1 kHz grid
JITTER_US = 25.0             # detector timing noise on event observations
DECAY_TAU_US = 1_000_000.0   # clock-confidence decay constant (>> event spacing)
SEED = 7

# device_id -> (rate_hz, true ForwardClock, stream_id, kind, payload_kind, shape)
DEVICES = {
    "A": (1000.0, ForwardClock.from_offset_skew(0.0, 0.0),
          "tactile_field", "tactile_field", "bmm350_cluster_uT", (21, 3)),
    "B": (1000.0, ForwardClock.from_offset_skew(5000.0, 40.0),
          "tactile_field", "tactile_field", "bmm350_cluster_uT", (21, 3)),
    "C": (200.0, ForwardClock.from_offset_skew(-12000.0, -25.0),
          "dynamics", "imu_dynamics", "lis2dtw12_accel_g", (3, 3)),
    "D": (500.0, ForwardClock.from_offset_skew(8000.0, 60.0),
          "tactile_field", "tactile_field", "bmm350_cluster_uT", (21, 3)),
}

# Shared events across overlapping subsets, spread over the full window so skew is
# observable. A and D NEVER co-observe an event -> D must reconcile transitively.
_SUBSET_CYCLE = [
    ["A", "B", "C"], ["A", "B"], ["A", "C"], ["B", "C", "D"],
    ["C", "D"], ["B", "D"], ["A", "B", "C"], ["B", "C", "D"],
]
_N_EVENTS = 24
EVENTS = [
    (int(t), _SUBSET_CYCLE[i % len(_SUBSET_CYCLE)])
    for i, t in enumerate(np.linspace(50_000, T_REF_US - 50_000, _N_EVENTS))
]


def _local_grid(rate_hz: float, clock: ForwardClock) -> np.ndarray:
    """Device-local sample timestamps covering the reference window [0, T_REF]."""
    step = 1e6 / rate_hz
    lo = float(clock.local_from_ref(0.0))
    hi = float(clock.local_from_ref(T_REF_US))
    n = int((hi - lo) // step) + 1
    return enforce_monotonic_int_us(lo + np.arange(n) * step)


def _write_payload(path: Path, n: int, payload_kind: str, rng) -> None:
    cols: dict[str, pa.Array] = {"t_master_us": pa.array(np.arange(n, dtype=np.int64))}
    if payload_kind == "bmm350_cluster_uT":
        for i in range(21):
            for ax in ("bx", "by", "bz"):
                cols[f"tactile.b{i:02d}.{ax}_uT"] = pa.array(
                    rng.normal(0, 1, n).astype(np.float32))
    elif payload_kind == "lis2dtw12_accel_g":
        for finger in ("thumb", "index", "middle"):
            for ax in ("ax", "ay", "az"):
                cols[f"dyn.{finger}.{ax}_g"] = pa.array(rng.normal(0, 1, n).astype(np.float32))
    pq.write_table(pa.table(cols), path, compression="zstd")


def _descriptor(dev: str) -> DeviceDescriptor:
    rate, _clk, stream_id, kind, payload_kind, shape = DEVICES[dev]
    return DeviceDescriptor(
        device_id=dev, modality="tactile" if "tactile" in kind else "imu",
        producer="sentrixsim", is_synthetic=True, reference_candidate=(dev == "A"),
        clock=ClockDescriptor(clock_id=f"{dev}_hub", resolution_us=1,
                              nominal_epoch="session_start"),
        evidence_tiers=[EvidenceTier.SHARED_EVENT, EvidenceTier.WALL_CLOCK],
        streams=[StreamDescriptor(
            stream_id=stream_id, device_id=dev, kind=kind, kernel=Kernel.CONTINUOUS,
            payload_kind=payload_kind, units="uT" if "tactile" in kind else "g",
            nominal_rate_hz=rate, payload_shape=list(shape), subframe_capable=True)])


def build_benchmark(out_root: Path) -> dict:
    rng = np.random.default_rng(SEED)
    out_root = Path(out_root)
    payload_dir = out_root / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    descriptors, stream_ts, stream_refs, ground_truth, expected_counts = {}, {}, {}, {}, {}
    local_grids: dict[str, np.ndarray] = {}

    for dev, (rate, clk, stream_id, _kind, payload_kind, shape) in DEVICES.items():
        descriptors[dev] = _descriptor(dev)
        grid = _local_grid(rate, clk)
        ground_truth[dev] = clk
        # Device D: drop ~5% of samples to create real dropout + coverage gaps.
        if dev == "D":
            keep = rng.random(grid.size) >= 0.05
            kept = grid[keep]
        else:
            kept = grid
        local_grids[dev] = kept
        stream_ts[(dev, stream_id)] = kept
        expected_counts[(dev, stream_id)] = int(grid.size)   # full grid expected -> dropout shows
        p = payload_dir / f"{dev}.parquet"
        _write_payload(p, kept.size, payload_kind, rng)
        stream_refs[dev] = {stream_id: str(p)}

    # Shared events: each observer records local time = local_from_ref(t_ref) + jitter.
    events: list[SyncEvent] = []
    for i, (t_ref, subset) in enumerate(EVENTS):
        obs = {}
        for dev in subset:
            clk = DEVICES[dev][1]
            t_local = float(clk.local_from_ref(t_ref)) + rng.normal(0.0, JITTER_US)
            obs[dev] = int(round(t_local))
        events.append(SyncEvent(event_id=f"ev{i}", tier=EvidenceTier.SHARED_EVENT,
                                observations=obs, detector="synthetic", kind="impulse"))

    sync_result = synchronize(
        reference_device_id="A", descriptors=descriptors,
        stream_timestamps=stream_ts, sync_events=events,
        grid_rate_hz=GRID_RATE_HZ, rejection_tolerance_us=TOLERANCE_US,
        ground_truth=ground_truth, expected_counts=expected_counts,
        robust_estimation=False, confidence_decay_tau_us=DECAY_TAU_US, min_events=2)

    session = Session(
        metadata=SessionMetadata(session_id="BENCH_MULTI_0001", origin=Origin.SYNTHETIC,
                                 producers=["sentrixsim"], grid_rate_hz=GRID_RATE_HZ,
                                 rejection_tolerance_us=TOLERANCE_US),
        devices=[DeviceRegistration(
            device_id=dev, role=DeviceRole.REFERENCE if dev == "A" else DeviceRole.FOLLOWER,
            descriptor=descriptors[dev], stream_refs=stream_refs[dev])
            for dev in DEVICES])

    return {"sync_result": sync_result, "session": session, "events": events,
            "ground_truth": ground_truth, "local_grids": local_grids,
            "descriptors": descriptors, "out_root": out_root}


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def clock_recovery_report(sync_result, ground_truth) -> dict:
    rt = sync_result.metrics.get("roundtrip_accuracy") or {}
    rows = {}
    for dev, model in sync_result.clock_models.items():
        gt = ground_truth[dev]
        rows[dev] = {
            "true_alpha": gt.alpha, "recovered_alpha": round(model.alpha, 9),
            "true_beta_us": gt.beta_us, "recovered_beta_us": round(model.beta_us, 3),
            "clock_confidence": (round(model.clock_confidence, 4)
                                 if model.clock_confidence is not None else None),
            "fit_residual_us": (round(model.fit_residual_us, 3)
                                if model.fit_residual_us is not None else None),
            "alpha_err": round(rt.get(dev, {}).get("alpha_err", 0.0), 9),
            "beta_err_us": round(rt.get(dev, {}).get("beta_err_us", 0.0), 3),
            "alignment_rmse_us": round(rt.get(dev, {}).get("alignment_rmse_us", 0.0), 3),
        }
    return rows


def graph_report(sync_result) -> dict:
    m = sync_result.metrics
    return {"reachable": m["reachable"], "unreachable": m["unreachable"],
            "hops": m["hops"], "n_edges": m["n_edges"]}


def confidence_report(sync_result) -> dict:
    out = {}
    for key, comp in sync_result.confidence.items():
        out[key] = {
            "source_mean": round(float(comp.source.mean()), 4),
            "clock_min": round(float(comp.clock.min()), 4),
            "clock_max": round(float(comp.clock.max()), 4),
            "clock_mean": round(float(comp.clock.mean()), 4),
            "interp_mean": round(float(comp.interpolation.mean()), 4),
            "scalar_mean": round(float(comp.derived_scalar().mean()), 4),
        }
    return out


def coverage_report(sync_result) -> dict:
    return {"coverage": {k: round(v, 4) for k, v in sync_result.metrics["coverage"].items()},
            "coverage_min": round(sync_result.metrics["coverage_min"], 4),
            "dropout": {k: round(v, 4) for k, v in sync_result.metrics["dropout"].items()},
            "dropout_max": round(sync_result.metrics["dropout_max"], 4),
            "sync_resid_us": round(sync_result.metrics["sync_resid_us"], 3),
            "sync_gate_verdict": sync_result.validation_report.gate_verdict.value}


def subframe_report(sync_result) -> dict:
    """Materialize sub-frame tactile buckets for device A (1000 Hz) anchored at
    200 Hz (device C's rate) and verify activation + materialization."""
    timeline = sync_result.timeline
    grid_us = np.asarray(timeline.grid_us, dtype=np.int64)
    align_A = timeline.per_stream["A::tactile_field"]
    n_grid = grid_us.size
    # synthetic high-rate values on the grid (shape [n_grid, 21, 3])
    values = np.tile(np.arange(n_grid, dtype=np.float32).reshape(n_grid, 1, 1), (1, 21, 3))
    anchor_fps = 200.0
    anchor_times = grid_us[::5]                       # 200 Hz frame edges from the 1 kHz grid
    sub = materialize_subframe(anchor_times, grid_us, values, GRID_RATE_HZ, anchor_fps)
    # cross-check the index logic against SentrixSync's approved bucketer
    R_expected = compute_R(GRID_RATE_HZ, anchor_fps)
    buckets = subframe_buckets(anchor_times, grid_us, R_expected)
    return {
        "anchor_fps": anchor_fps, "R_expected": R_expected, "R_materialized": sub.R,
        "tensor_shape": list(sub.tensor.shape), "n_frames": sub.n_frames,
        "all_frames_valid": bool(sub.valid.all()),
        "m_k_unique": sorted(set(int(x) for x in sub.m_k)),
        "matches_sync_indices": bool(np.array_equal(sub.m_k, buckets.m_k)),
        "first_frame_first_sample": float(sub.tensor[0, 0, 0, 0]),
        "first_frame_last_sample": float(sub.tensor[0, R_expected - 1, 0, 0]),
    }


def export_report(state) -> dict:
    result = Pipeline().run(MaterializationRequest(
        sync_result=state["sync_result"], session=state["session"],
        out_root=state["out_root"] / "gold",
        formats=("parquet", "lerobot", "hdf5", "mcap")))
    return {"qa_verdict": result.qa.gate_verdict, "qa_detail": result.qa.detail,
            "formats": [e.format for e in result.exports],
            "silver_streams": sorted(result.canonical.streams.keys()),
            "n_grid": result.canonical.n_grid,
            "content_hash": result.content_hash[:16],
            "base": str(result.layout.base),
            "session_export_records": len(state["session"].exports)}


def run() -> dict:
    import tempfile
    state = build_benchmark(Path(tempfile.mkdtemp(prefix="sentrix_bench_")))
    sr = state["sync_result"]
    return {
        "clock_recovery": clock_recovery_report(sr, state["ground_truth"]),
        "graph": graph_report(sr),
        "coverage": coverage_report(sr),
        "confidence": confidence_report(sr),
        "subframe": subframe_report(sr),
        "export": export_report(state),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
