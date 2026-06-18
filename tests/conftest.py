"""Test fixtures: a real SentrixSim-layout Parquet + SentrixSync SyncResult."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentrixsync.core import (
    ClockDescriptor,
    DeviceDescriptor,
    DeviceRegistration,
    DeviceRole,
    EvidenceTier,
    Kernel,
    Origin,
    Session,
    SessionMetadata,
    StreamDescriptor,
)
from sentrixsync.sync.engine import synchronize

N = 16
STEP_US = 625          # 1600 Hz
N_CLUSTERS = 21


def _write_sim_parquet(path: Path) -> None:
    cols: dict[str, pa.Array] = {
        "t_master_us": pa.array(np.arange(N, dtype=np.int64) * STEP_US),
    }
    rng = np.random.default_rng(0)
    for i in range(N_CLUSTERS):
        for ax in ("bx", "by", "bz"):
            cols[f"tactile.b{i:02d}.{ax}_uT"] = pa.array(
                rng.normal(0, 1, N).astype(np.float32))
    pq.write_table(pa.table(cols), path, compression="zstd")


def _tactile_descriptor(device_id: str = "glove_L") -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id, modality="tactile", producer="sentrixsim",
        is_synthetic=True, reference_candidate=True,
        clock=ClockDescriptor(clock_id=f"{device_id}_hub", resolution_us=1,
                              nominal_epoch="session_start"),
        evidence_tiers=[EvidenceTier.SHARED_EVENT, EvidenceTier.WALL_CLOCK],
        streams=[StreamDescriptor(
            stream_id="tactile_field", device_id=device_id, kind="tactile_field",
            kernel=Kernel.CONTINUOUS, payload_kind="bmm350_cluster_uT", units="uT",
            nominal_rate_hz=1600.0, payload_shape=[21, 3], subframe_capable=True)])


@pytest.fixture
def sim_parquet(tmp_path) -> Path:
    p = tmp_path / "episode.parquet"
    _write_sim_parquet(p)
    return p


@pytest.fixture
def descriptor() -> DeviceDescriptor:
    return _tactile_descriptor()


@pytest.fixture
def session(sim_parquet, descriptor) -> Session:
    return Session(
        metadata=SessionMetadata(session_id="01J9SYNTH0001", origin=Origin.SYNTHETIC,
                                 producers=["sentrixsim"], grid_rate_hz=1600,
                                 rejection_tolerance_us=1875),
        devices=[DeviceRegistration(
            device_id="glove_L", role=DeviceRole.REFERENCE, descriptor=descriptor,
            stream_refs={"tactile_field": str(sim_parquet)})])


@pytest.fixture
def sync_result(descriptor):
    ts = np.arange(N, dtype=np.int64) * STEP_US
    return synchronize(
        reference_device_id="glove_L", descriptors={"glove_L": descriptor},
        stream_timestamps={("glove_L", "tactile_field"): ts}, sync_events=[],
        grid_rate_hz=1600.0, rejection_tolerance_us=1875)
