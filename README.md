# SentrixDataEngine

**Dataset materialization, export, validation, and packaging for synchronized
Sentrix sessions.**

SentrixDataEngine is the offline, deterministic pipeline that turns a *synchronized
timeline* (produced by [SentrixSync](#architecture)) into versioned, validated,
ML-ready, provenance-stamped datasets on disk. It is the consumer that sits between
synchronization infrastructure and the formats robotics/ML teams actually train on
(LeRobot, RLDS, HDF5, MCAP, Parquet).

---

## Overview

The Sentrix data path has four repositories (plus the shared `sentrix_contracts`
layer) with strict, non-overlapping responsibilities:

| Repo | Role | Produces |
|---|---|---|
| **SentrixSim** | Forward simulator for the Mark 2 visuotactile glove | Raw per-device episode streams (Parquet/MCAP) |
| **SentrixCapture** | Real-hardware capture for the same glove | Raw per-device episode streams (same self-describing artifact as SentrixSim) |
| **SentrixSync** | Multi-device timeline synchronization | An in-memory `SyncResult` (reference grid + as-of join indices + confidence) and a `Session` manifest |
| **SentrixDataEngine** | **This repo** — dataset materialization → export → validation → packaging | Silver canonical table + Gold format exports + manifest + provenance |

SentrixDataEngine is, concretely, four layers:

- **Dataset materialization layer** — resolves the payload references SentrixSync
  recorded and applies its as-of join indices to build *one canonical aligned
  columnar table* (the "Silver" representation).
- **Export layer** — projects that single canonical table into target formats
  (LeRobot, RLDS, HDF5, MCAP, Parquet). Every format is a pure projection of the
  same source of truth.
- **Validation layer** — schema / timeline / metadata / confidence checks plus a
  dataset-level QA release gate (`CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED`).
- **Packaging layer** — Gold directory layout, dataset manifest, Merkle +
  Ed25519 provenance sidecar, data card, reproducible content hash, and an
  append-only `ExportRecord` write-back into the originating `Session`.
  Per-device topology provenance (`device_id` → `topology_ref` → `topology_hash`,
  read from each `DeviceDescriptor`) is carried into the Silver KV metadata, the
  manifest, the Merkle-signed provenance sidecar, and the data card — closing the
  lineage from a delivered dataset to the exact hardware revision it was produced on.

### What SentrixDataEngine is NOT

- **Not a simulator.** It never generates sensor data. SentrixSim does that.
- **Not a synchronization engine.** It never fits clocks, builds grids, or
  performs as-of joins. SentrixSync does that; this engine *applies* its results.
- **Not a perception system.** No SAM2 / FoundationPose / VLM / autolabeling.
- **Not a training framework.** It prepares datasets; it does not train models.
- **Not a catalog / storefront.** Entitlement and watermark are interface seams
  (`hooks/`) with local no-op implementations; commerce lives elsewhere.

---

## Architecture

```
┌──────────────────────┐  raw episodes        ┌──────────────┐
│ SentrixSim /         │ ───(parquet/mcap)───▶ │  SentrixSync │
│ SentrixCapture (HW)  │                       └──────┬───────┘
└──────────────────────┘     synchronize N devices    │
                                                       │  SyncResult (in-memory):
                                                       │    timeline.grid_us
                                                       │    per_stream StreamAlignment (join indices)
                                                       │    confidence (source/clock/interp)
                                                       │  + Session manifest (descriptors, stream_refs)
                                                       ▼
                                          ┌──────────────────────────────┐
                                          │       SentrixDataEngine       │
                                          │                               │
                                          │  resolve/    payload_ref → ndarray
                                          │  materialize/  apply join → CANONICAL (Silver)
                                          │  validate/   schema/timeline/conf + release gate
                                          │  exporters/  project → Gold formats
                                          │  package/    manifest + Merkle/Ed25519 + datacard
                                          └───────────────┬───────────────┘
                                                          │
                       ┌──────────────────────────────────┼───────────────────────┐
                       ▼                                   ▼                       ▼
              Silver: aligned/part-000.parquet     Gold: format=lerobot/    format=mcap/ rlds/ hdf5/ parquet/
              (one canonical source of truth)       manifest.json · provenance.sidecar.json · DATACARD.md · qa_report.json
```

**The boundary is the `SyncResult` / `Session` object.** SentrixDataEngine imports
SentrixSync types (read-only) and never imports the producers (SentrixSim or
SentrixCapture) — it reaches raw bytes only by resolving the opaque `payload_ref`
URIs SentrixSync already recorded (e.g.
`parquet://<abs>#stream=tactile_field&row=12`). The only write-back is appending an
`ExportRecord` to a `Session` (additive, contract-allowed).

---

## Features

| Capability | Module | Notes |
|---|---|---|
| **Canonical Silver materialization** | `materialize/{projector,canonical,silver_writer}.py` | One in-memory `CanonicalTable` + one Silver Parquet; every export projects from it |
| **Payload resolution** | `resolve/{parquet,mcap}_resolver.py` | `parquet://`, `file://`, `mcap://`; self-describing — reads each producer file's per-modality sensor order from KV metadata and builds sensor_id-keyed columns, so any sensor count resolves with no code change (legacy fixed-column layout still read via a fallback) |
| **Confidence propagation** | `materialize/confidence.py` | Carries SentrixSync's three components (source/clock/interp) + export scalar `source*clock*interp` |
| **Sub-frame tactile bucketing** | `materialize/subframe.py` | Fixed-`R` `[R,*shape]` burst per anchor frame; reuses `sentrixsync.sync.join.subframe_buckets` (the manual's premium-fidelity rule) |
| **Dataset validation** | `validate/{schema,timeline,metadata,confidence}_check.py` | Shapes, monotonic grid, bounded step, no fabricated gaps, confidence ∈ [0,1] |
| **QA release gate** | `validate/release_gate.py` | `CERTIFIED/RELEASE/NEEDS_REVIEW/BLOCKED`; thresholds in `configs/qa_thresholds.yaml`; inherits Sync verdict as a ceiling |
| **Provenance** | `package/provenance.py` | SHA-256 per file → Merkle root → Ed25519 signature (HMAC fallback flagged unsigned) |
| **Topology provenance** | `package/{manifest,provenance,datacard}.py`, `materialize/silver_writer.py` | Per-device `{device_id, topology_ref, topology_hash}` written into Silver KV (`topology`), `manifest.json`, the signed `provenance.sidecar.json`, and the data card (`## Topology`) — dataset → descriptor → hardware revision lineage |
| **Reproducibility** | `package/versioning.py` | Order-independent content hash; same inputs → identical hash |
| **ExportRecord write-back** | `package/manifest.py` | Appends one `ExportRecord` per format into the `Session` |
| **LeRobot export** | `exporters/lerobot.py` | v3 (default) / v2 layout flag; `info.json` + `episodes.jsonl` + chunked parquet; optional lerobot-version cross-check |
| **MCAP export** | `exporters/mcap.py` | One JSON channel per stream, valid frames only, full-fidelity replay |
| **RLDS export** | `exporters/rlds.py` | Portable step-dict (`steps.parquet` + `episode_metadata.json`); real `is_first/is_last/is_terminal` |
| **HDF5 export** | `exporters/hdf5.py` | robomimic-style resizable/chunked/compressed datasets; bounded writer memory |
| **Parquet export** | `exporters/parquet.py` | Canonical columnar passthrough |
| **Derived export** (opt-in) | `exporters/derived.py` | Topology-dependent proxies per cluster from raw B (`derived.<cluster>.{normal_proxy,shear_x,shear_y,shear_mag,centroid_x_m,centroid_y_m}`); the only component that consumes the descriptor's spatial layout (positions/clusters via `sentrix_contracts`). Records formula + `derived_version` + descriptor hash + cluster map; canonical Silver stays raw-only |
| **Dataset inspection** | `inspect/summary.py` | Coverage, confidence, value ranges, QA verdict, content hash |
| **Dataset diffing** | `inspect/diff.py` | Content-hash identity, per-stream coverage/confidence deltas |

---

## Installation

**Python:** 3.11+ (developed and tested on CPython 3.12).

**Runtime dependencies** (from `pyproject.toml`): `numpy>=1.26`, `pyarrow>=14`,
`pyyaml>=6.0`, `typer>=0.9`, plus **`sentrixsync`** (provides the `SyncResult` /
`Session` / timeline types this engine consumes).

**Optional dependencies (extras):**

| Extra | Pulls | Needed for |
|---|---|---|
| `mcap` | `mcap>=1.1` | MCAP export + `mcap://` payload resolution |
| `hdf5` | `h5py>=3` | HDF5 export |
| `sign` | `cryptography>=42` | Ed25519-signed provenance (otherwise HMAC fallback → unsigned-grade → gate BLOCKs) |
| `dev` | `pytest`, `mcap`, `h5py`, `cryptography` | Running the full test suite |

### Editable install

```bash
# 1. install SentrixSync (path dependency)
pip install -e /path/to/SentrixSync

# 2. install SentrixDataEngine with the dev extras
pip install -e "/path/to/SentrixDataEngine[dev]"
```

> **Note:** there is no published wheel; both repos are installed editable from
> source. In this workspace the packages were installed into the SentrixSim
> virtual environment (CPython 3.12) which already carried `numpy`/`pyarrow`/`mcap`.

Verify:

```bash
python -c "import sentrixsync, sentrixdataengine; print(sentrixdataengine.__version__)"
sentrixdataengine version
```

---

## Quick Start

### 1. Produce a `SyncResult` with SentrixSync

```python
import numpy as np
from sentrixsync.core import (
    ClockDescriptor, DeviceDescriptor, DeviceRegistration, DeviceRole,
    EvidenceTier, Kernel, Origin, Session, SessionMetadata, StreamDescriptor,
)
from sentrixsync.sync.engine import synchronize

descriptor = DeviceDescriptor(
    device_id="glove_L", modality="tactile", producer="sentrixsim",
    is_synthetic=True, reference_candidate=True,
    clock=ClockDescriptor(clock_id="glove_L_hub", resolution_us=1,
                          nominal_epoch="session_start"),
    evidence_tiers=[EvidenceTier.SHARED_EVENT, EvidenceTier.WALL_CLOCK],
    streams=[StreamDescriptor(
        stream_id="tactile_field", device_id="glove_L", kind="tactile_field",
        kernel=Kernel.CONTINUOUS, payload_kind="bmm350_cluster_uT", units="uT",
        nominal_rate_hz=1600.0, payload_shape=[21, 3], subframe_capable=True)])

session = Session(
    metadata=SessionMetadata(session_id="01J9SYNTH0001", origin=Origin.SYNTHETIC,
                             producers=["sentrixsim"], grid_rate_hz=1600,
                             rejection_tolerance_us=1875),
    devices=[DeviceRegistration(
        device_id="glove_L", role=DeviceRole.REFERENCE, descriptor=descriptor,
        stream_refs={"tactile_field": "/abs/path/to/episode.parquet"})])

ts = np.arange(16, dtype=np.int64) * 625   # device-local timestamps, 1600 Hz
sync_result = synchronize(
    reference_device_id="glove_L", descriptors={"glove_L": descriptor},
    stream_timestamps={("glove_L", "tactile_field"): ts}, sync_events=[],
    grid_rate_hz=1600.0, rejection_tolerance_us=1875)
```

### 2–3. Materialize + export (one call)

```python
from pathlib import Path
from sentrixdataengine import Pipeline, MaterializationRequest

result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result,
    session=session,                       # supplies descriptors + payload stream_refs
    out_root=Path("gold"),
    formats=("lerobot", "mcap", "rlds", "hdf5", "parquet"),
    format_options={"lerobot": {"layout": "v3"}},
))

print(result.qa.gate_verdict)              # e.g. "CERTIFIED"
print(result.layout.base)                  # gold/dataset=<id>/version=<semver>/
```

### 4. Inspect

```bash
sentrixdataengine inspect --dataset "gold/dataset=<id>/version=0.1.0"
sentrixdataengine validate --dataset "gold/dataset=<id>/version=0.1.0"
sentrixdataengine diff --a <ver_dir_A> --b <ver_dir_B>
```

> **CLI scope:** end-to-end `materialize` is driven from Python because it requires
> an in-memory `SyncResult` (a `Session` manifest alone does not carry the timeline
> grid). The CLI covers `version`, `formats`, `validate`, `inspect`, `diff`. A
> manifest-only `materialize` entry point lands once timelines are persisted on disk.

---

## Repository Structure

```
SentrixDataEngine/
├── pyproject.toml                 # package metadata, deps, extras, console script
├── README.md
├── configs/
│   ├── qa_thresholds.yaml         # release-gate bands (hard_fail/release/certified)
│   └── dataset_profiles/          # named materialization profiles
│       ├── lerobot_v3.yaml
│       └── full_export.yaml
├── docs/
│   ├── CANONICAL_SCHEMA.md        # frozen Silver schema contract (v1.0)
│   └── USER_GUIDE.md              # operational guide
├── src/sentrixdataengine/
│   ├── __init__.py                # __version__, SCHEMA_VERSION, public exports
│   ├── contracts.py               # MaterializationRequest, DatasetSpec, ExportResult,
│   │                              #   QAReport, PayloadResolver protocol
│   ├── pipeline.py                # Pipeline.run(): resolve→materialize→validate→export→package
│   ├── cli.py                     # typer CLI (version/formats/validate/inspect/diff)
│   ├── resolve/                   # payload resolution (SentrixSync deferred item #1)
│   │   ├── resolver.py            #   ResolverRegistry + scheme dispatch
│   │   ├── parquet_resolver.py    #   parquet:// / file:// → ndarray (self-describing, sensor_id-keyed)
│   │   └── mcap_resolver.py       #   mcap:// → ndarray (producer channel layout)
│   ├── materialize/               # Silver materialization (deferred items #2–#4)
│   │   ├── canonical.py           #   CanonicalTable / CanonicalStream (the source of truth)
│   │   ├── projector.py           #   apply StreamAlignment join indices → canonical
│   │   ├── confidence.py          #   fold three-component confidence
│   │   ├── subframe.py            #   [R,*shape] sub-frame tactile bucketing
│   │   └── silver_writer.py       #   write canonical → Silver Parquet
│   ├── exporters/                 # Gold projections
│   │   ├── base.py                #   Exporter ABC + @register_exporter registry
│   │   ├── parquet.py  lerobot.py  mcap.py  rlds.py  hdf5.py  derived.py
│   ├── validate/                  # validation + QA
│   │   ├── schema_check.py  timeline_check.py  metadata_check.py  confidence_check.py
│   │   └── release_gate.py        #   composes QAReport verdict
│   ├── package/                   # packaging, provenance, versioning
│   │   ├── layout.py              #   Gold directory layout
│   │   ├── manifest.py            #   dataset manifest + ExportRecord write-back
│   │   ├── provenance.py          #   Merkle root + Ed25519/HMAC signature
│   │   ├── datacard.py            #   human-readable DATACARD.md
│   │   └── versioning.py          #   dataset_id derivation + content hash
│   ├── inspect/                   # inspection / diff
│   │   ├── summary.py  diff.py
│   └── hooks/                     # Phase-4 seams (local no-ops)
│       ├── authorize.py  watermark.py
└── tests/
    ├── conftest.py                # fixtures: self-describing producer parquet + real SyncResult
    ├── test_resolver.py  test_resolver_topology.py  test_projector.py  test_subframe.py
    ├── test_release_gate.py  test_provenance.py  test_pipeline_e2e.py
    ├── test_exporters_v2.py  test_derived_exporter.py  test_topology_provenance.py
    ├── test_mcap_resolver.py  test_inspect.py  test_multi_device.py
```

---

## Supported Formats

| Format | Purpose | Status |
|---|---|---|
| **Silver Parquet** (`silver/aligned/part-000.parquet`) | Canonical aligned source of truth; always written | ✅ Stable |
| **LeRobot** (v3 default, v2 flag) | VLA / dexterous-manipulation training (PyTorch) | ✅ Stable |
| **MCAP** | Full-fidelity replayable multimodal log | ✅ Stable (needs `mcap` extra) |
| **RLDS** | World/reward-model + RL (step-dict; portable, TF-free) | ✅ Stable (TFDS wrapper deferred) |
| **HDF5** | robomimic-compatible single-lab pipelines | ✅ Stable (needs `h5py` extra) |
| **Parquet** (Gold passthrough) | Plain columnar analytics | ✅ Stable |
| **Derived** (opt-in, `formats=("derived",)`) | Topology-dependent per-cluster proxies from raw B (normal/shear/centroid) | ✅ Stable (needs a topology descriptor + `sentrix_contracts`) |

---

## Testing

The suite uses real `SyncResult` objects (built via `sentrixsync.sync.engine.synchronize`)
over a synthetic self-describing producer-layout Parquet fixture — no mocks of the boundary.

```bash
pip install -e ".[dev]"          # pytest + mcap + h5py + cryptography
pytest tests -q
```

Expected:

```
.........................................                                [100%]
41 passed
```

Coverage by area: payload resolution (including self-describing topology-driven
columns), projector/join application, sub-frame bucketing, release gate (including
BLOCKED on fabricated gaps and unsigned lineage), provenance/Merkle, topology
provenance, end-to-end pipeline + reproducible content hash, V2 exporters
(HDF5/RLDS/LeRobot layout), derived exporter, MCAP resolver, inspect/diff,
multi-device.

---

## Roadmap

### Completed

**V1**
- Payload resolution (`parquet://` / `file://`)
- Canonical Silver materialization + Silver Parquet
- Exporters: LeRobot v3, MCAP, Parquet
- Validation + QA release gate
- Packaging: Gold layout, dataset manifest, Merkle + Ed25519 provenance, data card,
  reproducible content hash, `ExportRecord` write-back

**V2**
- Exporters: HDF5 (resizable/chunked), RLDS (portable step-dict)
- `mcap://` payload resolver
- LeRobot v2/v3 layout flag + lerobot-version `info.json` cross-check
- `inspect` + `diff` CLI commands
- Per-format `format_options`; dataset profiles in `configs/dataset_profiles/`
- Self-describing, topology-driven payload resolution (sensor_id-keyed columns from
  producer KV metadata; legacy fixed-column layout via fallback)
- Topology provenance (per-device `topology_ref`/`topology_hash`) closed through
  Silver KV, manifest, signed sidecar, and data card
- Opt-in `derived` exporter (topology-dependent per-cluster proxies)

### Future — V3 (blocked on upstream inputs, not effort)

- Video / AV1 encoding + fidelity tiers (requires an egocentric vision stream upstream)
- Per-licensee traitor-tracing + radioactive-data watermark (requires the Phase-4 catalog)
- Redaction/anonymization seam (PII models live elsewhere)
- Rerun `.rrd` Silver mirror for query/viz
- Streaming/incremental materialization for very large sessions
- Manifest-only CLI `materialize` once timelines persist to disk

---

## License

Proprietary (Internal Engineering).

## See also

- `docs/USER_GUIDE.md` — concepts, workflows, troubleshooting, design principles
- `docs/CANONICAL_SCHEMA.md` — the frozen Silver schema contract
