# Sentrix Ecosystem Guide

**An internal architecture handbook for the Sentrix data-generation stack.**

This document explains how the three repositories — **SentrixSim**, **SentrixSync**,
and **SentrixDataEngine** — compose into a single pipeline that turns a simulated
(eventually real) visuotactile glove into versioned, validated, ML-ready datasets.
It documents the system *as implemented*, and maps it onto the long-term *Sentrix
Physical Data Engine Architecture Manual* without inventing capability that does
not yet exist.

> Audience: a new engineer who needs to understand the whole stack before touching
> any one repo.

Related reading:
- `README.md` — SentrixDataEngine overview + install
- `docs/USER_GUIDE.md` — SentrixDataEngine operational workflows
- `docs/CANONICAL_SCHEMA.md` — the frozen Silver schema contract
- *Sentrix Physical Data Engine Architecture Manual* (`Final/…docx`) — the 10-phase vision

---

## Section 1 — Ecosystem Overview

```
   ┌──────────────┐        ┌──────────────┐        ┌────────────────────┐
   │  SentrixSim  │ ─────▶ │  SentrixSync │ ─────▶ │  SentrixDataEngine  │
   └──────────────┘        └──────────────┘        └────────────────────┘
   simulate ONE device      synchronize N devices    materialize + export +
   → raw episode streams     → reference timeline +    validate + package
     (parquet / mcap)          join indices + conf      → Silver + Gold datasets
```

| Repo | Single responsibility | Input | Output |
|---|---|---|---|
| **SentrixSim** | Forward-simulate the Mark 2 glove | event configs, topology, parameter registry, seeds | per-device `Episode` → Parquet / MCAP / LeRobot files |
| **SentrixSync** | Reconcile device clocks onto one timeline | device-local timestamps + sync events | in-memory `SyncResult` + `Session` manifest |
| **SentrixDataEngine** | Turn a synchronized session into datasets | `SyncResult` + `Session` (+ payload refs) | Silver canonical Parquet + Gold exports + manifest + provenance |

### Why the repositories are separated

The separation is deliberate and load-bearing:

1. **Different change cadences and dependencies.** Simulation needs sensor physics
   (scipy, dipole models). Synchronization needs clock math and graph algorithms.
   Dataset materialization needs format/IO stacks (pyarrow, h5py, mcap, ffmpeg
   later). Fusing them would couple unrelated dependency trees and release cycles.

2. **A one-directional boundary you can reason about.** Data flows strictly
   left→right. SentrixSync depends on SentrixSim's *output artifacts* (never its
   code). SentrixDataEngine depends on SentrixSync's *types* (never SentrixSim).
   No repo reaches backwards.

3. **Each layer is independently testable and swappable.** SentrixSim can be
   replaced by a real hardware producer; SentrixSync stays identical because it
   only sees descriptors + timestamps + payload refs. SentrixDataEngine stays
   identical because it only sees a `SyncResult`.

4. **The "two moats" discipline (manual §0).** The manual says build only the
   Physical Data Engine and the catalog; rent commodity layers. Keeping
   simulation, synchronization, and dataset generation in separate repos keeps the
   moats sharply scoped and prevents scope creep into a monolith.

---

## Section 2 — SentrixSim

**Purpose.** A forward-model simulator for the Sentrix Mark 2 visuotactile glove
(21× BMM350 magnetometers + 3× LIS2DTW12 accel/temp sensors). Python 3.11,
numpy/scipy/pyarrow/mcap/pydantic.

### Architecture — an L0→L7 forward pipeline

| Layer | Module | Responsibility | Output |
|---|---|---|---|
| L0 | `events/generator.py` | Ground-truth force/kinematics from an event YAML on a 1600 Hz master grid | `GroundTruth` |
| L1 | `layers/l1_contact.py` | Normalized contact wrench → magnet kinematics (lumped compliance) | `Deformation` |
| L2 | `layers/l2_field.py` | Dipole 1/r³ magnetic-field model | `B_true[T,21,3]` µT |
| L3 | `layers/l3_bmm350.py` | BMM350 model: noise, saturation, quantization, dropout | `B_read_uT`, `B_lsb`, `sat_flag`, `dropout` |
| L4 | `layers/l4_lis2dtw12.py` | LIS2DTW12 accel + temperature model | `accel_read_g`, `temp_read_c` |
| L5 | `layers/l5_noise_drift.py` | Gaussian noise + per-episode static drift | `NoiseModel` |
| L6 | `layers/l6_sync.py` | Master-grid assembly, zero-order hold (latest-at) | `aligned` dict |
| L7 | `layers/l7_export/` | Export to Parquet / MCAP / LeRobot | files |

### What an Episode is

An `Episode` (`core_types.py`) is one simulation run of one event over its full
duration:

```python
@dataclass
class Episode:
    name: str                          # e.g. "tap__d0_n0_r0"
    meta: dict[str, Any]               # seed, duration_s, physics_fidelity, units, rates
    t_master_us: np.ndarray            # (T,) int64 microseconds
    aligned: dict[str, np.ndarray]     # B_read_uT[T,21,3], accel_read_g[T,3,3], temp_read_c[T,3], masks, phase_id
    labels: dict[str, np.ndarray]      # label.* (ground truth) + est.* (estimates) per finger
    label_meta: dict[str, dict]        # {source, units, confidence, tier}
    provenance: list[dict]             # full parameter table
```

A **sample** is one row on the master grid at index `t` (1600 Hz → 625 µs spacing):
a timestamp plus B-field (21×3), tripod accel (3×3), temp (3), phase, labels.

### Sensor models (fidelity discipline)

Every parameter is tiered **KNOWN / ESTIMATED / UNKNOWN** with a confidence score
(`params/registry.py`). The simulator refuses to silently invent unknowns:
magnitudes are "relative/shape-only" presentation scales unless run with
`--allow-placeholders`. `meta["physics_fidelity"]` is one of
`relative | placeholder | relative+hardmode`.

### Supported exports (`layers/l7_export/`)

- **Parquet** (`parquet.py`) — one flat table per episode: `t_master_us`, 63
  flattened tactile columns `tactile.bNN.{bx,by,bz}_uT`, 9 accel `dyn.{thumb,index,middle}.{ax,ay,az}_g`,
  3 temp, validity masks, `phase_id`, `label.*`/`est.*`; schema-level JSON metadata.
- **MCAP** (`mcap.py`) — 3 JSON channels: `tactile_field` (400 Hz), `dynamics_accel`
  (1600 Hz), `dynamics_temp` (50 Hz).
- **LeRobot v3** (`lerobot.py` single-episode, `lerobot_dataset.py` multi-episode
  buffered) — `meta/info.json` + `meta/episodes.jsonl` + `data/chunk-NNN/file-NNN.parquet`.

### CLI

```
sentrixsim simulate --event tap --out ./out --formats parquet,mcap,lerobot
sentrixsim simulate-all --out ./out
sentrixsim build-dataset --out ./out [--n-noise 5 --n-drift 4 --hard-mode]
sentrixsim list-events
sentrixsim show-params [--tier KNOWN|ESTIMATED|UNKNOWN]
```

### Repository structure (abridged)

```
SentrixSim/
├── src/sentrixsim/
│   ├── cli.py  core_types.py  pipeline.py  dataset.py  decode.py  topology.py  hardmode.py
│   ├── params/registry.py            # KNOWN/ESTIMATED/UNKNOWN tiering
│   ├── events/generator.py           # L0 ground truth
│   └── layers/l1..l6 + l7_export/{schema,parquet,mcap,lerobot,lerobot_dataset}.py
├── configs/{parameters,topology_layoutB,scene_default,scene_hard}.yaml + events/*.yaml (9 gestures)
└── tests/
```

### Limitations (relevant downstream)

Single-device only; no absolute physics (magnitudes are scales); no vision stream;
exporters are **single-device, single-episode** — unaware of multi-device
synchronization, confidence masks, or sub-frame bucketing. This is precisely why a
separate materialization engine exists.

---

## Section 3 — SentrixSync

**Purpose.** Modality-agnostic multi-device timeline synchronization. It models the
full inverse lifecycle: ingest device-local timestamps → gather sync evidence → fit
per-device affine clock models → build a reference-time grid with as-of joins →
compute confidence + QA → emit a `Session` manifest.

### Synchronization architecture

```
S0 ingest    DeviceAdapter → Sample(t_device_us, payload_ref, seq, meta)
S1 evidence  HARDWARE_PTP | SHARED_EVENT (detectors) | WALL_CLOCK
S2 estimate  per-device ClockModel (alpha, beta_us); Dijkstra spanning tree to reference
S3 correct   t_device_us → t_ref_us  (t_ref = alpha*t + beta)
S4 timeline  master grid @ grid_rate_hz → as-of join per stream → StreamAlignment
S5 metrics   sync_resid_us, coverage, dropout, roundtrip_accuracy, QA gate
S6 emit      SyncResult (in memory) + Session manifest
```

**Invariants:** timestamps are device-local only (never pre-corrected); payloads
carried by reference; reference clock = designated anchor; gaps beyond tolerance
are flagged, not fabricated; confidence has three separate components. The core
**never branches on `modality`** — only `kernel`, `nominal_rate_hz`, `units`,
`payload_kind` affect behavior.

### Adapters

`ingest/adapter.py` defines the `DeviceAdapter` ABC (`descriptor/open/close/read/read_batch`,
optional `stream_ref`/`ground_truth`). `SentrixSimAdapter.from_parquet(path, descriptor,
ts_column="t_master_us")` wraps a SentrixSim Parquet episode and emits samples whose
`payload_ref` looks like `parquet:///abs/path#stream=tactile_field&row=12`. This
adapter is the **only** connection point between Sim and Sync; it reads Sim output,
never imports Sim.

### Detectors

`detect/detector.py` provides a `@register_detector` registry and a
`SyncEventDetector` ABC (`detect(t_us, signal) → Detection`). Detectors are the
*only* code permitted to read payloads; they emit `SyncEvent`s. Shipped:
`TactileTap`, `VisualFlash`.

### Clock models

`clock/estimate.py` fits affine `t_ref = alpha·t_local + beta` via TLS (default) or
RANSAC (robust). The reference device uses `identity_model` (`alpha=1, beta=0,
clock_confidence=1.0`). `graph.py` reconciles arbitrary co-observation topologies
into a spanning tree; unreachable devices degrade gracefully.

### Confidence propagation

`sync/confidence.py` keeps three authoritative components per grid point:
`source` (raw-sample trust), `clock` (`base · exp(-d/τ)`, decaying with distance to
the nearest sync event), `interpolation` (`clip(1 − gap/tolerance, 0, 1)`, 0 at
gaps). `derived_scalar()` = `source·clock·interp` is **export-only**.

### Timeline generation — key structures

```python
@dataclass
class BuiltTimeline:
    reference_clock_id: str
    grid_us: np.ndarray                    # int64 reference-time grid
    per_stream: dict[str, StreamAlignment] # key = "device::stream"

@dataclass
class StreamAlignment:                     # per-grid-point join result for one stream
    stream_id: str
    kernel: Kernel                         # CONTINUOUS | HOLD
    source_index: np.ndarray               # int64, -1 = no sample within tolerance
    next_index: np.ndarray                 # int64, -1 = none (CONTINUOUS only)
    weight: np.ndarray                     # float [0,1] interp weight
    valid: np.ndarray                      # bool, False at gaps
    interp_confidence: np.ndarray          # float [0,1], decays with gap
```

The engine output:

```python
@dataclass
class SyncResult:
    reference_device_id: str
    reference_clock_id: str
    clock_models: dict[str, ClockModel]
    timeline: BuiltTimeline
    confidence: dict[str, ConfidenceComponents]
    sync_report: SyncReport
    validation_report: ValidationReport      # gate_verdict: CERTIFIED|RELEASE|NEEDS_REVIEW|BLOCKED
    diagnostics: ReconcileDiagnostics
    metrics: dict                             # sync_resid_us, coverage, dropout, ...
```

A **`Session`** (`core/session.py`) is the persistent manifest: `metadata`,
`devices` (each `DeviceRegistration` with a `DeviceDescriptor` + `stream_refs`),
`calibration_refs`, `timeline: TimelineRef`, `sync_report`, `validation_report`,
`exports: list[ExportRecord]`, `ground_truth`.

### QA

`sync/metrics.py` + `config.GateThresholds` compute the sync-level verdict
(`gate(sync_resid_us, coverage_min, dropout_max)`), and property checks
(`grid_monotonic`, `bounded_step`, `no_fabricated_gaps`).

### What SentrixSync deliberately does NOT do

Its own v0.3 "deferred / not callable" list:
- **payload resolution** (URI handles carried, not resolved)
- **sub-frame bucketing materialization** (`subframe_buckets` computes indices, not values)
- **materialized aligned-table export** (`TimelineRef.aligned_table_uri` slot left unfilled)
- **multimodal export** (LeRobot/MCAP — "framework ready, consumer not yet integrated")

It does not generate datasets, build a catalog, view data, or label. Those are
out of charter — and are exactly what SentrixDataEngine picks up.

---

## Section 4 — SentrixDataEngine

**Why it exists.** SentrixSync produces a synchronized timeline as an *in-memory
object with join indices* — not data on disk. The four items it explicitly defers
(above) plus the manual's Phase 5 (export), Phase 7.3 (QA gate), and Phase 6.3a
(provenance) are an entire repository's worth of work. SentrixDataEngine is that
repository: it consumes `SyncResult`/`Session` and emits datasets.

| Layer | Module | Responsibility |
|---|---|---|
| **Payload resolution** | `resolve/` | `payload_ref` URI → ndarray (`parquet`, `file`, `mcap` schemes) |
| **Materialization** | `materialize/projector.py` | apply `StreamAlignment` join indices to resolved payloads |
| **Canonical Silver** | `materialize/{canonical,silver_writer}.py` | one `CanonicalTable` → one Silver Parquet (the source of truth) |
| **Sub-frame bucketing** | `materialize/subframe.py` | `[R,*shape]` tactile burst per anchor frame (reuses Sync's index logic) |
| **Confidence fold** | `materialize/confidence.py` | carry source/clock/interp + export scalar |
| **Exporters** | `exporters/` | project canonical → LeRobot / RLDS / HDF5 / MCAP / Parquet |
| **Validation** | `validate/` | schema/timeline/metadata/confidence + `release_gate` |
| **Packaging** | `package/` | Gold layout, manifest, Merkle+Ed25519 provenance, data card, content hash |
| **Hooks** | `hooks/` | authorize / watermark seams (local no-ops; Phase-4 later) |

**Boundary mechanics:** imports SentrixSync types read-only; never imports SentrixSim;
reaches Sim bytes only via `payload_ref` resolution; the only write-back is appending
an `ExportRecord` to a `Session`.

---

## Section 5 — End-to-End Workflow

```
Generate Episode  →  Synchronize Devices  →  Materialize Dataset
                                                     │
                                  Export Formats  ──▶ Validate  ──▶ Package
```

### Step 1 — Generate an episode (SentrixSim)

```bash
sentrixsim simulate --event tap --out ./out --formats parquet
# → ./out/parquet/tap/tap__d0_n0_r0.parquet
```

### Step 2 — Synchronize (SentrixSync)

```python
import numpy as np
from sentrixsync.core import (ClockDescriptor, DeviceDescriptor, DeviceRegistration,
    DeviceRole, EvidenceTier, Kernel, Origin, Session, SessionMetadata, StreamDescriptor)
from sentrixsync.sync.engine import synchronize

descriptor = DeviceDescriptor(
    device_id="glove_L", modality="tactile", producer="sentrixsim",
    is_synthetic=True, reference_candidate=True,
    clock=ClockDescriptor(clock_id="glove_L_hub", resolution_us=1, nominal_epoch="session_start"),
    evidence_tiers=[EvidenceTier.SHARED_EVENT, EvidenceTier.WALL_CLOCK],
    streams=[StreamDescriptor(stream_id="tactile_field", device_id="glove_L",
        kind="tactile_field", kernel=Kernel.CONTINUOUS, payload_kind="bmm350_cluster_uT",
        units="uT", nominal_rate_hz=1600.0, payload_shape=[21, 3], subframe_capable=True)])

session = Session(
    metadata=SessionMetadata(session_id="01J9SYNTH0001", origin=Origin.SYNTHETIC,
        producers=["sentrixsim"], grid_rate_hz=1600, rejection_tolerance_us=1875),
    devices=[DeviceRegistration(device_id="glove_L", role=DeviceRole.REFERENCE,
        descriptor=descriptor,
        stream_refs={"tactile_field": "out/parquet/tap/tap__d0_n0_r0.parquet"})])

ts = np.arange(16, dtype=np.int64) * 625
sync_result = synchronize(reference_device_id="glove_L", descriptors={"glove_L": descriptor},
    stream_timestamps={("glove_L", "tactile_field"): ts}, sync_events=[],
    grid_rate_hz=1600.0, rejection_tolerance_us=1875)
```

### Steps 3–6 — Materialize, export, validate, package (SentrixDataEngine)

```python
from pathlib import Path
from sentrixdataengine import Pipeline, MaterializationRequest

result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result, session=session, out_root=Path("gold"),
    formats=("lerobot", "mcap", "rlds", "hdf5", "parquet"),
    format_options={"lerobot": {"layout": "v3"}}))

print(result.qa.gate_verdict)     # CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED
print(result.layout.base)         # gold/dataset=<id>/version=0.1.0/
```

Inspect / validate / diff:

```bash
sentrixdataengine inspect  --dataset "gold/dataset=<id>/version=0.1.0"
sentrixdataengine validate --dataset "gold/dataset=<id>/version=0.1.0"
sentrixdataengine diff --a <ver_dir_A> --b <ver_dir_B>
```

Packaged output:

```
gold/dataset=<id>/version=0.1.0/
├── silver/aligned/part-000.parquet     # canonical source of truth
├── format=lerobot/  meta/info.json  data/chunk-000/file-000.parquet
├── format=mcap/     session-000.mcap
├── format=rlds/     steps.parquet  episode_metadata.json
├── format=hdf5/     dataset.hdf5
├── format=parquet/  part-000.parquet
├── manifest.json  provenance.sidecar.json  DATACARD.md  qa_report.json
```

---

## Section 6 — Data Flow Deep Dive

Trace one tactile reading at master-grid sample `t` for cluster `b05`, axis `bz`,
through the whole stack.

### 1. SentrixSim — physics → reading → file

```
L0 ground truth:  normal-force profile at time t (normalized [0,1])
L1 deformation:   force → magnet displacement dz/dx/dy (mm)
L2 field:         dipole 1/r³ → B_true[t, 5, 2]  (cluster 5, z-axis, µT)
L3 BMM350:        + drift + Gaussian noise → clip ±2000 µT → quantize 0.1 µT
                  → B_read_uT[t, 5, 2]
L6 sync:          held onto the 1600 Hz master grid
L7 parquet:       column  tactile.b05.bz_uT  at row t
```

The written cell is `episode.parquet[row=t]["tactile.b05.bz_uT"] = <float32 µT>`.

### 2. SentrixSync — reference, not value

```
SentrixSimAdapter.from_parquet reads ONLY the t_master_us column → Sample for row t:
    Sample(stream_id="tactile_field", t_device_us = t_master_us[t],
           payload_ref = "parquet:///abs/episode.parquet#stream=tactile_field&row=t")
synchronize():
    clock model (reference glove → identity)  t_ref = 1·t_device + 0
    grid_us built at 1600 Hz over the reference span
    asof_join(CONTINUOUS):  for each grid point i →
        source_index[i], next_index[i], weight[i], valid[i], interp_confidence[i]
    confidence:  source[i], clock[i], interpolation[i]
```

SentrixSync never opened the `bz` value. It produced *indices and confidence* that
say "grid point `i` is built from device sample `source_index[i]` (and `next_index[i]`
with `weight[i]`), it is valid, and here is how much to trust it."

### 3. SentrixDataEngine — resolve, apply, carry

```
ParquetPayloadResolver.resolve_stream(base_uri, "bmm350_cluster_uT", (21,3)):
    reads tactile.b00..b20 {bx,by,bz} columns → payload[N, 21, 3]   (b05/bz lives at [:,5,2])
projector._apply_alignment (CONTINUOUS):
    values[i, 5, 2] = payload[source_index[i], 5, 2] * (1-weight[i])
                    + payload[next_index[i],   5, 2] *   weight[i]      (where valid)
    values[i, 5, 2] = NaN                                              (where invalid — never faked)
confidence.fold:
    confidence[i] = source[i] * clock[i] * interpolation[i]
silver_writer:
    flatten (21,3) → 63 columns; b05/bz becomes  tactile_field.c017   (= 5*3 + 2)
    + tactile_field.valid, tactile_field.confidence  →  silver/aligned/part-000.parquet
```

### 4. LeRobot export — projection

```
LeRobotExporter reads the canonical table (NOT the timeline, NOT the raw parquet):
    observation.tactile_field         → list column, per row the flattened [21,3] (63 floats); b05/bz at index 17
    observation.tactile_field.confidence → float column = confidence[i]
    timestamp                          → grid_us[i] / 1e6  (seconds)
    frame_index, episode_index=0, index, task_index=0
    meta/info.json features:  "observation.tactile_field": {"dtype":"float32","shape":[21,3],"units":"uT"}
```

The same `bz` reading, originally a noisy quantized µT cell in a single-device
episode, is now a confidence-tagged, reference-time-aligned tensor element inside a
LeRobot frame — with a NaN-flagged gap wherever synchronization could not vouch for
it, and a Merkle-signed provenance record for the file that contains it.

---

## Section 7 — Repository Boundaries

This section is normative. Treat it as the contract.

### SentrixSim — belongs / forbidden

**Belongs:** glove physics and sensor models (L0–L6); event/gesture configs;
parameter registry with KNOWN/ESTIMATED/UNKNOWN tiering; single-device episode
generation; convenience single-device exporters; Hard Mode augmentations.

**Must never be added:** multi-device synchronization or clock reconciliation; any
import of SentrixSync or SentrixDataEngine; cross-device timeline logic; dataset
catalog/validation/packaging concerns; resolving another producer's payload refs.

### SentrixSync — belongs / forbidden

**Belongs:** ingestion adapters; sync-event detectors; clock estimation
(TLS/RANSAC/identity); graph reconciliation; reference-grid construction; as-of
join (HOLD/CONTINUOUS); three-component confidence; sync-level QA + reports;
`Session`/`SyncResult` schemas and manifest (de)serialization.

**Must never be added:** payload resolution / reading bulk values (only detectors
touch signals, and only to emit events); dataset materialization or export to
LeRobot/RLDS/HDF5; format writers; a viewer or labeling system; branching on
`modality`; any import of SentrixSim or SentrixDataEngine; mutating data into a
dataset shape. *SentrixSync is synchronization infrastructure only.*

### SentrixDataEngine — belongs / forbidden

**Belongs:** payload resolution; canonical Silver materialization; sub-frame
bucketing materialization; confidence folding; format exporters; dataset-level
validation + release gate; packaging (manifest, provenance, versioning, data card);
inspection/diff; Phase-4 hook seams.

**Must never be added:** clock fitting, grid building, or re-running the as-of join
(consume Sync's indices, never re-derive); any import of SentrixSim; mutation of a
`SyncResult` (the only write-back is appending an `ExportRecord`); simulation;
perception/autolabeling models; a training loop; a storefront/commerce engine;
fabricating values across a flagged gap.

---

## Section 8 — Current Status

### Implemented

- **SentrixSim:** full L0–L7 forward pipeline; 9 gestures; KNOWN/ESTIMATED/UNKNOWN
  tiering; Parquet/MCAP/LeRobot (single + multi-episode) exporters; Hard Mode; CLI.
- **SentrixSync:** ingest adapters incl. `SentrixSimAdapter`; TactileTap/VisualFlash
  detectors; TLS/RANSAC/identity clock models; graph reconciliation; reference grid
  + as-of join; three-component confidence; sync-level QA; `Session`/`SyncResult`
  schemas + manifest I/O.
- **SentrixDataEngine (V1+V2):** payload resolution (`parquet`/`file`/`mcap`);
  canonical Silver materialization + Silver Parquet; sub-frame bucketing capability;
  exporters LeRobot (v2/v3), MCAP, RLDS, HDF5, Parquet; schema/timeline/metadata/
  confidence validation + release gate; packaging with Merkle + Ed25519 provenance,
  content-hash reproducibility, data card, `ExportRecord` write-back; `inspect`/`diff`
  CLI. **24 tests passing.**

### Partially implemented

- **Sub-frame tactile bucketing** — the materializer is implemented and tested as a
  capability, but the default single-glove pipeline does not activate it because no
  *anchor* (slow, e.g. video) stream exists upstream yet. It engages when an anchor
  stream/fps is present.
- **RLDS** — full RLDS *step structure* in portable Parquet form; the thin TFDS
  `GeneratorBasedBuilder` wrapper is deferred (avoids a hard TensorFlow dependency).
- **Phase-4 hooks** (`authorize`, `watermark`) — interface seams with local no-op
  implementations; real entitlement/watermark logic awaits the catalog.
- **CLI `materialize`** — Python-only today (needs an in-memory `SyncResult`); a
  manifest-only entry point waits for on-disk timelines.

### Future work (V3 — blocked on upstream inputs, not effort)

Video / AV1 encoding + fidelity tiers (needs a vision stream); per-licensee
fingerprint + radioactive-data watermark (needs the catalog); redaction/PII seam;
Rerun `.rrd` Silver mirror; streaming materialization for very large sessions; real
hardware producers replacing the simulator behind the same adapter contract.

---

## Section 9 — Architecture Vision

The implemented stack is the first three layers of the manual's ten-phase
*Physical Data Engine* roadmap, built in the order that de-risks the moats.

```
   IMPLEMENTED TODAY                         MANUAL ROADMAP (future)
   ─────────────────                         ───────────────────────
   SentrixSim  ───────────────────────────▶  Physical hardware capture
   (synthetic glove)                          (Mark 2 glove + ego camera; PTP/SMPTE sync)
        │                                          swaps in behind the SAME adapter contract
        ▼
   SentrixSync  ──────────────────────────▶  Phase 2.1 ingestion & hardware sync
   (clock reconciliation, timeline)            (real multi-device, real evidence tiers)
        │
        ▼
   SentrixDataEngine  ────────────────────▶  Phase 5 export layer + Phase 2.2 Silver/Gold
   (materialize → export → validate           + Phase 7.3 QA gates + Phase 6.3 provenance
    → package)                                 (already the shape of these phases)
                                              
   NOT BUILT (separate moats / later):
     Phase 3  Physical Data Engine — autolabeling (SAM2 / FoundationPose / VLM, contact/force/object-state)
     Phase 4  Transactional catalog — search, bundles, pricing, entitlements
     Phase 6.3b/c  per-licensee watermark / radioactive marking
     Marketplace → Engine APIs → Embodiment interface (manual Phase 10 stages)
```

**How the pieces connect to the vision without overreach:**

- **Simulation → physical hardware.** SentrixSim exists so the rest of the stack
  can be built and validated before silicon. Because SentrixSync only consumes a
  `DeviceDescriptor` + timestamps + payload refs, a real glove/camera producer
  replaces the simulator with **no change** to Sync or DataEngine.

- **Synchronization → the manual's ingestion/sync phase.** SentrixSync already
  implements the affine clock model, evidence tiers (PTP / shared-event / wall-clock),
  and the as-of join the manual specifies. Real PTP/SMPTE inputs feed the same
  estimator.

- **Dataset generation → export + QA + provenance phases.** SentrixDataEngine is
  already the *shape* of Phase 5 (multi-format export from one canonical store),
  Phase 7.3 (release gate verdicts), and Phase 6.3a (Merkle + signed lineage). The
  entitlement/watermark *seams* are present as no-ops so the catalog can wire in
  later without re-architecting.

- **What stays out, on purpose.** Autolabeling (Phase 3) and the catalog (Phase 4)
  are the two declared moats and are *not* in any of these three repos. Training
  systems consume the Gold datasets but live entirely outside the stack. Keeping
  them out is what makes the boundaries in Section 7 enforceable.

The through-line: **ground the signal first** (contact, force, timing, confidence),
preserve it losslessly and provably (never fabricate a gap; carry confidence; sign
the bytes), and let every higher layer — catalog, marketplace, embodiment
retargeting — rest on supervision the stack has owned since the first simulated
sample.
