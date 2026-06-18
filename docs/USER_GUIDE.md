# SentrixDataEngine — User Guide

A practical, operational guide to materializing, exporting, validating, and
inspecting datasets with SentrixDataEngine. For the high-level overview and
install steps, see the top-level `README.md`. For the frozen on-disk Silver
schema, see `docs/CANONICAL_SCHEMA.md`.

---

## Concepts

### Session
A `sentrixsync.core.session.Session` — the manifest describing a synchronized
capture/generation: its `metadata` (id, origin, grid rate, rejection tolerance),
its `devices` (each a `DeviceRegistration` holding a `DeviceDescriptor` and
`stream_refs`), and pointers to timeline/reports/exports. SentrixDataEngine reads
two things from it: the **device descriptors** (for each stream's `payload_kind`,
`payload_shape`, `units`, `kernel`) and the **`stream_refs`** (the payload base
URIs). It writes back exactly one thing: an `ExportRecord` per produced format.

### SyncResult
The in-memory output of `sentrixsync.sync.engine.synchronize(...)`. The engine
consumes its `timeline` (a `BuiltTimeline` with `grid_us` and per-stream
`StreamAlignment` join indices), its `confidence` (per-stream three-component
arrays), its `reference_clock_id`, its `validation_report.gate_verdict`, and its
`metrics` (notably `sync_resid_us`). **A `SyncResult` is required** — a `Session`
manifest alone does not carry the timeline grid.

### PayloadRef
SentrixSync carries payloads *by reference*, never inline. A reference is a URI of
the form `scheme://location[#fragment]`, e.g.
`parquet:///abs/path/episode.parquet#stream=tactile_field&row=12`. SentrixDataEngine
resolves these via the `resolve/` layer. Supported schemes today: `parquet`,
`file`, `mcap`. The fragment (`#stream=..&row=..`) is stripped to get the stream
base URI; the resolver reads the whole stream once and indexes rows via the join.

### Canonical Silver
The single internal source of truth: a `CanonicalTable` (`materialize/canonical.py`)
holding the reference `grid_us`, a `frame_index`, and per-stream `CanonicalStream`
objects with `values[n_grid, *shape]` (float32, **NaN at gaps**), `valid[n_grid]`,
and confidence arrays. It is written to `silver/aligned/part-000.parquet`. Every
Gold export is a pure projection of this table — never of the timeline directly.

### Gold Exports
The target formats projected from Silver: LeRobot, RLDS, HDF5, MCAP, Parquet. Each
lives under `format=<fmt>/` in the dataset version directory.

### QA Gates
`validate/release_gate.py` composes integrity + quality + property checks into one
verdict: `CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED`. Thresholds come from
`configs/qa_thresholds.yaml`. The dataset verdict can never exceed the SentrixSync
`ValidationReport.gate_verdict` (the sync result is the ceiling).

### Confidence
SentrixSync keeps three separate, authoritative components per grid point —
**source** (trust in the raw sample), **clock** (trust in the fitted clock model),
**interpolation** (decays with the gap to the nearest real sample). SentrixDataEngine
carries all three into Silver and adds an export-only scalar
`confidence = source * clock * interp`. At a gap (`valid == False`) every component
and the scalar are 0.

---

## Common Workflows

All workflows assume a `sync_result` and `session` as built in the README Quick
Start (a single reference glove device with a `tactile_field` stream).

### Workflow 1 — Materialize a synchronized session

```python
from pathlib import Path
from sentrixdataengine import Pipeline, MaterializationRequest

result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result,
    session=session,
    out_root=Path("gold"),
    formats=("parquet",),          # at minimum, write the canonical passthrough
))

print(result.qa.gate_verdict)      # CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED
print(result.content_hash)         # reproducible content hash
print(result.layout.base)          # gold/dataset=<id>/version=0.1.0/
```

The Silver table is always written (`<base>/silver/aligned/part-000.parquet`),
regardless of which Gold formats are requested.

### Workflow 2 — Export LeRobot

```python
result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result, session=session, out_root=Path("gold"),
    formats=("lerobot",),
    format_options={"lerobot": {"layout": "v3"}},   # "v2" also supported
))
```

Output under `format=lerobot/`:
- `meta/info.json` — features (`observation.<stream>` with its `[*shape]`,
  `observation.<stream>.confidence`, `timestamp`, indices), `fps`, layout, and a
  `lerobot_validation` note (cross-checks keys if `lerobot` is installed).
- `meta/episodes.jsonl` — one record (this session = one episode).
- `data/chunk-000/file-000.parquet` (v3) or `data/chunk-000/episode_000000.parquet` (v2).

### Workflow 3 — Export MCAP

Requires the `mcap` extra (`pip install -e ".[mcap]"`).

```python
result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result, session=session, out_root=Path("gold"),
    formats=("mcap",)))
```

Writes `format=mcap/session-000.mcap` — one JSON channel per stream, messages at
the reference grid times for **valid frames only** (gaps are omitted, not faked).

### Workflow 4 — Export RLDS

No TensorFlow required. Produces the RLDS step structure in portable form.

```python
result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result, session=session, out_root=Path("gold"),
    formats=("rlds",)))
```

Output under `format=rlds/`:
- `steps.parquet` — one row per step: `observation.<stream>`,
  `observation.<stream>.confidence`, `reward`, `discount`, `is_first`, `is_last`,
  `is_terminal`, `frame_index`. (`reward`/`discount` are neutral until a reward
  signal exists upstream; boundary flags are real.)
- `episode_metadata.json`.

### Workflow 5 — Export HDF5

Requires the `hdf5` extra (`pip install -e ".[hdf5]"`).

```python
result = Pipeline().run(MaterializationRequest(
    sync_result=sync_result, session=session, out_root=Path("gold"),
    formats=("hdf5",),
    format_options={"hdf5": {"compression": "gzip"}},
))
```

Writes `format=hdf5/dataset.hdf5` with `data/demo_0/obs/<stream>` (resizable,
chunked, compressed), `<stream>.confidence`, `<stream>.valid`, `timestamp`, and
`num_samples` / `total` attrs.

### Workflow 6 — Inspect a dataset

```bash
sentrixdataengine inspect --dataset "gold/dataset=<id>/version=0.1.0"
```

Or from Python:

```python
from sentrixdataengine.inspect import summarize_dataset, summarize_canonical
print(summarize_dataset(result.layout.base))   # from a packaged dir
print(summarize_canonical(result.canonical))   # from the in-memory table
```

Reports `n_grid`, `coverage_min`, per-stream coverage / confidence_mean / value
range, plus `qa_verdict` and `content_hash` from the manifest.

### Workflow 7 — Compare dataset versions

```bash
sentrixdataengine diff --a "gold/.../version=0.1.0" --b "gold/.../version=0.2.0"
```

Or:

```python
from sentrixdataengine.inspect import diff_datasets
print(diff_datasets(dir_a, dir_b))
```

Returns `content_hash_identical`, `n_grid_delta`, both QA verdicts, coverage_min
for each, and per-stream `status` (`same|changed|added|removed`) with coverage /
confidence deltas.

---

## Troubleshooting

### Missing payload refs
**Symptom:** `KeyError: no payload source for stream 'glove_L::tactile_field'; provide it via MaterializationRequest.payload_sources or Session stream_refs`

**Cause:** the projector found a stream in `timeline.per_stream` with no resolvable
base URI. **Fix:** ensure the `Session` device's `stream_refs` maps the stream id
to a real path/URI, or pass `payload_sources={"glove_L::tactile_field": "parquet://…"}`
on the `MaterializationRequest`. Bare `.parquet` paths are auto-normalized to
`parquet://<abspath>`.

### Schema mismatch
**Symptom:** `validate` reports `stream_shapes_consistent: fail` or
`grid_frame_index_aligned: fail`, and the gate returns `BLOCKED`.

**Cause:** a stream's resolved payload shape disagrees with the descriptor's
`payload_shape`, or array lengths don't match the grid. **Fix:** confirm the
descriptor `payload_shape` matches the actual columns the resolver pulls (e.g.
`bmm350_cluster_uT` → 21×3 columns `tactile.bNN.{bx,by,bz}_uT`).

### Failed QA gate (`BLOCKED` / `NEEDS_REVIEW`)
Check `qa_report.json` → `detail` and `integrity`/`quality`. Common hard-fails
(`BLOCKED`):
- `missing_frame_pct >= 3.0` — coverage too low (gaps in the timeline).
- `sync_resid_us >= 5000` — synchronization residual too high (a SentrixSync issue).
- `label_confidence < 0.85` — mean confidence too low.
- unsigned lineage — provenance fell back to HMAC (install the `sign` extra).
- a critical check failed (`no_fabricated_gaps`, `stream_shapes_consistent`,
  `confidence_in_unit_interval`).
`NEEDS_REVIEW` means it cleared hard-fails but missed a release band; see the
bands in `configs/qa_thresholds.yaml`.

### Unsupported exporter
**Symptom:** `KeyError: unknown exporter 'foo'; registered: ['hdf5', 'lerobot', 'mcap', 'parquet', 'rlds']`

**Fix:** request a registered format. List them with `sentrixdataengine formats`
or `sentrixdataengine.exporters.registered_exporters()`.

### Missing dependency
**Symptom:** `ImportError: mcap export requires the optional 'mcap' dependency` (or
the same for `h5py`).

**Fix:** install the matching extra: `pip install -e ".[mcap]"` /
`".[hdf5]"` / `".[dev]"`.

### Unsigned provenance → BLOCKED
If `cryptography` is not installed, provenance falls back to HMAC-SHA256 and is
flagged `signed=False`; the gate hard-fails by default
(`require_lineage_signed: true`). **Fix:** `pip install -e ".[sign]"`, or relax the
band in `configs/qa_thresholds.yaml` for non-release builds.

---

## Design Principles

1. **One canonical representation.** A single Silver `CanonicalTable` is the source
   of truth; every Gold format is a pure projection of it. This guarantees format
   parity and makes reproducibility a property of one artifact.

2. **No Sync mutation.** SentrixDataEngine treats `SyncResult` as immutable. It
   never re-fits clocks, rebuilds grids, or re-runs the as-of join — it *applies*
   SentrixSync's join indices. The only write-back is appending an `ExportRecord`
   to a `Session` (additive, contract-allowed).

3. **No Sim dependency.** The engine never imports SentrixSim. It reaches raw bytes
   only by resolving the `payload_ref` URIs SentrixSync recorded, preserving the
   one-directional repository boundary.

4. **Never fabricate gaps.** Where SentrixSync marks a frame invalid, the value is
   NaN and confidence is 0 — never interpolated across a dropout. The
   `no_fabricated_gaps` check enforces this and a violation is `BLOCKED`.

5. **Confidence preservation.** The three confidence components (source, clock,
   interpolation) are carried verbatim into Silver; the single scalar is computed
   for export only and never treated as the internal source of truth.

6. **Deterministic & reproducible.** Same `Session` + same profile + same engine
   version → identical content hash. Provenance (Merkle root + signature) makes any
   delivered byte auditable.
