# SentrixDataEngine — Repository Memory

> Dataset materialization, export, validation, packaging for synchronized Sentrix
> sessions. See root `../CLAUDE.md` for ecosystem context; full docs in `README.md`,
> `docs/USER_GUIDE.md`, `docs/SENTRIX_ECOSYSTEM_GUIDE.md`, `docs/CANONICAL_SCHEMA.md`,
> `SentrixDataEngine_DESIGN.md`.

## Purpose / why it exists

Final stage: `SentrixSim → SentrixSync → SentrixDataEngine`. SentrixSync produces a
synchronized timeline as an in-memory object with **join indices** — not data on
disk. The four items Sync explicitly defers (payload resolution, aligned-table
materialization, sub-frame bucketing materialization, multimodal export) plus the
manual's Phase 5 (export), Phase 7.3 (QA gate), Phase 6.3a (provenance) ARE this
repo. It consumes `SyncResult`/`Session` and emits datasets.

- Version: `__version__ = "0.1.0"`, `SCHEMA_VERSION = "1.0"` (Silver contract).
- Deps: numpy, pyarrow, pyyaml, typer, **sentrixsync**. Extras: `mcap`, `hdf5`
  (h5py), `sign` (cryptography), `dev`.

## Architecture

```
SyncResult (+ Session)
  → resolve/      payload_ref URI → ndarray
  → materialize/  apply StreamAlignment join → CanonicalTable → Silver Parquet
  → validate/     schema/timeline/metadata/confidence + release_gate
  → exporters/    project Silver → Gold formats
  → package/      manifest + Merkle/Ed25519 provenance + datacard + content hash
```

`pipeline.py::Pipeline.run(MaterializationRequest)` orchestrates. The request carries
`sync_result`, `session`, `out_root`, `formats`, `format_options`, `payload_sources`,
`version`, `customer_id`.

## Canonical Silver schema (the source of truth — `docs/CANONICAL_SCHEMA.md` v1.0)

In-memory `CanonicalTable`: `grid_us[n], frame_index[n], streams{key→CanonicalStream},
reference_clock_id, grid_rate_hz, session_id, schema_version, extra`.
`CanonicalStream`: `key, device_id, stream_id, payload_kind, units, kernel, shape,
values[n,*shape] (float32, NaN at gaps), valid[n], confidence[n], conf_source,
conf_clock, conf_interp`.

On disk `silver/aligned/part-000.parquet`: `t_ref_us`, `frame_index`, per stream
`<stream>.cNNN` (flattened payload), `<stream>.valid`, `<stream>.confidence`. KV meta:
schema_version, session_id, reference_clock_id, grid_rate_hz, streams, source_episode_hashes.

## Exporters (`exporters/`, `@register_exporter`)

`parquet`, `lerobot` (v2/v3 layout flag + optional lerobot-version info.json check),
`mcap` (needs mcap extra), `rlds` (portable step-dict: steps.parquet +
episode_metadata.json; real is_first/is_last/is_terminal; neutral reward/discount),
`hdf5` (needs h5py; resizable/chunked/compressed). All are pure projections of the
CanonicalTable.

`derived` (opt-in; needs `sentrix_contracts`) — the ONLY exporter that consumes the
topology descriptor's SPATIAL parts (positions, clusters), loaded via the session's
`topology_ref`. Materializes topology-DEPENDENT proxies per cluster from raw B:
`derived.<cluster>.{normal_proxy, shear_x, shear_y, shear_mag, centroid_x_m,
centroid_y_m}` (ΔB vs per-sensor median baseline). Records formula + `derived_version`
+ descriptor hash + cluster map in KV meta. Canonical Silver stays raw-only; request
via `formats=("derived",)`. Topology-independent signals (|B|, dB) are recomputable
anywhere and not materialized here. Raises if no topology provenance / no magnetic
stream. Validated on real `press`: contacted-finger cluster normal proxy dominates.

## Resolvers (`resolve/`)

`ParquetPayloadResolver` (`parquet://`, `file://`), `McapPayloadResolver` (`mcap://`).
Registry in `resolver.py`; `default_registry()` wires both.

**Topology-driven (Migration Phase 3, done).** The parquet resolver is
**self-describing**: it reads per-modality sensor order (`bmm_sensor_ids` /
`lis_sensor_ids`) from the parquet KV metadata (`sentrixsim_meta` /
`sentrix_capture_meta`) and builds canonical sensor_id-keyed columns
(`mag.<sid>.{bx,by,bz}_uT`, `dyn.<sid>.{ax,ay,az}_g`, `dyn.<sid>.temp_c`).
`payload_kind` selects the modality; the column SET + ORDER come from the file, so
ANY sensor count resolves with no code change (verified: Mark2_v1 21/3 and a 6/1
revision both resolve from real Sim output). Pre-Phase-1 `tactile.bNN` /
`dyn.<finger>` datasets still resolve via a legacy fallback. The MCAP resolver is
array-based (channel payload fields) and already count-agnostic.

## Validators (`validate/`)

`schema_check`, `timeline_check` (monotonic grid, bounded step, no_fabricated_gaps),
`metadata_check`, `confidence_check` (∈[0,1], zero at gaps), `release_gate`.
Gate verdict `CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED`; thresholds in
`configs/qa_thresholds.yaml`; **inherits SentrixSync's verdict as a ceiling**
(can never exceed it).

## Packaging / provenance (`package/`)

Gold layout `dataset=<id>/version=<semver>/{format=*/, silver/, manifest.json,
provenance.sidecar.json, DATACARD.md, qa_report.json}`. Provenance: SHA-256 per file
→ Merkle root → Ed25519 signature (HMAC-SHA256 fallback flagged `signed=False`).
`versioning.content_hash` = order-independent hash → reproducibility. `manifest.py`
links back to the Session and appends one `ExportRecord` per format.

**Topology provenance (Phase 3 extra, done).** Per-device `{device_id, topology_ref,
topology_hash}` is read from the Session's `DeviceDescriptor.topology_ref/hash`
(`pipeline._topology_provenance`) into `canonical.extra["topology"]` and stamped into
the Silver KV (`b"topology"`), `manifest.json`, `provenance.sidecar.json`, and the
datacard's `## Topology` section. Closes the loop dataset → `descriptor_hash` → exact
hardware revision. Empty list when no descriptor declares it (synthetic fixtures).

## Inspection (`inspect/`)

`summarize_dataset` / `summarize_canonical` (coverage, confidence, value ranges, QA,
content hash); `diff_datasets` (content-hash identity, per-stream coverage/confidence
deltas).

## CLI (`cli.py`, typer)

`version`, `formats`, `validate --dataset`, `inspect --dataset`, `diff --a --b`.
End-to-end `materialize` is **Python-only** (needs in-memory SyncResult); manifest-only
entry point deferred until timelines persist on disk.

## Hooks (`hooks/`) — Phase-4 seams, local no-ops

`authorize` (always allow), `watermark` (records intent, no embedding). Real
entitlement/watermark await the catalog.

## Critical Rules

1. **Never import SentrixSim.** Reach raw bytes only via `payload_ref` resolution.
2. **Never mutate `SyncResult`.** Only write-back = appending `ExportRecord` to a Session.
3. **All exporters project from Silver**, never from the timeline or raw payloads.
4. **Never re-derive synchronization.** Apply Sync's join indices; do not fit clocks,
   build grids, or re-run the as-of join.
5. **Gaps remain gaps.** `valid=False` → NaN value + 0 confidence; never interpolate.
   `no_fabricated_gaps` check → BLOCKED on violation.
6. **Confidence preserved.** Carry source/clock/interp; scalar is export-only.
7. **One canonical representation** — bump `SCHEMA_VERSION` if the Silver contract changes.

## Current Status

- **V1 complete:** parquet/file resolution, Silver, LeRobot v3 + MCAP + Parquet,
  validation + gate, packaging (Merkle/Ed25519, content hash, datacard, ExportRecord).
- **V2 complete:** HDF5 + RLDS exporters, mcap resolver, LeRobot v2/v3 flag +
  info.json check, inspect/diff CLI, format_options, dataset profiles.
- **41 tests passing** (`pytest tests -q`).
- **V3 blocked by upstream requirements:** video/AV1 (needs vision stream),
  per-licensee watermark (needs catalog), redaction seam, Rerun `.rrd` mirror,
  streaming materialization. Sub-frame bucketing is built+tested but inactive in the
  default single-glove path (no anchor frame rate yet).
