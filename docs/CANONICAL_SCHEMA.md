# Canonical (Silver) Schema — v1.0

The single internal source of truth. Every Gold exporter is a pure projection of
this representation. Frozen contract; bump `schema_version` on change.

## In-memory (`materialize.canonical`)

`CanonicalTable`
- `grid_us: int64[n_grid]` — reference-time grid (from `SyncResult.timeline.grid_us`)
- `frame_index: int64[n_grid]`
- `streams: dict[key -> CanonicalStream]` where `key = "device::stream"`
- `reference_clock_id, grid_rate_hz, session_id, schema_version`
- `extra: dict` — `source_episode_hashes`, passthrough labels

`CanonicalStream`
- `key, device_id, stream_id, payload_kind, units, kernel`
- `shape: tuple` — per-frame payload shape, e.g. `(21, 3)`
- `values: float32[n_grid, *shape]` — **NaN at gaps** (never fabricated)
- `valid: bool[n_grid]`
- `confidence: float[n_grid]` — export scalar = `source * clock * interp`
- `conf_source, conf_clock, conf_interp: float[n_grid]` — the three components, retained

## On-disk (Silver Parquet, `aligned/part-000.parquet`)

Columns:
- `t_ref_us: int64`
- `frame_index: int64`
- per stream `<stream_id>.cNNN: float32` (flattened payload, `NNN` = 0..width-1)
- per stream `<stream_id>.valid: bool`
- per stream `<stream_id>.confidence: float32`

File-level KV metadata:
- `sentrixdataengine_schema_version`, `session_id`, `reference_clock_id`,
  `grid_rate_hz`, `streams` (JSON shape/units/kernel/coverage map),
  `source_episode_hashes` (JSON list)

## Invariants

1. `valid == False` ⇒ value is NaN and confidence is 0. No interpolation across a gap.
2. The three confidence components are authoritative; the scalar is export-only.
3. Join indices come from SentrixSync; this layer applies them, never re-derives them.
