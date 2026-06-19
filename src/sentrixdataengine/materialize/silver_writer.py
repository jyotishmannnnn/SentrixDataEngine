"""Write the canonical table to Silver Parquet (streamed, flattened columns).

Layout: ``t_ref_us``, ``frame_index``, then per stream the flattened value
columns ``<stream_id>.cNNN`` plus ``<stream_id>.valid`` and
``<stream_id>.confidence``. File-level KV metadata records the schema contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .canonical import CanonicalTable


def _stream_columns(name: str, s) -> dict[str, pa.Array]:
    flat = s.flat_values()  # [n_grid, width]
    cols: dict[str, pa.Array] = {}
    for j in range(s.width):
        cols[f"{name}.c{j:03d}"] = pa.array(flat[:, j].astype(np.float32))
    cols[f"{name}.valid"] = pa.array(s.valid.astype(bool))
    cols[f"{name}.confidence"] = pa.array(s.confidence.astype(np.float32))
    return cols


def build_arrow(table: CanonicalTable) -> pa.Table:
    data: dict[str, pa.Array] = {
        "t_ref_us": pa.array(table.grid_us.astype(np.int64)),
        "frame_index": pa.array(table.frame_index.astype(np.int64)),
    }
    stream_meta = {}
    names = table.feature_names()
    for key, s in table.streams.items():
        name = names[key]
        data.update(_stream_columns(name, s))
        stream_meta[name] = {
            "key": key, "device_id": s.device_id, "payload_kind": s.payload_kind,
            "units": s.units, "kernel": s.kernel, "shape": list(s.shape),
            "width": s.width, "coverage": s.coverage(),
        }
    schema_meta = {
        b"sentrixdataengine_schema_version": table.schema_version.encode(),
        b"session_id": table.session_id.encode(),
        b"reference_clock_id": table.reference_clock_id.encode(),
        b"grid_rate_hz": str(table.grid_rate_hz).encode(),
        b"streams": json.dumps(stream_meta).encode(),
        b"source_episode_hashes": json.dumps(
            table.extra.get("source_episode_hashes", [])).encode(),
        b"topology": json.dumps(table.extra.get("topology", [])).encode(),
    }
    arrow = pa.table(data)
    return arrow.replace_schema_metadata(schema_meta)


def write_silver(table: CanonicalTable, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "aligned").mkdir(parents=True, exist_ok=True)
    path = out_dir / "aligned" / "part-000.parquet"
    pq.write_table(build_arrow(table), path, compression="zstd")
    return path
