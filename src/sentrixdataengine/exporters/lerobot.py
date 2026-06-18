"""LeRobot v3 exporter (multi-episode shard layout, one session = one episode).

Follows the SentrixSim LeRobot conventions (meta/info.json + meta/episodes.jsonl
+ data/chunk-NNN/file-NNN.parquet) but operates over the SYNCHRONIZED canonical
session table rather than a single simulator episode. Each stream becomes an
``observation.<stream_id>`` feature carrying its full per-frame payload shape;
the sub-frame burst, when present, is exported as a ``[R,U,V]`` tensor feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable
from .base import Exporter, register_exporter

_DATA_PATH = {
    "v3": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "v2": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
}


def _validate_against_lerobot(info: dict) -> str:
    """Best-effort: if lerobot is installed, check info.json carries the keys its
    version expects. Returns a note string; never raises on absence."""
    try:
        import lerobot  # type: ignore
    except Exception:
        return "lerobot not installed; info.json schema not cross-checked"
    required = {"codebase_version", "robot_type", "fps", "features", "data_path"}
    missing = required - set(info)
    ver = getattr(lerobot, "__version__", "unknown")
    if missing:
        return f"lerobot {ver}: info.json missing {sorted(missing)}"
    return f"lerobot {ver}: info.json keys ok"


def _features(canonical: CanonicalTable) -> dict:
    feats: dict = {}
    for s in canonical.streams.values():
        feats[f"observation.{s.stream_id}"] = {
            "dtype": "float32", "shape": list(s.shape) or [1], "units": s.units,
        }
        feats[f"observation.{s.stream_id}.confidence"] = {
            "dtype": "float32", "shape": [1], "units": "none"}
    feats.update({
        "timestamp": {"dtype": "float32", "shape": [1], "units": "s"},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    })
    return feats


@register_exporter
class LeRobotExporter(Exporter):
    name = "lerobot"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        options = options or {}
        layout = options.get("layout", "v3")
        if layout not in _DATA_PATH:
            raise ValueError(f"lerobot layout must be 'v2' or 'v3' (got {layout!r})")
        out_dir = Path(out_dir)
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        n = canonical.n_grid

        cols: dict[str, pa.Array] = {}
        for s in canonical.streams.values():
            flat = s.flat_values()  # [n, width]
            cols[f"observation.{s.stream_id}"] = pa.array(list(flat))  # list<float32> per row
            cols[f"observation.{s.stream_id}.confidence"] = pa.array(
                s.confidence.astype(np.float32))
        cols["timestamp"] = pa.array((canonical.grid_us / 1e6).astype(np.float32))
        cols["frame_index"] = pa.array(canonical.frame_index.astype(np.int64))
        cols["episode_index"] = pa.array(np.zeros(n, dtype=np.int64))
        cols["index"] = pa.array(np.arange(n, dtype=np.int64))
        cols["task_index"] = pa.array(np.zeros(n, dtype=np.int64))
        data_rel = _DATA_PATH[layout].format(chunk_index=0, file_index=0, episode_index=0)
        data_path = out_dir / data_rel
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), data_path, compression="zstd")

        info = {
            "codebase_version": f"{layout}.0-sentrixdataengine",
            "robot_type": "sentrix_visuotactile_session",
            "total_episodes": 1,
            "total_frames": n,
            "total_chunks": 1,
            "chunks_size": 1,
            "fps": float(canonical.grid_rate_hz),
            "data_path": _DATA_PATH[layout],
            "features": _features(canonical),
            "layout": layout,
            "tasks": [spec.session_id],
            "sentrix_meta": {
                "dataset_id": spec.dataset_id, "version": spec.version,
                "session_id": spec.session_id,
                "reference_clock_id": spec.reference_clock_id,
                "schema_version": spec.schema_version},
        }
        info["lerobot_validation"] = _validate_against_lerobot(info)
        (out_dir / "meta" / "info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")
        episode_rec = {"episode_index": 0, "tasks": [spec.session_id], "length": n}
        (out_dir / "meta" / "episodes.jsonl").write_text(
            json.dumps(episode_rec) + "\n", encoding="utf-8")

        sample_count = sum(int(s.valid.sum()) for s in canonical.streams.values())
        return ExportResult(
            format=self.name, out_dir=out_dir,
            files=[data_path, out_dir / "meta" / "info.json",
                   out_dir / "meta" / "episodes.jsonl"],
            frame_count=n, sample_count=sample_count)
