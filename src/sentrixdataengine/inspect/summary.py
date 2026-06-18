"""Dataset summary statistics.

Works on a packaged dataset directory (manifest + Silver Parquet) or directly on
an in-memory CanonicalTable. Reports frame counts, per-stream coverage,
confidence distribution, and value ranges — the inputs a buyer/QA reviewer wants
before trusting a package.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarize_canonical(table) -> dict:
    streams = {}
    names = table.feature_names()
    for key, s in table.streams.items():
        vals = s.values[s.valid] if s.valid.any() else np.empty((0,))
        streams[names[key]] = {
            "device_id": s.device_id, "payload_kind": s.payload_kind,
            "shape": list(s.shape), "units": s.units, "kernel": s.kernel,
            "coverage": round(s.coverage(), 6),
            "confidence_mean": round(float(np.mean(s.confidence[s.valid])), 6)
            if s.valid.any() else 0.0,
            "value_min": (float(np.nanmin(vals)) if vals.size else None),
            "value_max": (float(np.nanmax(vals)) if vals.size else None),
        }
    return {
        "session_id": table.session_id,
        "reference_clock_id": table.reference_clock_id,
        "grid_rate_hz": table.grid_rate_hz,
        "n_grid": table.n_grid,
        "coverage_min": round(table.coverage_min(), 6),
        "streams": streams,
    }


def summarize_dataset(dataset_dir: Path) -> dict:
    """Summarize a packaged dataset (a ``version=…`` directory)."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    silver = dataset_dir / "silver" / "aligned" / "part-000.parquet"
    if not silver.exists():
        raise FileNotFoundError(f"no Silver table under {dataset_dir}")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(silver)
    schema = pf.schema_arrow
    n_rows = pf.metadata.num_rows
    meta = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()}
    stream_meta = json.loads(meta.get("streams", "{}"))

    table = pf.read()
    streams = {}
    for name, sm in stream_meta.items():
        valid_col = f"{name}.valid"
        conf_col = f"{name}.confidence"
        valid = (np.asarray(table.column(valid_col).to_numpy())
                 if valid_col in table.column_names else np.ones(n_rows, bool))
        conf = (np.asarray(table.column(conf_col).to_numpy())
                if conf_col in table.column_names else np.ones(n_rows))
        streams[name] = {
            **{k: sm.get(k) for k in ("device_id", "payload_kind", "shape", "units", "kernel")},
            "coverage": round(float(valid.mean()), 6) if n_rows else 0.0,
            "confidence_mean": round(float(conf[valid].mean()), 6) if valid.any() else 0.0,
        }

    out = {
        "session_id": meta.get("session_id"),
        "reference_clock_id": meta.get("reference_clock_id"),
        "grid_rate_hz": float(meta.get("grid_rate_hz", 0.0) or 0.0),
        "n_grid": int(n_rows),
        "coverage_min": round(min((s["coverage"] for s in streams.values()), default=1.0), 6),
        "streams": streams,
    }
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        out["qa_verdict"] = m.get("qa_verdict")
        out["content_hash"] = m.get("content_hash")
        out["formats"] = [f["format"] for f in m.get("formats", [])]
    return out
