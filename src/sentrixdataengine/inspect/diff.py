"""Diff two packaged dataset versions."""
from __future__ import annotations

from pathlib import Path

from .summary import summarize_dataset


def diff_datasets(a_dir: Path, b_dir: Path) -> dict:
    a = summarize_dataset(Path(a_dir))
    b = summarize_dataset(Path(b_dir))

    identical = a.get("content_hash") and a.get("content_hash") == b.get("content_hash")
    stream_deltas = {}
    for name in sorted(set(a["streams"]) | set(b["streams"])):
        sa = a["streams"].get(name)
        sb = b["streams"].get(name)
        if sa is None:
            stream_deltas[name] = {"status": "added"}
        elif sb is None:
            stream_deltas[name] = {"status": "removed"}
        else:
            stream_deltas[name] = {
                "status": "changed" if sa != sb else "same",
                "coverage_delta": round(sb["coverage"] - sa["coverage"], 6),
                "confidence_delta": round(sb["confidence_mean"] - sa["confidence_mean"], 6),
            }
    return {
        "content_hash_identical": bool(identical),
        "n_grid_delta": b["n_grid"] - a["n_grid"],
        "qa_verdict": {"a": a.get("qa_verdict"), "b": b.get("qa_verdict")},
        "coverage_min": {"a": a["coverage_min"], "b": b["coverage_min"]},
        "streams": stream_deltas,
    }
