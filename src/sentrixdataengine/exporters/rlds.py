"""RLDS exporter (manual Phase 5.2) — episode as a sequence of step dicts.

RLDS's canonical sink is TFDS/TFRecord, which drags in TensorFlow. To keep V2
dependency-light and testable, this writes the RLDS *step structure* in a
portable form: a ``steps.parquet`` (one row per step, RLDS step fields) plus an
``episode_metadata.json``. A thin TFDS ``GeneratorBasedBuilder`` that yields
these rows is a later, optional wrapper — the step semantics are already here.
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


@register_exporter
class RldsExporter(Exporter):
    name = "rlds"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = canonical.n_grid

        cols: dict[str, pa.Array] = {}
        names = canonical.feature_names()
        for key, s in canonical.streams.items():
            name = names[key]
            cols[f"observation.{name}"] = pa.array(list(s.flat_values()))
            cols[f"observation.{name}.confidence"] = pa.array(
                s.confidence.astype(np.float32))
        # RLDS step scaffolding. reward/discount are neutral until a reward signal
        # exists upstream (Phase 3); the boundary flags are real.
        is_first = np.zeros(n, dtype=bool); is_first[0] = True if n else False
        is_last = np.zeros(n, dtype=bool)
        is_terminal = np.zeros(n, dtype=bool)
        if n:
            is_last[-1] = True
            is_terminal[-1] = True
        cols["reward"] = pa.array(np.zeros(n, dtype=np.float32))
        cols["discount"] = pa.array(np.ones(n, dtype=np.float32))
        cols["is_first"] = pa.array(is_first)
        cols["is_last"] = pa.array(is_last)
        cols["is_terminal"] = pa.array(is_terminal)
        cols["frame_index"] = pa.array(canonical.frame_index.astype(np.int64))

        steps_path = out_dir / "steps.parquet"
        pq.write_table(pa.table(cols), steps_path, compression="zstd")

        episode_meta = {
            "episode_id": spec.session_id, "outcome": "unknown",
            "num_steps": n, "dataset_id": spec.dataset_id, "version": spec.version,
            "note": "RLDS step structure in portable parquet; TFDS build is a wrapper",
        }
        meta_path = out_dir / "episode_metadata.json"
        meta_path.write_text(json.dumps(episode_meta, indent=2), encoding="utf-8")

        sample_count = sum(int(s.valid.sum()) for s in canonical.streams.values())
        return ExportResult(format=self.name, out_dir=out_dir,
                            files=[steps_path, meta_path],
                            frame_count=n, sample_count=sample_count)
