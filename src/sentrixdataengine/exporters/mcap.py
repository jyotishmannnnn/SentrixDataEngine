"""MCAP exporter — full-fidelity, replayable aligned session log.

One channel per stream (JSON-schema encoded), logged at the reference grid times
for valid frames only. Requires the optional ``mcap`` dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable
from .base import Exporter, register_exporter


@register_exporter
class McapExporter(Exporter):
    name = "mcap"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        try:
            from mcap.writer import Writer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "mcap export requires the optional 'mcap' dependency "
                "(pip install sentrixdataengine[mcap])") from e

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "session-000.mcap"

        with path.open("wb") as f:
            writer = Writer(f)
            writer.start()
            channels: dict[str, int] = {}
            for s in canonical.streams.values():
                schema_id = writer.register_schema(
                    name=f"sentrix.{s.stream_id}", encoding="jsonschema",
                    data=json.dumps({
                        "type": "object",
                        "properties": {
                            "t_ref_us": {"type": "integer"},
                            "value": {"type": "array"},
                            "confidence": {"type": "number"},
                        }}).encode())
                channels[s.key] = writer.register_channel(
                    topic=s.stream_id, message_encoding="json", schema_id=schema_id)

            grid = canonical.grid_us
            sample_count = 0
            for s in canonical.streams.values():
                flat = s.flat_values()
                ch = channels[s.key]
                for i in range(canonical.n_grid):
                    if not s.valid[i]:
                        continue
                    msg = {"t_ref_us": int(grid[i]),
                           "value": [float(x) for x in flat[i]],
                           "confidence": float(s.confidence[i])}
                    writer.add_message(
                        channel_id=ch, log_time=int(grid[i]) * 1000,
                        publish_time=int(grid[i]) * 1000,
                        data=json.dumps(msg).encode())
                    sample_count += 1
            writer.finish()

        return ExportResult(format=self.name, out_dir=out_dir, files=[path],
                            frame_count=canonical.n_grid, sample_count=sample_count)
