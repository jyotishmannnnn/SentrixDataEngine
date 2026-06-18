"""Plain columnar passthrough exporter — the canonical table as-is."""
from __future__ import annotations

from pathlib import Path

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable
from ..materialize.silver_writer import build_arrow
from .base import Exporter, register_exporter


@register_exporter
class ParquetExporter(Exporter):
    name = "parquet"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq
        path = out_dir / "part-000.parquet"
        pq.write_table(build_arrow(canonical), path, compression="zstd")
        sample_count = sum(int(s.valid.sum()) for s in canonical.streams.values())
        return ExportResult(format=self.name, out_dir=out_dir, files=[path],
                            frame_count=canonical.n_grid, sample_count=sample_count)
