"""Gold directory layout (medallion, manual Phase 2.2).

<out_root>/dataset=<id>/version=<semver>/
    format=<fmt>/...
    silver/aligned/part-000.parquet
    manifest.json
    provenance.sidecar.json
    DATACARD.md
    qa_report.json
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldLayout:
    root: Path
    dataset_id: str
    version: str

    @property
    def base(self) -> Path:
        return self.root / f"dataset={self.dataset_id}" / f"version={self.version}"

    def format_dir(self, fmt: str) -> Path:
        return self.base / f"format={fmt}"

    @property
    def silver_dir(self) -> Path:
        return self.base / "silver"

    @property
    def manifest_path(self) -> Path:
        return self.base / "manifest.json"

    @property
    def provenance_path(self) -> Path:
        return self.base / "provenance.sidecar.json"

    @property
    def datacard_path(self) -> Path:
        return self.base / "DATACARD.md"

    @property
    def qa_path(self) -> Path:
        return self.base / "qa_report.json"

    def ensure(self) -> "GoldLayout":
        self.base.mkdir(parents=True, exist_ok=True)
        return self
