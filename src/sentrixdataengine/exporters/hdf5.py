"""HDF5 exporter (robomimic-style), manual Phase 5.3.

Streams each canonical stream into resizable, chunked, compressed datasets so a
trainer can do partial reads and writer memory stays bounded regardless of
episode length. One session → one ``demo_0`` group under ``data/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable
from .base import Exporter, register_exporter

_BATCH = 4096   # row-group-sized append chunks (bounded writer memory)


@register_exporter
class Hdf5Exporter(Exporter):
    name = "hdf5"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        try:
            import h5py
        except ImportError as e:  # pragma: no cover
            raise ImportError("hdf5 export requires the optional 'h5py' dependency "
                              "(pip install sentrixdataengine[hdf5])") from e
        options = options or {}
        compression = options.get("compression", "gzip")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "dataset.hdf5"
        n = canonical.n_grid

        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs["env"] = "sentrix_visuotactile_session"
            data.attrs["session_id"] = spec.session_id
            g = data.create_group("demo_0")
            obs = g.create_group("obs")
            # reference time + per-frame indices
            self._stream_to_resizable(g, "timestamp",
                                      (canonical.grid_us / 1e6).astype(np.float32),
                                      compression)
            sample_count = 0
            names = canonical.feature_names()
            for key, s in canonical.streams.items():
                name = names[key]
                flat = s.flat_values().astype(np.float32)        # [n, width]
                self._stream_to_resizable(obs, name, flat, compression)
                self._stream_to_resizable(obs, f"{name}.confidence",
                                          s.confidence.astype(np.float32), compression)
                self._stream_to_resizable(obs, f"{name}.valid",
                                          s.valid.astype(np.uint8), compression)
                sample_count += int(s.valid.sum())
            g.attrs["num_samples"] = n
            data.attrs["total"] = n

        return ExportResult(format=self.name, out_dir=out_dir, files=[path],
                            frame_count=n, sample_count=sample_count)

    @staticmethod
    def _stream_to_resizable(group, name: str, arr: np.ndarray, compression: str) -> None:
        """Create a resizable dataset and fill it in bounded batches (Phase 5.3)."""
        n = arr.shape[0]
        feat = arr.shape[1:]
        chunk0 = min(_BATCH, n) if n else 1
        d = group.create_dataset(
            name, shape=(0, *feat), maxshape=(None, *feat),
            chunks=(chunk0, *feat) if feat else (chunk0,),
            compression=compression, dtype=arr.dtype)
        for start in range(0, n, _BATCH):
            end = min(start + _BATCH, n)
            d.resize(end, axis=0)
            d[start:end] = arr[start:end]
