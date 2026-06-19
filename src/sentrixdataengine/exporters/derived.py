"""Derived-feature Gold exporter (opt-in) — the ONLY component that consumes the
topology descriptor's SPATIAL parts (sensor positions, clusters).

Materializes topology-DEPENDENT proxies from the raw magnetic field per cluster:
per-cluster normal proxy, shear vector + magnitude, and contact centroid. These
are deterministic functions of raw B + the descriptor; they are recorded with
their formula + code version + descriptor hash so they are reproducible and never
mistaken for measured truth. Canonical Silver is untouched (raw only) — this is a
pure projection, requested explicitly via formats=("derived",).

Design rules honoured:
  * raw-only Silver: features are recomputed here, never persisted as canonical.
  * topology from the descriptor: clusters/positions come from sentrix_contracts,
    NOT hardcoded. Any sensor count/layout works.
  * proxies, not physics: ΔB uses a documented per-sensor baseline; magnitudes are
    relative (µT), matching SentrixSim's relative-physics stance.

Assumption (asserted): the tactile stream's payload index order equals the
descriptor's magnetic-sensor order — both originate from the same producer under
the same topology (true for SentrixSim / SentrixCapture output).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable
from .base import Exporter, register_exporter

DERIVED_VERSION = "1.0"
_MAG_KIND = "bmm350_cluster_uT"

# formula registry recorded into the output (reproducibility / provenance)
_FORMULAS = {
    "baseline": "B0[k] = nanmedian_t(B[:,k,:])  (per-sensor, over valid frames)",
    "dB": "dB[t,k] = B[t,k] - B0[k]",
    "normal_proxy": "mean_k |dB[t,k]|            (common-mode magnitude, uT)",
    "shear_x/shear_y": "mean_k dB[t,k,:2]         (lateral field shift, uT)",
    "shear_mag": "|(shear_x, shear_y)|",
    "centroid_x_m/centroid_y_m":
        "sum_k |dB[t,k]|*pos_xy[k] / sum_k |dB[t,k]|   (response-weighted, metres)",
}


def _load_descriptor(topology_ref: str):
    try:
        from sentrix_contracts import bundled_descriptor_path, load_descriptor
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the 'derived' exporter needs the sentrix_contracts package to read "
            "the topology descriptor's spatial layout") from e
    return load_descriptor(bundled_descriptor_path(topology_ref))


def _topology_ref_for(device_id: str, topology: list[dict]) -> str | None:
    for t in topology or []:
        if t.get("device_id") == device_id:
            return t.get("topology_ref")
    return None


@register_exporter
class DerivedExporter(Exporter):
    """Topology-dependent derived features. Requires a topology descriptor."""
    name = "derived"

    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        topology = canonical.extra.get("topology") or []

        mag_streams = [s for s in canonical.streams.values()
                       if s.payload_kind == _MAG_KIND]
        if not mag_streams:
            raise ValueError(
                "derived exporter found no magnetic stream "
                f"(payload_kind={_MAG_KIND!r}); nothing to derive")

        files: list[Path] = []
        sample_count = 0
        for s in mag_streams:
            ref = _topology_ref_for(s.device_id, topology)
            if ref is None:
                raise ValueError(
                    f"derived exporter needs a topology descriptor for device "
                    f"{s.device_id!r}; none in session provenance (run Phase 2 so "
                    "SentrixSync carries topology_ref/topology_hash)")
            desc = _load_descriptor(ref)
            mags = [sen for sen in desc.sensors.values() if sen.modality == "magnetic"]
            if len(mags) != s.shape[0]:
                raise ValueError(
                    f"descriptor {ref!r} has {len(mags)} magnetic sensors but the "
                    f"stream payload has {s.shape[0]}; producer/topology mismatch")

            idx_of = {sen.sensor_id: k for k, sen in enumerate(mags)}
            pos_xy = np.array([[float(sen.position_m[0]), float(sen.position_m[1])]
                               for sen in mags], dtype=np.float64)

            # cluster_id -> magnetic-sensor indices (spatial grouping from descriptor)
            clusters: dict[str, list[int]] = {}
            for cid, cl in desc.clusters.items():
                members = [idx_of[m] for m in cl.members if m in idx_of]
                if members:
                    clusters[cid] = members
            if not clusters:  # descriptor without explicit clusters -> per-sensor
                clusters = {sen.sensor_id: [k] for k, sen in enumerate(mags)}

            cols, feat_hash = self._features(s, pos_xy, clusters, canonical)
            data = {"t_ref_us": pa.array(canonical.grid_us.astype(np.int64)),
                    "frame_index": pa.array(canonical.frame_index.astype(np.int64))}
            for name, arr in cols.items():
                data[name] = pa.array(arr.astype(np.float32))

            topo_hash = next((t.get("topology_hash") for t in topology
                              if t.get("device_id") == s.device_id), None)
            meta = {
                b"sentrixdataengine_derived_version": DERIVED_VERSION.encode(),
                b"descriptor_version": ref.encode(),
                b"descriptor_hash": (topo_hash or "").encode(),
                b"device_id": s.device_id.encode(),
                b"baseline_method": b"per_sensor_median_over_valid",
                b"formulas": json.dumps(_FORMULAS).encode(),
                b"units": json.dumps({"normal_proxy": "uT", "shear": "uT",
                                      "centroid": "m"}).encode(),
                b"clusters": json.dumps({c: [mags[i].sensor_id for i in idx]
                                         for c, idx in clusters.items()}).encode(),
            }
            table = pa.table(data).replace_schema_metadata(meta)
            fname = (f"derived-{s.device_id}.parquet"
                     if len(mag_streams) > 1 else "part-000.parquet")
            path = out_dir / fname
            pq.write_table(table, path, compression="zstd")
            files.append(path)
            sample_count += int(s.valid.sum())

        return ExportResult(format=self.name, out_dir=out_dir, files=files,
                            frame_count=canonical.n_grid, sample_count=sample_count)

    @staticmethod
    def _features(stream, pos_xy: np.ndarray, clusters: dict[str, list[int]],
                  canonical: CanonicalTable):
        B = stream.values.astype(np.float64)              # [T, M, 3], NaN at gaps
        B0 = np.nanmedian(B, axis=0)                       # [M, 3]
        dB = B - B0[None, :, :]                            # [T, M, 3]
        r = np.linalg.norm(dB, axis=2)                     # [T, M]  response magnitude

        cols: dict[str, np.ndarray] = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for cid, idxs in clusters.items():
                rc = r[:, idxs]                            # [T, m]
                normal = np.nanmean(rc, axis=1)
                shear = np.nanmean(dB[:, idxs, :2], axis=1)        # [T, 2]
                shear_mag = np.linalg.norm(shear, axis=1)
                pos = pos_xy[idxs]                                  # [m, 2]
                wsum = np.nansum(rc, axis=1)                        # [T]
                num = np.nansum(rc[:, :, None] * pos[None, :, :], axis=1)  # [T, 2]
                centroid = np.where(wsum[:, None] > 0, num / wsum[:, None], np.nan)
                pre = f"derived.{cid}"
                cols[f"{pre}.normal_proxy"] = normal
                cols[f"{pre}.shear_x"] = shear[:, 0]
                cols[f"{pre}.shear_y"] = shear[:, 1]
                cols[f"{pre}.shear_mag"] = shear_mag
                cols[f"{pre}.centroid_x_m"] = centroid[:, 0]
                cols[f"{pre}.centroid_y_m"] = centroid[:, 1]
        return cols, None
