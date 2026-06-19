"""Resolve SentrixSim / SentrixCapture Parquet episodes into per-stream arrays.

The producer writes one flat table per episode. As of Migration Phase 1 the
canonical column scheme is **topology-driven, sensor_id-keyed**:
  * tactile field : ``mag.<sensor_id>.{bx,by,bz}_uT``
  * dynamics accel: ``dyn.<sensor_id>.{ax,ay,az}_g``
  * dynamics temp : ``dyn.<sensor_id>.temp_c``
with the per-modality sensor order carried in the parquet KV metadata
(``sentrixsim_meta`` / ``sentrix_capture_meta`` → ``bmm_sensor_ids`` /
``lis_sensor_ids``). This resolver is therefore self-describing: it reads the
sensor order from the file and needs no external descriptor.

Legacy Layout-B datasets (``tactile.bNN.*`` / ``dyn.<finger>.*``) are still read
via a fallback, so pre-migration artifacts keep resolving.

A ``payload_kind`` selects the modality; the column SET comes from the file.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_TACTILE_AXES = ("bx", "by", "bz")
_ACCEL_AXES = ("ax", "ay", "az")
_TRIPOD = ("thumb", "index", "middle")
_META_KEYS = (b"sentrixsim_meta", b"sentrix_capture_meta")


def _uri_to_path(rest: str) -> Path:
    """``parquet://D:/a/b.parquet`` rest == ``D:/a/b.parquet`` → Path."""
    if rest.startswith("/") and len(rest) > 2 and rest[2] == ":":
        rest = rest[1:]  # /D:/x -> D:/x
    return Path(rest)


@lru_cache(maxsize=32)
def _schema_info(path_str: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (column_names, bmm_ids, lis_ids) read cheaply from the file schema +
    KV metadata. sensor-id tuples are empty when the file predates Phase 1."""
    import pyarrow.parquet as pq
    schema = pq.read_schema(path_str)
    names = tuple(schema.names)
    bmm_ids: tuple[str, ...] = ()
    lis_ids: tuple[str, ...] = ()
    meta = schema.metadata or {}
    for key in _META_KEYS:
        if key in meta:
            try:
                blob = json.loads(meta[key].decode())
            except (ValueError, UnicodeDecodeError):
                continue
            bmm_ids = tuple(blob.get("bmm_sensor_ids") or ())
            lis_ids = tuple(blob.get("lis_sensor_ids") or ())
            if bmm_ids or lis_ids:
                break
    return names, bmm_ids, lis_ids


# ---- canonical sensor_id-keyed column builders ----
def _mag_columns(bmm_ids: tuple[str, ...]) -> list[str]:
    return [f"mag.{sid}.{ax}_uT" for sid in bmm_ids for ax in _TACTILE_AXES]


def _dyn_accel_columns(lis_ids: tuple[str, ...]) -> list[str]:
    return [f"dyn.{sid}.{ax}_g" for sid in lis_ids for ax in _ACCEL_AXES]


def _dyn_temp_columns(lis_ids: tuple[str, ...]) -> list[str]:
    return [f"dyn.{sid}.temp_c" for sid in lis_ids]


# ---- legacy Layout-B builders ----
def _legacy_tactile_columns(n_clusters: int) -> list[str]:
    return [f"tactile.b{i:02d}.{ax}_uT"
            for i in range(n_clusters) for ax in _TACTILE_AXES]


def _legacy_accel_columns() -> list[str]:
    return [f"dyn.{finger}.{ax}_g" for finger in _TRIPOD for ax in _ACCEL_AXES]


def _legacy_temp_columns() -> list[str]:
    return [f"dyn.{finger}.temp_degC" for finger in _TRIPOD]


def _columns_for(payload_kind: str, payload_shape: tuple[int, ...] | None,
                 names: tuple[str, ...], bmm_ids: tuple[str, ...],
                 lis_ids: tuple[str, ...]) -> list[str]:
    """Pick the column set for a payload_kind, preferring the canonical
    sensor_id-keyed scheme when the file declares sensor ids and carries those
    columns; otherwise fall back to legacy Layout-B names."""
    name_set = set(names)

    def _use(cols: list[str]) -> bool:
        return bool(cols) and all(c in name_set for c in cols)

    if payload_kind == "bmm350_cluster_uT":
        canonical = _mag_columns(bmm_ids)
        if _use(canonical):
            return canonical
        n = payload_shape[0] if payload_shape else 21
        legacy = _legacy_tactile_columns(n)
        if _use(legacy):
            return legacy
        return canonical or legacy  # let the read raise a clear missing-column error
    if payload_kind == "lis2dtw12_accel_g":
        canonical = _dyn_accel_columns(lis_ids)
        if _use(canonical):
            return canonical
        return _legacy_accel_columns()
    if payload_kind == "lis2dtw12_temp_degC":
        canonical = _dyn_temp_columns(lis_ids)
        if _use(canonical):
            return canonical
        return _legacy_temp_columns()
    raise ValueError(
        f"ParquetPayloadResolver does not know payload_kind {payload_kind!r}; "
        "register a column mapping in parquet_resolver._columns_for")


@lru_cache(maxsize=32)
def _read_columns(path_str: str, columns: tuple[str, ...]) -> np.ndarray:
    import pyarrow.parquet as pq
    table = pq.read_table(path_str, columns=list(columns))
    cols = [np.asarray(table.column(c).to_numpy(), dtype=np.float32) for c in columns]
    return np.stack(cols, axis=1)  # [N, len(columns)]


class ParquetPayloadResolver:
    """Reads ``parquet://`` / ``file://...parquet`` stream payloads."""

    SCHEMES = ("parquet", "file")

    def supports(self, scheme: str) -> bool:
        return scheme in self.SCHEMES

    def resolve_stream(self, base_uri: str, payload_kind: str,
                       payload_shape: tuple[int, ...] | None) -> np.ndarray:
        scheme, rest = base_uri.split("://", 1)
        if scheme == "file" and not rest.lower().endswith(".parquet"):
            raise ValueError(f"file resolver expects a .parquet location: {base_uri!r}")
        path = _uri_to_path(rest)
        if not path.exists():
            raise FileNotFoundError(f"payload parquet not found: {path}")
        names, bmm_ids, lis_ids = _schema_info(str(path))
        columns = tuple(_columns_for(payload_kind, payload_shape, names, bmm_ids, lis_ids))
        flat = _read_columns(str(path), columns)             # [N, prod(shape)]
        if payload_shape:
            n = flat.shape[0]
            return flat.reshape((n, *payload_shape))
        return flat
