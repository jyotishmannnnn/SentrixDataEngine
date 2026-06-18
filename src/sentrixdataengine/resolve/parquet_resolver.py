"""Resolve SentrixSim Parquet episodes into per-stream payload arrays.

SentrixSim writes one flat table per episode (l7_export/parquet.py):
  * tactile field : columns ``tactile.bNN.{bx,by,bz}_uT``  (NN = 00..n_bmm-1)
  * dynamics accel: columns ``dyn.{thumb,index,middle}.{ax,ay,az}_g``

SentrixSync's SentrixSimAdapter addresses these as
``parquet://<abs-path>#stream=<id>&row=<i>`` and declares the stream's
``payload_kind`` + ``payload_shape`` in the DeviceDescriptor. This resolver maps
a ``payload_kind`` to its column layout and returns the whole stream as one
array (cached per file) — the caller indexes rows via the join indices.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_TACTILE_AXES = ("bx", "by", "bz")
_ACCEL_AXES = ("ax", "ay", "az")
_TRIPOD = ("thumb", "index", "middle")


def _uri_to_path(rest: str) -> Path:
    """``parquet://D:/a/b.parquet`` rest == ``D:/a/b.parquet`` → Path."""
    # tolerate a leading slash from file:///abs forms
    if rest.startswith("/") and len(rest) > 2 and rest[2] == ":":
        rest = rest[1:]  # /D:/x -> D:/x
    return Path(rest)


def _tactile_columns(n_clusters: int) -> list[str]:
    return [f"tactile.b{i:02d}.{ax}_uT"
            for i in range(n_clusters) for ax in _TACTILE_AXES]


def _accel_columns() -> list[str]:
    return [f"dyn.{finger}.{ax}_g" for finger in _TRIPOD for ax in _ACCEL_AXES]


def _temp_columns() -> list[str]:
    return [f"dyn.{finger}.temp_degC" for finger in _TRIPOD]


# payload_kind -> (column-builder, reshape) ; extensible for new sensors.
def _columns_for(payload_kind: str, payload_shape: tuple[int, ...] | None) -> list[str]:
    if payload_kind == "bmm350_cluster_uT":
        n = payload_shape[0] if payload_shape else 21
        return _tactile_columns(n)
    if payload_kind == "lis2dtw12_accel_g":
        return _accel_columns()
    if payload_kind == "lis2dtw12_temp_degC":
        return _temp_columns()
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
        columns = tuple(_columns_for(payload_kind, payload_shape))
        flat = _read_columns(str(path), columns)             # [N, prod(shape)]
        if payload_shape:
            n = flat.shape[0]
            return flat.reshape((n, *payload_shape))
        return flat
