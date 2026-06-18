"""Resolve SentrixSim MCAP episodes into per-stream payload arrays.

SentrixSim's MCAP writer (l7_export/mcap.py) logs 3 JSON channels:
  * topic ``tactile_field``  : {t_us, B_uT: [[bx,by,bz] x21], saturated}
  * topic ``dynamics_accel`` : {t_us, accel_g: [[ax,ay,az] x3]}
  * topic ``dynamics_temp``  : {t_us, temp_degC: [t0,t1,t2]}

A ``payload_kind`` maps to (topic, json-field). Messages are read in log order
and stacked into ``[N, *payload_shape]``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

# payload_kind -> (topic, json field)
_KIND_MAP = {
    "bmm350_cluster_uT": ("tactile_field", "B_uT"),
    "lis2dtw12_accel_g": ("dynamics_accel", "accel_g"),
    "lis2dtw12_temp_degC": ("dynamics_temp", "temp_degC"),
}


def _uri_to_path(rest: str) -> Path:
    if rest.startswith("/") and len(rest) > 2 and rest[2] == ":":
        rest = rest[1:]
    return Path(rest)


@lru_cache(maxsize=16)
def _read_topic(path_str: str, topic: str, field: str) -> tuple:
    from mcap.reader import make_reader
    rows: list = []
    with open(path_str, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic != topic:
                continue
            payload = json.loads(message.data)
            rows.append(payload[field])
    return tuple(map(tuple, ((tuple(_flatten_floats(r)) for r in rows)))) if rows else ()


def _flatten_floats(x):
    if isinstance(x, (list, tuple)):
        for v in x:
            yield from _flatten_floats(v)
    else:
        yield float(x)


class McapPayloadResolver:
    """Reads ``mcap://`` stream payloads from a SentrixSim episode log."""

    SCHEMES = ("mcap",)

    def supports(self, scheme: str) -> bool:
        return scheme in self.SCHEMES

    def resolve_stream(self, base_uri: str, payload_kind: str,
                       payload_shape: tuple[int, ...] | None) -> np.ndarray:
        if payload_kind not in _KIND_MAP:
            raise ValueError(
                f"McapPayloadResolver does not know payload_kind {payload_kind!r}; "
                "register it in mcap_resolver._KIND_MAP")
        scheme, rest = base_uri.split("://", 1)
        path = _uri_to_path(rest)
        if not path.exists():
            raise FileNotFoundError(f"payload mcap not found: {path}")
        topic, field = _KIND_MAP[payload_kind]
        flat_rows = _read_topic(str(path), topic, field)
        if not flat_rows:
            shape = payload_shape or (0,)
            return np.empty((0, *shape), dtype=np.float32)
        arr = np.asarray(flat_rows, dtype=np.float32)        # [N, prod(shape)]
        if payload_shape:
            return arr.reshape((arr.shape[0], *payload_shape))
        return arr
