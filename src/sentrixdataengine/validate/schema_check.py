"""Canonical schema validation: shapes/lengths internally consistent."""
from __future__ import annotations

from ..materialize.canonical import CanonicalTable


def check_schema(table: CanonicalTable) -> dict[str, str]:
    checks: dict[str, str] = {}
    n = table.n_grid
    checks["grid_frame_index_aligned"] = (
        "pass" if table.frame_index.shape[0] == n else "fail")
    ok_streams = True
    for s in table.streams.values():
        if s.values.shape[0] != n or s.valid.shape[0] != n or s.confidence.shape[0] != n:
            ok_streams = False
        if s.values.shape[1:] != s.shape:
            ok_streams = False
    checks["stream_shapes_consistent"] = "pass" if ok_streams else "fail"
    checks["schema_version_present"] = "pass" if table.schema_version else "fail"
    return checks
