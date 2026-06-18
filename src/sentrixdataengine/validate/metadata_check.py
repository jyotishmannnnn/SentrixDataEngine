"""Metadata validation: required identity keys present and coherent."""
from __future__ import annotations

from ..materialize.canonical import CanonicalTable


def check_metadata(table: CanonicalTable) -> dict[str, str]:
    return {
        "session_id_present": "pass" if table.session_id else "fail",
        "reference_clock_present": "pass" if table.reference_clock_id else "fail",
        "grid_rate_positive": "pass" if table.grid_rate_hz > 0 else "fail",
        "has_streams": "pass" if table.streams else "fail",
    }
