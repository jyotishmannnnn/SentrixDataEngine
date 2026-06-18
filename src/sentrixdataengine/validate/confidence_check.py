"""Confidence validation: all components within [0,1]; zero at gaps."""
from __future__ import annotations

import numpy as np

from ..materialize.canonical import CanonicalTable


def _in_unit(a: np.ndarray) -> bool:
    return bool(a.size == 0 or (np.nanmin(a) >= 0.0 and np.nanmax(a) <= 1.0))


def check_confidence(table: CanonicalTable) -> dict[str, str]:
    in_range = True
    zero_at_gap = True
    for s in table.streams.values():
        for comp in (s.confidence, s.conf_source, s.conf_clock, s.conf_interp):
            if not _in_unit(comp):
                in_range = False
        invalid = ~s.valid
        if invalid.any() and np.any(s.confidence[invalid] != 0.0):
            zero_at_gap = False
    return {
        "confidence_in_unit_interval": "pass" if in_range else "fail",
        "confidence_zero_at_gaps": "pass" if zero_at_gap else "fail",
    }
