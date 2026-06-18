"""Timeline validation: monotonic grid, bounded step, no fabricated gaps.

Re-asserts SentrixSync's own property checks at the materialized layer: every
``valid == False`` frame must carry NaN (a real flagged gap), never an
interpolated value.
"""
from __future__ import annotations

import numpy as np

from ..materialize.canonical import CanonicalTable


def check_timeline(table: CanonicalTable) -> dict[str, str]:
    g = table.grid_us
    monotonic = bool(g.size <= 1 or np.all(np.diff(g) > 0))
    if g.size > 1:
        dt = np.diff(g)
        bounded = bool(dt.min() >= 1 and dt.max() <= 2 * dt.min())
    else:
        bounded = True

    no_fabricated = True
    for s in table.streams.values():
        invalid = ~s.valid
        if invalid.any():
            vals = s.values[invalid]
            # gaps must be NaN-filled (not fabricated values)
            if not np.all(np.isnan(vals)):
                no_fabricated = False
    return {
        "grid_monotonic": "pass" if monotonic else "fail",
        "bounded_step": "pass" if bounded else "fail",
        "no_fabricated_gaps": "pass" if no_fabricated else "fail",
    }
