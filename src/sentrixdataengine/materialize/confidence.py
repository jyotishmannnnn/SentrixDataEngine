"""Fold SentrixSync's three-component confidence into per-frame arrays.

SentrixSync keeps source/clock/interpolation separate and authoritative; the
single scalar is EXPORT-ONLY (``ConfidenceComponents.derived_scalar``). We carry
all three plus the scalar so nothing is silently collapsed before Gold.
"""
from __future__ import annotations

import numpy as np


def fold(components, n_grid: int) -> dict[str, np.ndarray]:
    """Return {source, clock, interp, scalar} arrays of length n_grid.

    `components` is a sentrixsync ConfidenceComponents (or None → all-ones valid
    handled by the caller via the alignment mask)."""
    if components is None:
        ones = np.ones(n_grid, dtype=float)
        return {"source": ones, "clock": ones, "interp": ones, "scalar": ones}
    src = np.asarray(components.source, dtype=float)
    clk = np.asarray(components.clock, dtype=float)
    interp = np.asarray(components.interpolation, dtype=float)
    scalar = np.clip(src * clk * interp, 0.0, 1.0)
    return {"source": src, "clock": clk, "interp": interp, "scalar": scalar}
