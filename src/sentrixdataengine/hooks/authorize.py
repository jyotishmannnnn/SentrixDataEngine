"""Authorization seam (manual Phase 5.0 `authorize`). Local: always allow."""
from __future__ import annotations


def authorize(*, customer_id: str | None, dataset_id: str, formats: tuple[str, ...]
              ) -> bool:
    """Return True if the (customer, dataset, formats) export is permitted.

    V1 local policy: always allow. A real entitlement provider replaces this to
    enforce training_rights / frame_cap / redistribution (manual Phase 4)."""
    return True
