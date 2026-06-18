"""Inspection — dataset summary stats and version diff."""
from __future__ import annotations

from .diff import diff_datasets
from .summary import summarize_canonical, summarize_dataset

__all__ = ["summarize_dataset", "summarize_canonical", "diff_datasets"]
