"""Exporter contract + registry."""
from __future__ import annotations

import abc
from pathlib import Path

from ..contracts import DatasetSpec, ExportResult
from ..materialize.canonical import CanonicalTable

_REGISTRY: dict[str, type["Exporter"]] = {}


def register_exporter(cls: type["Exporter"]) -> type["Exporter"]:
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError("Exporter subclass must set a non-empty `name`")
    _REGISTRY[name] = cls
    return cls


def get_exporter(name: str) -> "Exporter":
    if name not in _REGISTRY:
        raise KeyError(f"unknown exporter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def registered_exporters() -> list[str]:
    return sorted(_REGISTRY)


class Exporter(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def export(self, canonical: CanonicalTable, spec: DatasetSpec,
               out_dir: Path, options: dict | None = None) -> ExportResult:
        """Project the canonical table into this format under ``out_dir``.

        `options` carries per-format knobs (e.g. lerobot ``layout``, hdf5
        ``embed`` / ``compression``). Implementations must tolerate ``None``."""
