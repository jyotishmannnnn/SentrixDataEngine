"""Dataset-level release gate (manual Phase 7.3).

Composes integrity + quality + property checks into a single verdict:
``CERTIFIED | RELEASE | NEEDS_REVIEW | BLOCKED``. Inherits SentrixSync's own
``ValidationReport.gate_verdict`` as a ceiling — the dataset gate can only be as
good as the synchronization it rests on, never better.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..contracts import QAReport
from ..materialize.canonical import CanonicalTable

_ORDER = {"BLOCKED": 0, "NEEDS_REVIEW": 1, "RELEASE": 2, "CERTIFIED": 3}
_CRITICAL_CHECKS = ("no_fabricated_gaps", "stream_shapes_consistent",
                    "confidence_in_unit_interval")

_DEFAULT_CFG = Path(__file__).resolve().parents[3] / "configs" / "qa_thresholds.yaml"


@dataclass
class GateThresholds:
    hard_fail: dict
    release: dict
    certified: dict

    @classmethod
    def load(cls, path: Path | None = None) -> "GateThresholds":
        p = Path(path) if path else _DEFAULT_CFG
        d = yaml.safe_load(p.read_text())
        return cls(hard_fail=d["hard_fail"], release=d["release"],
                   certified=d["certified"])


def _mean_confidence(table: CanonicalTable) -> float:
    vals = []
    for s in table.streams.values():
        if s.valid.any():
            vals.append(float(np.mean(s.confidence[s.valid])))
    return float(np.mean(vals)) if vals else 0.0


def _sync_ceiling(sync_result) -> str:
    raw = getattr(getattr(sync_result, "validation_report", None), "gate_verdict", None)
    if raw is None:
        return "CERTIFIED"
    v = getattr(raw, "value", raw)
    return str(v).upper()


def release_gate(table: CanonicalTable, sync_result, checks: dict[str, str],
                 *, lineage_signed: bool, thresholds: GateThresholds | None = None
                 ) -> QAReport:
    th = thresholds or GateThresholds.load()
    coverage_min = table.coverage_min()
    missing_pct = (1.0 - coverage_min) * 100.0
    sync_resid = float(sync_result.metrics.get("sync_resid_us", 0.0))
    label_conf = _mean_confidence(table)

    integrity = {"missing_frame_pct": missing_pct, "coverage_min": coverage_min,
                 "lineage_signed": lineage_signed}
    quality = {"sync_resid_us": sync_resid, "label_confidence": label_conf}

    hf = th.hard_fail
    hard = [
        missing_pct >= hf["missing_frame_pct"],
        sync_resid >= hf["sync_resid_us"],
        label_conf < hf["label_confidence"],
        hf.get("require_lineage_signed", True) and not lineage_signed,
        any(checks.get(c) == "fail" for c in _CRITICAL_CHECKS),
    ]
    if any(hard):
        verdict = "BLOCKED"
    else:
        any_fail = any(v == "fail" for v in checks.values())
        rel = th.release
        publishable = (missing_pct < rel["missing_frame_pct"]
                       and sync_resid < rel["sync_resid_us"]
                       and label_conf >= rel["label_confidence"]
                       and not any_fail)
        if not publishable:
            verdict = "NEEDS_REVIEW"
        else:
            ct = th.certified
            certified = (missing_pct < ct["missing_frame_pct"]
                         and sync_resid < ct["sync_resid_us"]
                         and label_conf >= ct["label_confidence"])
            verdict = "CERTIFIED" if certified else "RELEASE"

    # inherit the sync ceiling — never exceed it
    ceiling = _sync_ceiling(sync_result)
    if _ORDER.get(verdict, 0) > _ORDER.get(ceiling, 3):
        verdict = ceiling

    detail = (f"missing={missing_pct:.3f}% sync_resid={sync_resid:.1f}us "
              f"label_conf={label_conf:.3f} lineage_signed={lineage_signed} "
              f"sync_ceiling={ceiling}")
    return QAReport(gate_verdict=verdict, integrity=integrity, quality=quality,
                    checks=checks, detail=detail)
