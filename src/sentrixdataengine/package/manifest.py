"""Dataset manifest + write-back of ExportRecord into the Session.

The manifest links back to the originating Session (by id/clock/timeline) instead
of duplicating it. The only mutation of SentrixSync state is appending an
ExportRecord — additive and contract-allowed (SESSION_SCHEMA.md §8).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import DatasetSpec, ExportResult, QAReport


def write_manifest(path: Path, spec: DatasetSpec, exports: list[ExportResult],
                   qa: QAReport, provenance, *, content_hash: str) -> Path:
    manifest = {
        "dataset_id": spec.dataset_id,
        "version": spec.version,
        "engine_version": spec.engine_version,
        "schema_version": spec.schema_version,
        "profile": spec.profile,
        "source_session_id": spec.session_id,
        "reference_clock_id": spec.reference_clock_id,
        "grid_rate_hz": spec.grid_rate_hz,
        "content_hash": content_hash,
        "qa_verdict": qa.gate_verdict,
        "provenance": {"merkle_root": provenance.merkle_root,
                       "algorithm": provenance.algorithm, "signed": provenance.signed},
        "formats": [
            {"format": e.format, "uri": str(e.out_dir),
             "frame_count": e.frame_count, "sample_count": e.sample_count}
            for e in exports
        ],
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return Path(path)


def append_export_record(session, exports: list[ExportResult], *,
                         produced_at: str | None = None) -> int:
    """Append one ExportRecord per produced format to the Session. Returns count.

    No-op (returns 0) when `session` is None. Imports ExportRecord lazily so the
    engine has no hard import-time dependency surface beyond annotations.
    """
    if session is None:
        return 0
    from sentrixsync.core.session import ExportRecord
    n = 0
    for e in exports:
        session.exports.append(ExportRecord(
            format=e.format, uri=str(e.out_dir), produced_at=produced_at,
            frame_count=int(e.frame_count), sample_count=int(e.sample_count),
            consumer_hint="sentrixdataengine"))
        n += 1
    return n
