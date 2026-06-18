"""Human-readable data card for a materialized dataset."""
from __future__ import annotations

from pathlib import Path

from ..contracts import DatasetSpec, ExportResult, QAReport
from ..materialize.canonical import CanonicalTable


def write_datacard(path: Path, spec: DatasetSpec, table: CanonicalTable,
                   exports: list[ExportResult], qa: QAReport, provenance) -> Path:
    lines = [
        f"# Dataset {spec.dataset_id} (v{spec.version})",
        "",
        f"- **Source session:** `{spec.session_id}`",
        f"- **Reference clock:** `{spec.reference_clock_id}`",
        f"- **Grid rate:** {spec.grid_rate_hz:.1f} Hz",
        f"- **Schema version:** {spec.schema_version}",
        f"- **Engine version:** {spec.engine_version}",
        f"- **QA verdict:** **{qa.gate_verdict}** — {qa.detail}",
        f"- **Provenance:** {provenance.algorithm} "
        f"(signed={provenance.signed}), merkle `{provenance.merkle_root[:16]}…`",
        "",
        "## Streams",
        "",
        "| stream | device | payload_kind | shape | units | kernel | coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in table.streams.values():
        lines.append(
            f"| {s.stream_id} | {s.device_id} | {s.payload_kind} | "
            f"{list(s.shape)} | {s.units} | {s.kernel} | {s.coverage():.3f} |")
    lines += ["", "## Formats", ""]
    for e in exports:
        lines.append(f"- **{e.format}** — {e.frame_count} frames, "
                     f"{e.sample_count} samples → `{e.out_dir}`")
    lines += [
        "", "## Notes", "",
        "- Gaps are flagged (`valid=False`, NaN value); never interpolated across a dropout.",
        "- Confidence is the export scalar `source*clock*interp`; components retained in Silver.",
        "- Synthetic source: no PII; redaction N/A.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)
