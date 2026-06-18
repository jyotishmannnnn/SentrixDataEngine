"""Pipeline — resolve → materialize → validate → export → package.

Deterministic, offline. Consumes a SyncResult (+ optional Session) and produces a
packaged, validated, provenance-stamped dataset under a Gold root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .contracts import DatasetSpec, ExportResult, MaterializationRequest, QAReport
from .exporters import get_exporter
from .hooks import authorize, watermark
from .materialize import write_silver
from .materialize.canonical import CanonicalTable
from .materialize.projector import project
from .package import (
    GoldLayout,
    append_export_record,
    content_hash,
    derive_dataset_id,
    stamp_provenance,
    write_datacard,
    write_manifest,
)
from .package.versioning import sha256_file
from .resolve import default_registry
from .validate import (
    check_confidence,
    check_metadata,
    check_schema,
    check_timeline,
    release_gate,
)


@dataclass
class PipelineResult:
    spec: DatasetSpec
    canonical: CanonicalTable
    exports: list[ExportResult]
    qa: QAReport
    layout: GoldLayout
    content_hash: str
    provenance: object
    extras: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, registry=None) -> None:
        self.registry = registry or default_registry()

    # ---- helpers ---- #
    @staticmethod
    def _descriptors(session) -> dict:
        if session is None:
            raise ValueError("MaterializationRequest.session is required in V1 "
                             "(supplies device descriptors and stream refs)")
        out = {}
        for reg in session.devices:
            if reg.descriptor is None:
                raise ValueError(f"device {reg.device_id!r} has no inline descriptor")
            out[reg.device_id] = reg.descriptor
        return out

    @staticmethod
    def _normalize_uri(ref: str) -> str:
        if "://" in ref:
            return ref
        p = Path(ref)
        if p.suffix.lower() == ".parquet":
            return "parquet://" + str(p.resolve()).replace("\\", "/")
        return ref

    def _payload_sources(self, req: MaterializationRequest) -> dict[str, str]:
        sources: dict[str, str] = {}
        if req.session is not None:
            for reg in req.session.devices:
                for sid, ref in (reg.stream_refs or {}).items():
                    sources[f"{reg.device_id}::{sid}"] = self._normalize_uri(ref)
        sources.update({k: self._normalize_uri(v)
                        for k, v in req.payload_sources.items()})
        return sources

    @staticmethod
    def _source_hashes(payload_sources: dict[str, str]) -> list[str]:
        hashes = set()
        for uri in payload_sources.values():
            if "://" in uri:
                rest = uri.split("://", 1)[1].split("#", 1)[0]
                p = Path(rest)
                if p.is_file():
                    hashes.add(sha256_file(p))
        return sorted(hashes)

    # ---- main ---- #
    def run(self, req: MaterializationRequest) -> PipelineResult:
        if not authorize(customer_id=req.customer_id,
                         dataset_id=req.dataset_id or "",
                         formats=req.formats):
            raise PermissionError("authorization denied for export")

        sr = req.sync_result
        session_id = sr.metrics.get("session_id") or _session_id(req.session, sr)
        descriptors = self._descriptors(req.session)
        payload_sources = self._payload_sources(req)

        canonical = project(sr, descriptors, self.registry, payload_sources,
                            session_id=session_id, schema_version=SCHEMA_VERSION)
        canonical.extra["source_episode_hashes"] = self._source_hashes(payload_sources)

        dataset_id = req.dataset_id or derive_dataset_id(session_id)
        spec = DatasetSpec(
            dataset_id=dataset_id, version=req.version, session_id=session_id,
            reference_clock_id=canonical.reference_clock_id,
            grid_rate_hz=canonical.grid_rate_hz, schema_version=SCHEMA_VERSION,
            profile=req.profile, engine_version=__version__)

        layout = GoldLayout(Path(req.out_root), dataset_id, req.version).ensure()

        # Silver (canonical) — retained alongside Gold projections.
        silver_path = write_silver(canonical, layout.silver_dir)

        # Gold projections.
        exports: list[ExportResult] = []
        gold_files: list[Path] = [silver_path]
        for fmt in req.formats:
            res = get_exporter(fmt).export(canonical, spec, layout.format_dir(fmt),
                                           options=req.format_options.get(fmt))
            exports.append(res)
            gold_files.extend(res.files)

        # Validation.
        checks: dict[str, str] = {}
        checks.update(check_schema(canonical))
        checks.update(check_timeline(canonical))
        checks.update(check_metadata(canonical))
        checks.update(check_confidence(canonical))

        # Provenance over all written content.
        prov = stamp_provenance(
            gold_files, layout.provenance_path, dataset_id=dataset_id,
            version=req.version, session_id=session_id, schema_version=SCHEMA_VERSION,
            source_episode_hashes=canonical.extra["source_episode_hashes"],
            customer_id=req.customer_id)
        wm = watermark(customer_id=req.customer_id, dataset_id=dataset_id)

        qa = release_gate(canonical, sr, checks, lineage_signed=prov.signed)
        layout.qa_path.write_text(_json(qa.__dict__), encoding="utf-8")

        chash = content_hash(gold_files)
        write_manifest(layout.manifest_path, spec, exports, qa, prov, content_hash=chash)
        write_datacard(layout.datacard_path, spec, canonical, exports, qa, prov)

        append_export_record(
            req.session, exports,
            produced_at=datetime.now(timezone.utc).isoformat())

        return PipelineResult(spec=spec, canonical=canonical, exports=exports, qa=qa,
                              layout=layout, content_hash=chash, provenance=prov,
                              extras={"watermark": wm, "silver_path": str(silver_path)})


def _session_id(session, sync_result) -> str:
    if session is not None:
        return session.metadata.session_id
    return f"sess_{sync_result.reference_clock_id}"


def _json(d: dict) -> str:
    import json
    def enc(o):
        if isinstance(o, Path):
            return str(o)
        return str(o)
    return json.dumps(d, indent=2, default=enc)
