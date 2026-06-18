from __future__ import annotations

import numpy as np

from sentrixdataengine import SCHEMA_VERSION
from sentrixdataengine.materialize.projector import project
from sentrixdataengine.resolve import default_registry
from sentrixdataengine.validate import (
    check_confidence,
    check_metadata,
    check_schema,
    check_timeline,
    release_gate,
)


def _canonical(sync_result, descriptor, sim_parquet):
    sources = {"glove_L::tactile_field":
               "parquet://" + str(sim_parquet).replace("\\", "/")}
    return project(sync_result, {"glove_L": descriptor}, default_registry(), sources,
                   session_id="01J9SYNTH0001", schema_version=SCHEMA_VERSION)


def _all_checks(table):
    checks = {}
    checks.update(check_schema(table))
    checks.update(check_timeline(table))
    checks.update(check_metadata(table))
    checks.update(check_confidence(table))
    return checks


def test_clean_session_certified_when_signed(sync_result, descriptor, sim_parquet):
    table = _canonical(sync_result, descriptor, sim_parquet)
    checks = _all_checks(table)
    assert all(v == "pass" for v in checks.values())
    qa = release_gate(table, sync_result, checks, lineage_signed=True)
    assert qa.gate_verdict == "CERTIFIED"


def test_unsigned_lineage_blocks(sync_result, descriptor, sim_parquet):
    table = _canonical(sync_result, descriptor, sim_parquet)
    checks = _all_checks(table)
    qa = release_gate(table, sync_result, checks, lineage_signed=False)
    assert qa.gate_verdict == "BLOCKED"


def test_fabricated_gap_blocks(sync_result, descriptor, sim_parquet):
    table = _canonical(sync_result, descriptor, sim_parquet)
    s = table.streams["glove_L::tactile_field"]
    # inject an invalid frame that still carries a (fabricated) value
    s.valid[0] = False
    s.values[0] = 0.0   # not NaN -> fabricated
    checks = _all_checks(table)
    assert checks["no_fabricated_gaps"] == "fail"
    qa = release_gate(table, sync_result, checks, lineage_signed=True)
    assert qa.gate_verdict == "BLOCKED"
