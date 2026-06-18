from __future__ import annotations

import numpy as np

from sentrixdataengine import SCHEMA_VERSION
from sentrixdataengine.materialize.projector import project
from sentrixdataengine.resolve import default_registry


def test_project_builds_canonical(sync_result, descriptor, sim_parquet):
    sources = {"glove_L::tactile_field":
               "parquet://" + str(sim_parquet).replace("\\", "/")}
    table = project(sync_result, {"glove_L": descriptor}, default_registry(), sources,
                    session_id="01J9SYNTH0001", schema_version=SCHEMA_VERSION)
    assert "glove_L::tactile_field" in table.streams
    s = table.streams["glove_L::tactile_field"]
    assert s.shape == (21, 3)
    assert s.values.shape == (table.n_grid, 21, 3)
    # full coverage on a clean, aligned single-device stream
    assert s.coverage() == 1.0
    # no NaNs where valid
    assert not np.isnan(s.values[s.valid]).any()


def test_missing_payload_source_raises(sync_result, descriptor):
    try:
        project(sync_result, {"glove_L": descriptor}, default_registry(), {},
                session_id="x", schema_version=SCHEMA_VERSION)
        assert False, "expected KeyError"
    except KeyError:
        pass
