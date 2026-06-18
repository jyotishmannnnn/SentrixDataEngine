from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sentrixdataengine.resolve import McapPayloadResolver


def _write_sim_mcap(path: Path, n: int = 8) -> None:
    from mcap.writer import Writer
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="tactile", encoding="jsonschema",
                                data=json.dumps({"type": "object"}).encode())
        ch = w.register_channel(topic="tactile_field", message_encoding="json",
                                schema_id=sid)
        rng = np.random.default_rng(0)
        for i in range(n):
            B = rng.normal(0, 1, (21, 3)).round(3).tolist()
            msg = {"t_us": i * 625, "B_uT": B, "saturated": False}
            w.add_message(channel_id=ch, log_time=i * 625 * 1000,
                          publish_time=i * 625 * 1000, data=json.dumps(msg).encode())
        w.finish()


def test_mcap_resolver_reads_tactile(tmp_path):
    p = tmp_path / "episode.mcap"
    _write_sim_mcap(p, n=8)
    r = McapPayloadResolver()
    assert r.supports("mcap")
    arr = r.resolve_stream("mcap://" + str(p).replace("\\", "/"),
                           "bmm350_cluster_uT", (21, 3))
    assert arr.shape == (8, 21, 3)
    assert arr.dtype == np.float32


def test_mcap_resolver_unknown_kind(tmp_path):
    p = tmp_path / "e.mcap"
    _write_sim_mcap(p, n=2)
    try:
        McapPayloadResolver().resolve_stream("mcap://" + str(p).replace("\\", "/"),
                                             "nope", (1,))
        assert False
    except ValueError:
        pass
