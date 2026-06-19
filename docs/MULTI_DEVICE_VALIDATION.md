# Multi-Device Synchronization & Materialization Validation

**Objective:** prove the Sentrix stack (SentrixSync → SentrixDataEngine) behaves
correctly under realistic multi-device conditions — not just the identity-clock,
single-device path exercised by the unit tests.

**Artifact:** `benchmarks/multi_device_benchmark.py` (deterministic, seed=7).
**Durable check:** `tests/test_multi_device.py` (7 assertions, part of the suite —
**41 passed** total). Reproduce with `python benchmarks/multi_device_benchmark.py`.

> Scope note: this is validation only. No new exporters, formats, or architecture
> were added. Two genuine multi-device defects in existing exporters were found and
> fixed (see §8); everything else is exercise of the as-built stack.

---

## 1. Scenario

| Device | Stream | Rate | Kernel | True clock (injected) | Role |
|---|---|---|---|---|---|
| **A** | `tactile_field` [21,3] | 1000 Hz | continuous | identity (α=1, β=0) | reference / anchor |
| **B** | `tactile_field` [21,3] | 1000 Hz | continuous | β=+5000 µs, +40 ppm (α=1.00004) | follower |
| **C** | `dynamics` (IMU accel) [3,3] | 200 Hz | continuous | β=−12000 µs, −25 ppm (α=0.999975) | follower |
| **D** | `tactile_field` [21,3] | 500 Hz | continuous | β=+8000 µs, +60 ppm (α=1.00006) | follower, **5% sample dropout** |

- Reference window: **2.0 s**; reference grid 1000 Hz; rejection tolerance 6000 µs
  (so the 200 Hz device covers a 1 kHz grid via interpolation).
- **24 shared events** spread across the window over rotating subsets
  `{A,B,C}, {A,B}, {A,C}, {B,C,D}, {C,D}, {B,D}, …`. Each event observation carries
  Gaussian detector jitter (σ=25 µs).
- **Partial observability (the key stress):** **A and D never co-observe any event.**
  D can only be reconciled *transitively* through B and C.
- Clock corruption is injected with `sentrixsync.clock.forward.ForwardClock` and the
  same objects are passed back as segregated `ground_truth`, so recovery is measured
  against truth (`metrics.roundtrip_accuracy`).

Why these magnitudes: a 2 s baseline with 25 µs jitter puts the skew noise floor at
≈ σ/baseline ≈ 12 ppm, **below** the injected 25–60 ppm, so drift is observable. (An
earlier 0.2 s / 60 µs setup had a ~300 ppm floor that swamped the injected skew — a
scenario-design issue, documented here so it is not repeated.)

---

## 2. Expected vs recovered clock parameters

| Device | α true | α recovered | α error | β true (µs) | β recovered (µs) | β error (µs) | hops |
|---|---|---|---|---|---|---|---|
| A | 1.000000 | 1.000000 | 0 | 0 | 0.000 | 0.000 | 0 |
| B | 1.000040 | 1.000049 | 9.0e-06 | +5000 | 4990.738 | 9.262 | 1 |
| C | 0.999975 | 0.999960 | 1.55e-05 | −12000 | −11969.806 | 30.194 | 1 |
| D | 1.000060 | 1.000031 | 2.95e-05 | +8000 | 8045.535 | 45.535 | 2 |

All skews recovered to **< 30 ppm error**; offsets to **< 50 µs**. The reference is
exact by construction (identity model, `clock_confidence = 1.0`).

---

## 3. Residual / error analysis

| Metric | Value |
|---|---|
| `sync_resid_us` (RMS cross-observer spread) | **36.8 µs** |
| alignment RMSE — B (1 hop) | 5.21 µs |
| alignment RMSE — C (1 hop) | 17.06 µs |
| alignment RMSE — D (2 hop) | 23.31 µs |
| per-edge fit residual | 0.12–0.25 µs |

**Reading:** the 2-hop device (D) carries the largest alignment error — expected,
because error compounds when composing affine maps along a reconciliation path
(`graph._compose`). The residual (~37 µs) is consistent with the 25 µs injected
jitter. The architecture recovers injected drift/offset correctly and reports an
honest residual.

---

## 4. Confidence behavior analysis

Per-stream confidence (three components kept separate; scalar = source·clock·interp):

| Stream | source mean | clock min | clock max | clock mean | interp mean | scalar mean |
|---|---|---|---|---|---|---|
| A::tactile_field | 1.000 | 0.848 | 1.000 | 0.945 | 0.998 | 0.943 |
| B::tactile_field | 1.000 | 0.814 | 0.884 | 0.857 | 0.992 | 0.850 |
| C::dynamics | 0.9995 | 0.000 | 0.882 | 0.854 | 0.798 | 0.682 |
| D::tactile_field | 0.9985 | 0.000 | 0.780 | 0.731 | 0.904 | 0.662 |

**Reading — every component does its job:**
- **source** < 1.0 for C and D reflects their dropped/absent samples (D = 5% dropout).
- **clock** is highest for the reference (A, ≈1.0 at events) and lower for followers;
  it **decays away from sync events** (clock_min < clock_max) — `clock_min = 0.0` for
  C/D occurs at grid regions far from any event that device observed. This is the
  `exp(−d/τ)` long-gap uncertainty model working as designed.
- **interp** is lowest for C (200 Hz on a 1 kHz grid → larger interpolation gaps).
- The single **scalar** is correctly the product, never inflated above any component.

---

## 5. Sub-frame bucketing — activated and materialized

The high-rate tactile burst of device A (1000 Hz) is bucketed against a 200 Hz anchor
(device C's rate) via `materialize_subframe`, and the index logic is cross-checked
against SentrixSync's approved `sync.join.subframe_buckets`:

| Field | Value |
|---|---|
| anchor fps | 200 |
| R (expected = `ceil(1000/200)`) | 5 |
| R (materialized) | **5** |
| tensor shape `[n_frames, R, 21, 3]` | **[401, 5, 21, 3]** |
| all frames valid | true |
| `m_k` unique counts | {5} |
| matches Sync's bucket indices | **true** |
| frame-0 sample order (first / last) | 0.0 / 4.0 |

This confirms the manual's premium-fidelity rule is **actually materialized** (a real
`[R,U,V]`-shaped tensor), not merely indexed — and that DataEngine's materialization
agrees byte-for-byte with SentrixSync's index contract.

> Note: in the default single-glove pipeline path there is no slow anchor stream, so
> sub-frame bucketing is not auto-invoked there. This benchmark drives it explicitly,
> which is the supported way to activate it today (pipeline auto-wiring via
> `subframe_anchor_fps` remains a documented future step, not built here).

---

## 6. End-to-end export through SentrixDataEngine

| Field | Value |
|---|---|
| Silver streams | `A::tactile_field, B::tactile_field, C::dynamics, D::tactile_field` |
| Gold formats written | parquet, lerobot, hdf5, mcap |
| grid points (n_grid) | 2007 |
| `ExportRecord`s appended to Session | 4 |
| content hash | deterministic (reproducible) |
| **QA verdict** | **BLOCKED** |
| QA detail | `missing=0.149% sync_resid=36.8us label_conf=0.785 lineage_signed=True sync_ceiling=NEEDS_REVIEW` |

**The BLOCKED verdict is correct behavior, not a failure.** Two independent,
honest reasons:
1. The **sync gate is `NEEDS_REVIEW`** (device D's 5.9% dropout exceeds the sync
   dropout band), and the dataset gate **inherits the sync verdict as a ceiling**
   (ADR-012) — it can never certify above the synchronization it rests on.
2. The **mean label confidence (0.785) is below the 0.85 release floor**
   (`configs/qa_thresholds.yaml`), pulled down by C/D's event-sparse,
   interpolation-stretched, partially-dropped streams.

So the gate refuses to ship a dataset built on a needs-review sync with sub-0.85
mean confidence — exactly the conservative behavior the release gate exists to
enforce. A clean, fully-observed, dense-event session reaches RELEASE/CERTIFIED;
this scenario is deliberately imperfect to exercise the gate.

---

## 7. What this validates

| Capability | Result |
|---|---|
| Multiple devices, different rates (1000/1000/200/500 Hz) | ✅ |
| Clock offsets + drift recovered (< 30 ppm, < 50 µs) | ✅ |
| Graph reconciliation, transitive (D via B/C, 2 hops, never sees A) | ✅ |
| Partial observability + dropout (D 5%) surfaced in coverage/dropout/confidence | ✅ |
| Shared events across overlapping subsets | ✅ |
| Three-component confidence propagation + decay | ✅ |
| Gaps preserved, never fabricated (coverage_min < 1) | ✅ |
| Sub-frame bucketing activated + materialized, matches Sync indices | ✅ |
| End-to-end export of all 4 streams to 4 Gold formats | ✅ |
| QA gate blocks honestly on a needs-review/low-confidence session | ✅ |

---

## 8. Defects found and fixed (multi-device exposed them)

Both were latent because every prior test used a single device with a unique
`stream_id`. They are correctness fixes to existing exporters, not new features.

1. **Feature-name collisions when devices share a `stream_id`.** A, B, and D all
   expose `tactile_field`; HDF5 (and Silver/LeRobot/MCAP/RLDS) keyed datasets/columns
   by bare `stream_id` → "name already exists" / ambiguous columns. **Fix:**
   `CanonicalTable.feature_names()` returns the bare `stream_id` when unique
   (single-device output unchanged) and disambiguates with the device id
   (`<device>.<stream>`) only on collision. All exporters + Silver + inspect use it.

2. **Negative reference time crashed MCAP.** With positive device offsets the
   reconciled reference grid can start slightly below zero; MCAP `log_time` is
   unsigned → `struct.error`. **Fix:** the MCAP exporter shifts log times to a
   non-negative, reference-relative base (spacing preserved); the true reference time
   is retained verbatim in each message's `t_ref_us`.

Regression coverage: `tests/test_multi_device.py` plus the unchanged single-device
suite (**41 passed** total).

---

## 9. Reproduce

```bash
# in the dev environment (sentrixsync + sentrixdataengine + mcap + h5py + cryptography)
python benchmarks/multi_device_benchmark.py        # prints the full JSON report
pytest tests/test_multi_device.py -q               # durable assertions
```
