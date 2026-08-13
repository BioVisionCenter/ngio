# The performance gate

Two deterministic instruments, each blind to what the other measures:

1. **Op counts** (`test_op_counts.py` + committed baselines) — *"did this
   change make ngio do more work?"* Exact integers, zero variance.
2. **Concurrency** (`test_concurrency.py` + `_concurrency_gate.py`) — *"does
   the work that should overlap actually overlap?"* Op counts are invariant to
   concurrency: a serial and a parallel `get_wells` tally identically, so a
   parallelism regression is invisible to instrument 1. The rendezvous store
   parks ops on zarr's IO loop until `k` are in flight together and a gauge
   records the maximum overlap — exact integers again, no wall-clock, no
   thresholds. Success costs nothing; only a regression pays one bounded
   timeout before failing legibly.

Neither instrument measures time. Wall-clock and memory benchmarks live in the
separate `ngio-benchmarks` repo, and the moto/HTTP fixtures under
`tests/stores/` are correctness fixtures, not performance instruments — a real
server adds variance, which is exactly what both gates exist to avoid.

The op-count gate works because ngio's regressions are algorithmic — metadata
re-parsed per call, one group opened per well, a graph executed twice — so they
are exact integers with zero variance. Counts are also backend-independent, so
a local measurement predicts the cost on S3 where every op is a network
round-trip; that is checked rather than assumed, by running every scenario
against both a local and an in-memory store.

```bash
pixi run -e test11 pytest tests/performance
pixi run -e test11 pytest tests/performance -p no:xdist --update-baseline
```

Fixtures are tiny and **uncompressed** on purpose: exact byte counts matter more
than realism, and a numcodecs bump must not be able to move a baseline.

## Zarr versions

A count is a property of ngio *and* of the zarr underneath it. One baseline
currently holds across every supported zarr (3.1.6, 3.2.1, 3.3.0), and each file
records the version it was generated on under `generated_with.zarr`.

When counts differ on some *other* zarr the test skips rather than fails: that
is upstream drift, and `CI (pip)` installs dependencies unpinned, so an upstream
release must not fail this gate for a change nobody here made. `test11` runs the
generating version and asserts strictly on every PR.

The `*_sharded_*` scenarios are the first to have broken that single-baseline
property, and they are expected to keep doing so — sharding is where zarr's own
IO changes most between releases. On 3.1.6 a full-shard write costs two chunk
reads, one per shard, probing for content that is not there yet; on 3.2.1 that
probe is gone and the count is zero. So those scenarios skip on `test12`-`test14`
while the rest of the gate keeps asserting, and the baselines record the 3.1.6
numbers. A skip here is upstream drift; only `test11` holds these strictly.

**When a zarr bump lands in `pixi.lock`, regenerate the baselines** — otherwise
the pinned envs start skipping instead of asserting, and the gate goes quiet.
`test_instrumentation_complete` is version-agnostic and always runs; it fails
when zarr adds or removes a store method the counters do not account for.

There is no history file — the baselines are committed JSON, so git already is
the history:

```bash
git log -p tests/performance/baselines/local.json
```
