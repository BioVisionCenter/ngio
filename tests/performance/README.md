# The performance gate

Asserts exact store-operation counts against committed baselines. Answers *"did
this change make ngio do more work?"* — pass/fail, no thresholds, runs in CI
with everything else.

It works as a gate because ngio's regressions are algorithmic — metadata
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
