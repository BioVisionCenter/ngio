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

There is no history file — the baselines are committed JSON, so git already is
the history:

```bash
git log -p tests/performance/baselines/local.json
```
