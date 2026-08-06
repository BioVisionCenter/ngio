# ngio benchmarks

Two layers share one set of scenario declarations in `scenarios/`.

## Op counters — the CI gate

Counts store operations and bytes per scenario and asserts them against a
committed baseline. There are no thresholds: a count that moves is a failure
until someone regenerates the baseline deliberately, and the regeneration shows
up in review as an explicit diff.

This works as a *gate* because ngio's performance problems are algorithmic
rather than constant-factor — metadata re-parsed once per call, one group
opened per well — so they are exact integers with zero run-to-run variance.
It works as a *remote* signal because counts are backend-independent and every
store op on S3 is a network round-trip, so a local measurement predicts remote
cost without any credentials in CI.

```bash
# Run the gate (also runs as part of the normal test suite).
pixi run -e test11 pytest tests/benchmarks

# Regenerate the baseline after an intended change. Cannot run under xdist.
pixi run -e test11 pytest tests/benchmarks -p no:xdist --bench-update-baseline
```

Baselines live in `baselines/`. One file currently covers zarr 3.1/3.2 and
Python 3.11/3.13 — verified identical, so no per-version files are needed.

## Wall clock

Not built yet. It will be a separate, non-gating, manually-run layer under
`timing/`, kept off `testpaths` so the default test run never collects it.

## Adding a scenario

```python
@benchmark(name="images/my_scenario", area="images", fixture="img_v05")
def my_scenario(ctx):
    return open_ome_zarr_container(ctx.store_for("img_v05"), mode="r")
```

Pass `setup=` for the part that should not be measured — it runs outside the
count block, so "open once, then read 16 ROIs" attributes only the reads. Then
regenerate the baseline and commit the new entry.

Fixtures are declared in `_fixtures.py` and generated with
`create_synthetic_ome_zarr`; nothing is downloaded. They use `compressors=None`
so byte counts are exact and a numcodecs bump cannot move the baseline.
