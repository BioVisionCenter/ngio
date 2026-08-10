"""Wall-clock and peak-memory measurement for ngio.

The other half of ngio's performance work is `tests/performance/`, which gates
exact store-operation counts in CI. This half answers what counts structurally
cannot: how does it behave at scale, and will it fit in memory. Nothing here
is committed and nothing gates -- the numbers depend on the machine.

See `README.md`; run it with `python -m benchmarks`.
"""
