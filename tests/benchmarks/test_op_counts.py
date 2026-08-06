"""The performance gate: assert store-operation counts against the baseline.

This is a correctness test over performance-relevant behaviour, not a timing
test. It has no thresholds and no tolerance for variance, so it is safe to run
in the ordinary CI matrix alongside everything else.
"""

import pytest
from benchmarks._baseline import render_diff
from benchmarks._counting import count, zero_fill
from benchmarks._registry import registry

BENCHMARKS = registry(tier="ci")


@pytest.mark.parametrize("bench", BENCHMARKS, ids=lambda b: b.name)
def test_op_counts(bench, bench_ctx, baseline):
    state = bench.setup(bench_ctx) if bench.setup is not None else bench_ctx

    with count() as counters:
        bench.run(state)

    observed = zero_fill(counters)

    if baseline.updating:
        baseline.record(bench.name, observed)
        pytest.skip("baseline updated")

    expected = baseline.expect(bench.name)
    assert observed == expected, render_diff(bench.name, expected, observed)


def test_registry_is_not_empty():
    # A scenario module failing to import would otherwise turn the whole gate
    # into a silent no-op.
    assert BENCHMARKS, "no benchmarks registered"
