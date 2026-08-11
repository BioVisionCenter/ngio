"""Assert store-operation counts against the committed baselines.

This is a correctness test over performance-relevant behaviour, not a timing
test. No thresholds, no tolerance for variance, nothing machine-dependent — so
it runs in the ordinary CI matrix like any other test.
"""

import pytest
from tests.performance._baseline import render_diff
from tests.performance._counting import (
    assert_instrumentation_complete,
    count,
    zero_fill,
)
from tests.performance._probes import probes
from tests.performance.scenarios import SCENARIOS


def test_instrumentation_complete():
    """Every counter is still wired to the store surface zarr actually exposes.

    Standalone and fixture-free on purpose: when a zarr upgrade moves the
    surface this fails once, legibly, instead of erroring the setup of every
    scenario in the module.
    """
    assert_instrumentation_complete()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_op_counts(name, ctx, baseline):
    scenario = SCENARIOS[name]
    state = scenario.setup(ctx) if scenario.setup is not None else ctx

    with count() as counters, probes():
        scenario.run(state)

    observed = zero_fill(counters)

    if baseline.updating:
        baseline.record(name, observed)
        pytest.skip("baseline updated")

    expected = baseline.expect(name)
    if observed != expected:
        baseline.explain_mismatch(name)
    assert observed == expected, render_diff(name, expected, observed)
