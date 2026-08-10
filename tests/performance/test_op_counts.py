"""Assert store-operation counts against the committed baselines.

This is a correctness test over performance-relevant behaviour, not a timing
test. No thresholds, no tolerance for variance, nothing machine-dependent — so
it runs in the ordinary CI matrix like any other test.
"""

import pytest
from tests.performance._baseline import render_diff
from tests.performance._counting import count, zero_fill
from tests.performance._probes import probes
from tests.performance.scenarios import SCENARIOS


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

    assert observed == baseline.expect(name), render_diff(
        name, baseline.expect(name), observed
    )
