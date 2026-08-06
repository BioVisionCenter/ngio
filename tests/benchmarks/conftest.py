"""Fixtures for the op-count gate.

Lives under `tests/` on purpose: it inherits `testpaths`, the xdist run, the
`tests/**` ruff and check-manifest ignores, and the root conftest's
`NgioUserWarning` filter, without which `filterwarnings = ["error"]` would turn
ordinary store warnings into failures.
"""

import platform
import sys

import pytest
import zarr
from benchmarks._baseline import load_baseline, save_baseline
from benchmarks._counting import assert_instrumentation_complete, counting_store
from benchmarks._fixtures import FIXTURES, build_fixture

import ngio

#: Phase 1 gates a single hardcoded profile. Named store profiles arrive with
#: the prepare/run lifecycle.
GATE_PROFILE = "local"


def pytest_addoption(parser):
    """Register the baseline regeneration flag."""
    parser.addoption(
        "--bench-update-baseline",
        action="store_true",
        default=False,
        help="Rewrite the committed op-count baseline from this run.",
    )


class _BenchContext:
    """Hands each benchmark a counting store rooted at its fixture."""

    def __init__(self, roots):
        self._roots = roots

    def store_for(self, fixture: str):
        return counting_store(self._roots[fixture])


@pytest.fixture(scope="session")
def bench_ctx(tmp_path_factory):
    """Build every CI-tier fixture once per worker, outside any count block.

    `tmp_path_factory` already gives each xdist worker its own base directory,
    so parallel workers cannot race on generation.
    """
    assert_instrumentation_complete()
    base = tmp_path_factory.mktemp("ngio-bench")
    roots = {}
    for name, spec in FIXTURES.items():
        if spec.tier != "ci":
            continue
        root = base / f"{name}.zarr"
        build_fixture(spec, root)
        roots[name] = root
    return _BenchContext(roots)


@pytest.fixture(scope="session")
def baseline(request):
    """Load the committed baseline, or collect a new one under the flag."""
    updating = request.config.getoption("--bench-update-baseline")
    if updating and hasattr(request.config, "workerinput"):
        pytest.fail(
            "--bench-update-baseline cannot run under xdist; drop -n, e.g. "
            "`pixi run -e test11 pytest tests/benchmarks -p no:xdist "
            "--bench-update-baseline`"
        )
    collected: dict[str, dict[str, int]] = {}
    data = load_baseline(GATE_PROFILE)

    class _Baseline:
        updating = False

        def expect(self, name):
            if data is None or name not in data.get("benchmarks", {}):
                pytest.fail(
                    f"no committed baseline for {name!r}. Generate it with:\n"
                    "  pixi run -e test11 pytest tests/benchmarks -p no:xdist "
                    "--bench-update-baseline"
                )
            return data["benchmarks"][name]

        def record(self, name, counters):
            collected[name] = counters

    holder = _Baseline()
    holder.updating = updating
    yield holder

    if updating:
        env = {
            "ngio": ngio.__version__,
            "zarr": zarr.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system().lower(),
        }
        path = save_baseline(GATE_PROFILE, collected, env)
        print(f"\nbaseline written: {path} ({len(collected)} benchmarks)")
