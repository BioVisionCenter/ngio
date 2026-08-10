"""Fixtures for the performance gate.

This is a test, not a benchmark: it asserts exact store-operation counts and
has no notion of time. ngio's performance regressions are algorithmic —
metadata re-parsed per call, one group opened per well — so they are exact
integers with zero variance, which makes them safe to run in the ordinary CI
matrix alongside everything else.

Counts are also backend-independent, so measuring against a local directory
predicts what an operation costs on S3, where every store op is a network
round-trip. That is checked rather than assumed: every scenario runs against
both a local and an in-memory store, each with its own committed baseline.

Fixtures are built fresh into `tmp_path_factory` per session. That keeps the
gate hermetic — nothing to invalidate, no staleness logic, and xdist workers
isolated for free.
"""

import platform
import sys

import pytest
import zarr
from tests.performance._baseline import load_baseline, save_baseline
from tests.performance._counting import (
    assert_instrumentation_complete,
    counting_store,
)

import ngio
from ngio import ImageInWellPath, create_empty_plate, open_ome_zarr_container
from ngio.images._create_synt_container import create_synthetic_ome_zarr
from ngio.tables import FeatureTable

#: The two store kinds under test. Hardcoded: counts proved backend-independent
#: for the image and plate paths, and the pair is what keeps that claim honest.
STORE_KINDS = ("local", "memory")

#: Deliberately tiny and uncompressed. Exact byte counts matter more than
#: realism here, and a numcodecs bump must not be able to move a baseline.
_IMAGE = {
    "shape": (2, 8, 256, 256),
    "axes_names": ("c", "z", "y", "x"),
    "channels_meta": ("Channel 1", "Channel 2"),
    "levels": 3,
    "chunks": (1, 1, 128, 128),
    "compressors": None,
    "table_backend": "anndata_v1",
}


def pytest_addoption(parser):
    """Register the baseline regeneration flag."""
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Rewrite the committed op-count baselines from this run.",
    )


def _build_image(target, **overrides):
    params = {**_IMAGE, **overrides}
    create_synthetic_ome_zarr(
        store=target,
        shape=tuple(params.pop("shape")),
        axes_names=list(params.pop("axes_names")),
        channels_meta=list(params.pop("channels_meta")),
        overwrite=True,
        **params,
    )


def _build_plate(target):
    # Images are registered in plate metadata but no arrays are created: that
    # is enough for the enumeration paths, which is what costs, and it is the
    # only form that builds identically on a MemoryStore.
    images = [
        ImageInWellPath(row=row, column=f"{col + 1:02d}", path="0")
        for row in ("A", "B", "C", "D")
        for col in range(6)
    ]
    create_empty_plate(target, name="bench_plate", images=images, overwrite=True)


def _build_tables(target):
    import pandas as pd

    _build_image(target)
    container = open_ome_zarr_container(target, mode="r+")
    frame = pd.DataFrame(
        {
            "label": range(1, 501),
            **{f"feature_{i}": [float(i)] * 500 for i in range(8)},
        }
    ).set_index("label")
    for backend in ("anndata_v1", "experimental_json_v1"):
        container.add_table(
            name=f"features_{backend}",
            table=FeatureTable(table_data=frame, reference_label=None),
            backend=backend,
            overwrite=True,
        )


_BUILDERS = {
    "image": lambda t: _build_image(t, ngff_version="0.5"),
    "image_v04": lambda t: _build_image(t, ngff_version="0.4"),
    "plate": _build_plate,
    "tables": _build_tables,
}


class _Context:
    """Hands each scenario a counting store rooted at a named fixture."""

    def __init__(self, targets, scratch_factory):
        self._targets = targets
        self._scratch_factory = scratch_factory

    def store(self, fixture: str):
        return counting_store(self._targets[fixture])

    def scratch(self, name: str):
        """A writable throwaway target, outside the read-only fixtures.

        Counting, like `store`: a write scenario that built its target from a
        raw path would silently record zeros for everything.
        """
        return counting_store(self._scratch_factory(name))


@pytest.fixture(scope="session", params=STORE_KINDS)
def store_kind(request):
    """The store kind under test."""
    return request.param


@pytest.fixture(scope="session")
def ctx(store_kind, tmp_path_factory):
    """Build every fixture once per store kind, outside any count block."""
    assert_instrumentation_complete()
    if store_kind == "memory":
        # A MemoryStore is rooted at its dict, so each fixture needs its own.
        targets = {name: {} for name in _BUILDERS}
        scratch = {}

        def scratch_factory(name):
            return scratch.setdefault(name, {})
    else:
        base = tmp_path_factory.mktemp("perf-local")
        targets = {name: base / f"{name}.zarr" for name in _BUILDERS}

        def scratch_factory(name):
            return base / "scratch" / f"{name}.zarr"

    for name, build in _BUILDERS.items():
        build(targets[name])
    return _Context(targets, scratch_factory)


@pytest.fixture(scope="session")
def baseline(request, store_kind):
    """Load the committed baseline, or collect a new one under the flag."""
    updating = request.config.getoption("--update-baseline")
    if updating and hasattr(request.config, "workerinput"):
        pytest.fail(
            "--update-baseline cannot run under xdist; drop -n, e.g.\n"
            "  pixi run -e test11 pytest tests/performance -p no:xdist "
            "--update-baseline"
        )
    collected: dict[str, dict[str, int]] = {}
    data = load_baseline(store_kind)

    class _Baseline:
        updating = False

        def expect(self, name):
            if data is None or name not in data.get("scenarios", {}):
                pytest.fail(
                    f"no committed baseline for {name!r} on the {store_kind!r} "
                    "store. Generate it with:\n"
                    "  pixi run -e test11 pytest tests/performance -p no:xdist "
                    "--update-baseline"
                )
            return data["scenarios"][name]

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
        save_baseline(store_kind, collected, env)
