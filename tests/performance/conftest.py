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
from tests.performance._counting import counting_store

from ngio import (
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
)
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

#: The wells of the aggregation fixture. Four images is enough to make the
#: per-image cost of a plate-wide aggregation legible without paying for a full
#: plate of real arrays in session setup.
_AGG_WELLS = (("A", "01"), ("A", "02"), ("B", "01"), ("B", "02"))


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


def _build_image_no_tables(target):
    # No `/tables` group at all, which `_build_image` always creates. Answering
    # "this image has no tables" used to be the most expensive listing there
    # is, because the failed probe was never remembered.
    create_empty_ome_zarr(
        target,
        shape=(1, 4, 64, 64),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        overwrite=True,
    )


def _build_plate(target, ngff_version="0.4"):
    # Images are registered in plate metadata but no arrays are created: that
    # is enough for the enumeration paths, which is what costs, and it is the
    # only form that builds identically on a MemoryStore.
    #
    # 0.4 is `DefaultNgffVersion`, and it is also the version the decoder
    # registry tries first — so a 0.4 plate never pays a failed validation and
    # cannot see a regression in the version memo. `plate_v05` exists for that.
    images = [
        ImageInWellPath(row=row, column=f"{col + 1:02d}", path="0")
        for row in ("A", "B", "C", "D")
        for col in range(6)
    ]
    create_empty_plate(
        target,
        name="bench_plate",
        images=images,
        ngff_version=ngff_version,
        overwrite=True,
    )


def _feature_frame(rows: int, columns: int):
    import pandas as pd

    return pd.DataFrame(
        {
            "label": range(1, rows + 1),
            **{f"feature_{i}": [float(i)] * rows for i in range(columns)},
        }
    ).set_index("label")


def _build_tables(target):
    _build_image(target)
    container = open_ome_zarr_container(target, mode="r+")
    frame = _feature_frame(rows=500, columns=8)
    for backend in ("anndata_v1", "experimental_json_v1"):
        container.add_table(
            name=f"features_{backend}",
            table=FeatureTable(table_data=frame, reference_label=None),
            backend=backend,
            overwrite=True,
        )


def _build_plate_tables(target):
    # A plate with real images, unlike `_build_plate`: the aggregation paths
    # open every container and read a table from each, so registered paths are
    # not enough. `get_image_store` creates the group in place, keeping the
    # whole plate on one store (and so on one dict for the memory kind).
    plate = create_empty_plate(
        target,
        name="bench_plate_tables",
        images=[
            ImageInWellPath(row=row, column=column, path="0")
            for row, column in _AGG_WELLS
        ],
        overwrite=True,
    )
    frame = _feature_frame(rows=100, columns=4)
    for row, column in _AGG_WELLS:
        container = create_empty_ome_zarr(
            store=plate.get_image_store(row=row, column=column, image_path="0"),
            shape=(1, 1, 64, 64),
            axes_names=["c", "z", "y", "x"],
            channels_meta=["Channel 1"],
            levels=1,
            pixelsize=(0.65, 0.65),
            chunks=(1, 1, 64, 64),
            compressors=None,
            overwrite=True,
        )
        container.add_table(
            name="features",
            table=FeatureTable(table_data=frame, reference_label=None),
            backend="anndata_v1",
            overwrite=True,
        )


_BUILDERS = {
    "image": lambda t: _build_image(t, ngff_version="0.5"),
    "image_v04": lambda t: _build_image(t, ngff_version="0.4"),
    "image_no_tables": _build_image_no_tables,
    "plate": _build_plate,
    "plate_v05": lambda t: _build_plate(t, ngff_version="0.5"),
    "plate_tables": _build_plate_tables,
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

    baseline_zarr = (data or {}).get("generated_with", {}).get("zarr")

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

        def explain_mismatch(self, name):
            """Skip a mismatch that an unpinned zarr explains, else return.

            The counts are a property of ngio *and* of the zarr it runs on, and
            `ci_upstream.yml` installs zarr unpinned — so an upstream release
            that reshuffles zarr's own IO would fail this gate for a change
            nobody here made.

            Deliberately conditional on the counts *actually* differing rather
            than on the version alone: one baseline currently holds across
            every zarr ngio supports, and a version-only check would quietly
            stop asserting on the `test12`-`test14` envs, which run a different
            zarr from the one the baselines were generated on.
            """
            if baseline_zarr is None or zarr.__version__ == baseline_zarr:
                return
            pytest.skip(
                f"op counts for {name!r} differ under zarr {zarr.__version__}, "
                f"but the baseline was generated on zarr {baseline_zarr} — "
                "treating this as upstream drift, not an ngio regression. If "
                "ngio now pins this zarr, regenerate:\n"
                "  pixi run -e test11 pytest tests/performance -p no:xdist "
                "--update-baseline"
            )

        def record(self, name, counters):
            collected[name] = counters

    holder = _Baseline()
    holder.updating = updating
    yield holder

    if updating:
        # No ngio version here: `hatch-vcs` stamps it when the editable install
        # is built, not when the baseline is generated, so it lags the commit
        # being recorded and churns on a dirty tree. `git log -p` on the
        # baseline file is the accurate answer to "which change moved this".
        env = {
            "zarr": zarr.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system().lower(),
        }
        save_baseline(store_kind, collected, env)
