"""Concurrency assertions the op-count gate cannot express.

Each test states which overlap a code path must reach (or must not exceed) as
an exact integer — no baselines, no wall-clock. See `_concurrency_gate` for
why the rendezvous makes these deterministic.

Memory store only, module-scoped fixtures: the op-count gate already proves
counts are backend-independent, and these tests measure scheduling, which no
backend changes.
"""

import numpy as np
import pytest
import zarr

from ngio import (
    ImageInWellPath,
    create_empty_ome_zarr,
    create_empty_plate,
    open_image,
    open_ome_zarr_plate,
)
from tests.performance._concurrency_gate import concurrency_probe, rendezvous_store

_WELLS = [ImageInWellPath(row=row, column="01", path="0") for row in "ABCD"]


@pytest.fixture(scope="module")
def plate_target():
    target: dict = {}
    create_empty_plate(
        rendezvous_store(target), name="conc_plate", images=_WELLS, overwrite=True
    )
    return target


def _open_plate(plate_target):
    plate = open_ome_zarr_plate(rendezvous_store(plate_target), cache=True, mode="r")
    # Pre-warm the serial prefix (the plate document) so the ops inside the
    # probe are exactly the per-well fan-out reads.
    plate.wells_paths()
    return plate


def test_serial_get_wells_never_overlaps(plate_target):
    """`max_workers=1` is deliberately serial: one op in flight, structurally."""
    plate = _open_plate(plate_target)
    with concurrency_probe() as probe:
        plate.get_wells(max_workers=1)
    assert probe.arrivals > 0
    assert probe.max_in_flight == 1


def test_auto_get_wells_reads_all_wells_concurrently(plate_target):
    """The fan-out must put one op in flight per well, together.

    This is the assertion the op-count gate cannot make: the tally of a
    serial and a concurrent `get_wells` is identical, and on a remote store
    the difference is the whole prize.
    """
    plate = _open_plate(plate_target)
    with concurrency_probe(rendezvous=len(_WELLS)) as probe:
        plate.get_wells(max_workers="auto")
    assert probe.arrivals > 0
    assert probe.max_in_flight == len(_WELLS)


@pytest.fixture(scope="module")
def tables_plate_target():
    import pandas as pd

    from ngio.tables import FeatureTable

    target: dict = {}
    plate = create_empty_plate(
        rendezvous_store(target), name="conc_tables", images=_WELLS, overwrite=True
    )
    frame = pd.DataFrame({"label": [1, 2], "x": [0.5, 1.5]}).set_index("label")
    for well in _WELLS:
        container = create_empty_ome_zarr(
            store=plate.get_image_store(
                row=well.row, column=well.column, image_path="0"
            ),
            shape=(1, 1, 32, 32),
            axes_names=["c", "z", "y", "x"],
            channels_meta=["Channel 1"],
            levels=1,
            pixelsize=(0.65, 0.65),
            chunks=(1, 1, 32, 32),
            compressors=None,
            overwrite=True,
        )
        container.add_table(
            "features",
            FeatureTable(table_data=frame, reference_label=None),
            backend="anndata_v1",
            overwrite=True,
        )
    return target


def test_auto_list_image_tables_reads_images_concurrently(tables_plate_target):
    """The table aggregation fan-out is a separate call site from `get_wells`.

    `list_image_tables` drives `_map_workers` from `images/_table_ops.py`;
    a serial regression there would not show in the plate test above.
    """
    plate = open_ome_zarr_plate(
        rendezvous_store(tables_plate_target), cache=True, mode="r"
    )
    # Pre-warm the serial prefix (plate + well documents).
    plate.images_paths(max_workers=1)
    with concurrency_probe(rendezvous=len(_WELLS), kind="meta") as probe:
        plate.list_image_tables(max_workers="auto")
    assert probe.arrivals > 0
    assert probe.max_in_flight == len(_WELLS)


@pytest.fixture(scope="module")
def image_target():
    target: dict = {}
    container = create_empty_ome_zarr(
        rendezvous_store(target),
        shape=(256, 256),
        axes_names=["y", "x"],
        levels=1,
        pixelsize=(0.65, 0.65),
        chunks=(64, 64),
        compressors=None,
        overwrite=True,
    )
    container.get_image(path="0").set_array(np.ones((256, 256), dtype="uint16"))
    return target


def test_chunk_fanout_is_bounded_by_async_concurrency(image_target):
    """One 16-chunk read overlaps exactly `async.concurrency` fetches.

    zarr reads the knob at call time, which is what makes
    `NgioConfig.zarr.async_concurrency` worth exposing: this pins both the
    default ceiling and that lowering it actually narrows the fan-out.
    """
    # `kind="chunk"`: the read's serial metadata prefix must not burn the
    # rendezvous before the chunk fan-out arrives.
    image = open_image(rendezvous_store(image_target), path="0", mode="r")
    with concurrency_probe(rendezvous=10, kind="chunk") as probe:
        image.get_as_numpy()
    assert probe.arrivals > 0
    assert probe.max_in_flight == 10

    image = open_image(rendezvous_store(image_target), path="0", mode="r")
    with (
        zarr.config.set({"async.concurrency": 3}),
        concurrency_probe(rendezvous=3, kind="chunk") as probe,
    ):
        image.get_as_numpy()
    assert probe.arrivals > 0
    assert probe.max_in_flight == 3
