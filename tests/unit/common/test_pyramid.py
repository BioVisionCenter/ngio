from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr

from ngio.common._pyramid import InterpolationOrder, on_disk_zoom


@pytest.mark.parametrize(
    "order, mode",
    [
        ("nearest", "dask"),
        ("linear", "dask"),
        ("nearest", "numpy"),
        ("linear", "numpy"),
        ("nearest", "coarsen"),
        ("linear", "coarsen"),
    ],
)
def test_on_disk_zooms(
    tmp_path: Path, order: InterpolationOrder, mode: Literal["dask", "numpy", "coarsen"]
):
    source = tmp_path / "source.zarr"
    source_array = zarr.create_array(source, shape=(16, 128, 128), dtype="uint8")

    target = tmp_path / "target.zarr"
    target_array = zarr.create_array(target, shape=(16, 64, 64), dtype="uint8")

    on_disk_zoom(source_array, target_array, order=order, mode=mode)


def test_coarsen_nearest_takes_the_max(tmp_path: Path):
    """`order="nearest"` must take the max over each block, never the mean.

    `Label.consolidate()` passes `order="nearest"` precisely so a label pyramid
    keeps IDs that were really segmented. Averaging invents ones that were not —
    the mean of labels 3 and 7 is 5 — and truncates on an integer dtype, so this
    asserts the blockwise max exactly rather than only that the call returns.
    """
    labels = np.array(
        [
            [3, 3, 0, 0],
            [3, 7, 0, 9],
            [4, 4, 8, 8],
            [4, 4, 8, 2],
        ],
        dtype="uint16",
    )
    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(1, 4, 4), dtype="uint16"
    )
    source_array[...] = labels[np.newaxis]

    target_array = zarr.create_array(
        tmp_path / "target.zarr", shape=(1, 2, 2), dtype="uint16"
    )
    on_disk_zoom(source_array, target_array, order="nearest", mode="coarsen")

    # Blockwise max. The mean would give [[4, 2], [4, 6]] — three of the four
    # cells wrong, and 6 is a label that appears nowhere in the source.
    np.testing.assert_array_equal(
        target_array[...], np.array([[[7, 9], [4, 8]]], dtype="uint16")
    )
    assert set(np.unique(target_array[...])) <= set(np.unique(labels))


@pytest.mark.parametrize("mode", ["dask", "coarsen"])
def test_on_disk_zoom_sharded_matches_unsharded(
    tmp_path: Path, mode: Literal["dask", "coarsen"]
):
    """A zoom onto a sharded target matches the same zoom onto an unsharded one.

    Writes are atomic per shard object, so a block covering only part of a shard
    makes zarr read-modify-write the whole thing — and several such blocks race.
    Both dask paths hand the zoomed array to `store_dask`, which rechunks onto
    the target's write unit (`shards or chunks`) so each shard has exactly one
    writer. The unsharded layout is race-free by alignment, so it is the
    reference the sharded result has to match.
    """
    rng = np.random.default_rng(0)
    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(16, 128, 128), dtype="uint8"
    )
    source_array[...] = rng.integers(0, 255, size=(16, 128, 128), dtype="uint8")

    results = {}
    for name, shards in (("unsharded", None), ("sharded", (8, 32, 32))):
        target_array = zarr.create_array(
            tmp_path / f"target_{mode}_{name}.zarr",
            shape=(16, 64, 64),
            chunks=(4, 16, 16),
            shards=shards,
            dtype="uint8",
        )
        on_disk_zoom(source_array, target_array, order="nearest", mode=mode)
        results[name] = target_array[...]

    np.testing.assert_array_equal(results["sharded"], results["unsharded"])
