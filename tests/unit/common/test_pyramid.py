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


@pytest.mark.parametrize("mode", ["dask", "coarsen"])
def test_on_disk_zoom_sharded_matches_unsharded(
    tmp_path: Path, mode: Literal["dask", "coarsen"]
):
    """A zoom onto a sharded target matches the same zoom onto an unsharded one.

    Both dask paths rechunk to `target.chunks`, which for a sharded array is the
    *inner* chunk shape, while writes are atomic per shard object. Several blocks
    therefore read-modify-write one shard and race without the shared `da.store`
    lock. The unsharded layout is race-free by alignment, so it is the reference.
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
