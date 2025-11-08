from pathlib import Path
from typing import Literal

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
