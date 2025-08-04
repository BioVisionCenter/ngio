from pathlib import Path

import pytest
import zarr

from ngio.common._pyramid import on_disk_zoom


@pytest.mark.parametrize(
    "order, mode",
    [
        (0, "dask"),
        (1, "dask"),
        (0, "numpy"),
        (1, "numpy"),
        (0, "coarsen"),
        (1, "coarsen"),
    ],
)
def test_on_disk_zooms(tmp_path: Path, order: int, mode: str):
    source = tmp_path / "source.zarr"
    source_array = zarr.create_array(source, shape=(16, 128, 128), dtype="uint8")

    target = tmp_path / "target.zarr"
    target_array = zarr.create_array(target, shape=(16, 64, 64), dtype="uint8")

    on_disk_zoom(source_array, target_array, order=order, mode=mode)
