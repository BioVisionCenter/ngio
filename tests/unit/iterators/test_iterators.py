from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from ngio import open_ome_zarr_container
from ngio.experimental.iterators import SegmentationIterator


@pytest.mark.parametrize(
    "zarr_name",
    [
        "test_image_yx.zarr",
        "test_image_cyx.zarr",
        "test_image_zyx.zarr",
        "test_image_czyx.zarr",
        "test_image_c1yx.zarr",
        "test_image_tyx.zarr",
        "test_image_tcyx.zarr",
        "test_image_tzyx.zarr",
        "test_image_tczyx.zarr",
    ],
)
def test_segmentation_iterator(images_v04: dict[str, Path], zarr_name: str):
    # Base test only the API, not the actual segmentation logic
    path = images_v04[zarr_name]
    ome_zarr = open_ome_zarr_container(path)
    image = ome_zarr.get_image()
    label = ome_zarr.get_label("label")
    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    iterator = iterator.by_yx()
    for i, (img_chunk, writer) in enumerate(iterator.iter_as_numpy()):
        label_patch = np.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_numpy(lambda x: np.zeros_like(x, dtype=np.uint8))

    for i, (img_chunk, writer) in enumerate(iterator.iter_as_dask()):
        label_patch = da.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))
