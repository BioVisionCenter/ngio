from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import open_ome_zarr_container
from ngio.experimental.iterators import (
    FeatureExtractorIterator,
    ImageProcessingIterator,
    MaskedSegmentationIterator,
    SegmentationIterator,
)


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
    iterator = iterator.by_chunks()
    iterator = iterator.by_yx()
    for i, (img_chunk, writer) in enumerate(iterator.iter_as_numpy()):
        label_patch = np.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_numpy(lambda x: np.zeros_like(x, dtype=np.uint8))

    for i, (img_chunk, writer) in enumerate(iterator.iter_as_dask()):
        label_patch = da.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


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
def test_masked_segmentation_iterator(images_v04: dict[str, Path], zarr_name: str):
    # Base test only the API, not the actual segmentation logic
    path = images_v04[zarr_name]
    ome_zarr = open_ome_zarr_container(path)

    masked_label = ome_zarr.derive_label("masking_label")
    masked_label.set_array(np.ones(shape=masked_label.shape, dtype=np.uint8))
    masked_label.consolidate()

    ome_zarr.add_table(
        "masking_label_ROI_table", masked_label.build_masking_roi_table()
    )

    image = ome_zarr.get_masked_image(masking_label_name="masking_label")
    label = ome_zarr.get_label("label")

    iterator = MaskedSegmentationIterator(
        image, label, channel_selection=0, axes_order="yx"
    )

    iterator = iterator.by_yx()
    for i, (img_chunk, writer) in enumerate(iterator.iter_as_numpy()):
        label_patch = np.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_numpy(lambda x: np.zeros_like(x, dtype=np.uint8))

    for i, (img_chunk, writer) in enumerate(iterator.iter_as_dask()):
        label_patch = da.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


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
def test_img_processing_iterator(images_v04: dict[str, Path], zarr_name: str):
    # Base test only the API, not the actual segmentation logic
    path = images_v04[zarr_name]
    ome_zarr = open_ome_zarr_container(path)
    image = ome_zarr.get_image()
    t_ome_zarr = ome_zarr.derive_image(store=MemoryStore())
    t_image = t_ome_zarr.get_image()

    iterator = ImageProcessingIterator(input_image=image, output_image=t_image)
    iterator = iterator.by_zyx(strict=False)
    for img_chunk, writer in iterator.iter_as_numpy():
        label_patch = np.zeros_like(img_chunk, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_numpy(lambda x: np.zeros_like(x, dtype=np.uint8))

    for img_chunk, writer in iterator.iter_as_dask():
        label_patch = da.zeros_like(img_chunk, dtype=np.uint8)
        writer(label_patch)

    iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


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
def test_features_iterator(images_v04: dict[str, Path], zarr_name: str):
    # Base test only the API, not the actual segmentation logic
    path = images_v04[zarr_name]
    ome_zarr = open_ome_zarr_container(path)

    image = ome_zarr.get_image()
    label = ome_zarr.get_label(name="label")

    feat_iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label,
        channel_selection=0,
        axes_order="xy",
    )
    feat_iterator = feat_iterator.by_yx()
    for data, seg, _ in feat_iterator.iter_as_numpy():
        assert data.shape == seg.shape
