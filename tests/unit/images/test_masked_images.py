from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from skimage.segmentation import watershed

from ngio import create_ome_zarr_from_array


def _draw_random_labels(shape: tuple[int, ...], num_regions: int):
    np.random.seed(0)
    markers = np.zeros(shape, dtype=np.int32)
    seeds_list = np.random.randint(0, shape[0], size=(num_regions, 2))
    for i, (y, x) in enumerate(seeds_list, start=1):
        markers[y, x] = i

    image = ndimage.distance_transform_edt(markers == 0).astype("uint16")
    labels = watershed(image, markers)
    return image, labels

@pytest.mark.skip
@pytest.mark.parametrize(
    "array_mode, shape",
    [("numpy", (64, 64)), ("dask", (64, 64)), ("numpy", (16, 32, 32))],
)
def test_masking(tmp_path: Path, array_mode: str, shape: tuple[int, ...]):
    mask, label_image = _draw_random_labels(shape=shape, num_regions=20)
    unique_labels, counts = np.unique(label_image, return_counts=True)
    labels_stats = dict(zip(unique_labels, counts, strict=True))

    store = tmp_path / "test_image_yx_random_label.zarr"
    # Create a new ome_zarr with the mask
    ome_zarr = create_ome_zarr_from_array(
        store=store,
        array=mask,
        xy_pixelsize=0.5,
        levels=1,
        overwrite=True,
    )
    label = ome_zarr.derive_label("label")
    label.set_array(label_image)

    # Masking image test
    masked_image = ome_zarr.get_masked_image("label")
    assert isinstance(masked_image.__repr__(), str)
    _roi_array = masked_image.get_roi(label=1, zoom_factor=1.123, mode=array_mode)
    masked_image.set_roi_masked(
        label=1, patch=np.ones_like(_roi_array), zoom_factor=1.123
    )

    _roi_mask = masked_image.get_roi_masked(label=1, mode=array_mode)
    # Check that the mask is binary after masking
    np.testing.assert_allclose(np.unique(_roi_mask), [0, 1])

    # Just test the API
    masked_image.set_roi(label=1, patch=np.zeros_like(_roi_array), zoom_factor=1.123)

    # Masking label test (recreate the label)
    ome_zarr.derive_label("empty_label")
    masked_new_label = ome_zarr.get_masked_label(
        "empty_label", masking_label_name="label"
    )
    assert isinstance(masked_new_label.__repr__(), str)

    for label_id in labels_stats.keys():
        label_mask = masked_new_label.get_roi(label_id, mode=array_mode)
        label_mask = np.full(label_mask.shape, label_id, dtype=label_mask.dtype)
        # Set the label only inside the mask
        masked_new_label.set_roi_masked(label_id, label_mask)

    # rerun the stats on the new masked label
    unique_labels, counts = np.unique(masked_new_label.get_array(), return_counts=True)
    labels_stats_masked = dict(zip(unique_labels, counts, strict=True))
    assert labels_stats == labels_stats_masked
