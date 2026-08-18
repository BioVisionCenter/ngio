"""Tests for the public `MaskTransform` composed onto ordinary roi io."""

import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes import NumpyGetterMasked
from ngio.io_pipes._mask_transform import BaseMaskTransform
from ngio.transforms import MaskTransform
from ngio.utils import NgioValueError


class PlusN:
    def __init__(self, n: int):
        self.n = n

    def on_get(self, array, ctx):
        return array + self.n

    def on_set(self, array, ctx):
        return array - self.n


def _make_setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.integers(10, 200, size=(32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label_img = np.zeros((32, 32), dtype="uint32")
    label_img[8:20, 10:24] = 3
    label_img[2:6, 2:8] = 5
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    return image, label, data, label_img


def test_mask_transform_get_matches_masked_pipe():
    image, label, data, label_img = _make_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    mask = MaskTransform(label=label, target_image=image)

    via_transform = image.get_roi_as_numpy(roi=roi, transforms=[mask])
    via_masked_pipe = NumpyGetterMasked(
        zarr_array=image.zarr_array,
        dimensions=image.dimensions,
        label_zarr_array=label.zarr_array,
        label_dimensions=label.dimensions,
        roi=roi,
    )()
    np.testing.assert_array_equal(via_transform, via_masked_pipe)
    np.testing.assert_array_equal(
        via_transform,
        np.where(label_img[6:22, 8:26] == 3, data[6:22, 8:26], 0),
    )


def test_mask_transform_get_dask_parity():
    image, label, data, label_img = _make_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    mask = MaskTransform(label=label, target_image=image)

    via_dask = image.get_roi_as_dask(roi=roi, transforms=[mask]).compute()
    np.testing.assert_array_equal(
        via_dask, np.where(label_img[6:22, 8:26] == 3, data[6:22, 8:26], 0)
    )


def test_mask_transform_instance_reusable_across_rois():
    image, label, data, label_img = _make_setup()
    mask = MaskTransform(label=label, target_image=image)

    roi_a = Roi.from_values(slices={"y": (8, 8), "x": (10, 8)}, name="a", label=3)
    roi_b = Roi.from_values(slices={"y": (2, 4), "x": (2, 6)}, name="b", label=5)

    result_a = image.get_roi_as_numpy(roi=roi_a, transforms=[mask])
    result_b = image.get_roi_as_numpy(roi=roi_b, transforms=[mask])
    np.testing.assert_array_equal(
        result_a, np.where(label_img[8:16, 10:18] == 3, data[8:16, 10:18], 0)
    )
    np.testing.assert_array_equal(
        result_b, np.where(label_img[2:6, 2:8] == 5, data[2:6, 2:8], 0)
    )


def test_mask_transform_set_merges_with_disk_data():
    image, label, data, label_img = _make_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    mask = MaskTransform(label=label, target_image=image)
    patch = np.full((16, 18), 250, dtype="uint16")

    image.set_roi(roi=roi, patch=patch, transforms=[mask])

    bool_mask = label_img[6:22, 8:26] == 3
    expected = data.copy()
    expected[6:22, 8:26] = np.where(bool_mask, patch, data[6:22, 8:26])
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_mask_transform_set_with_prefix_transforms():
    image, label, data, label_img = _make_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    plus = PlusN(5)
    mask = MaskTransform(label=label, target_image=image, set_transforms=[plus])
    patch = np.full((16, 18), 60, dtype="uint16")

    image.set_roi(roi=roi, patch=patch, transforms=[plus, mask])

    # Inside the mask the patch lands minus the inverse transform; outside
    # the disk data is untouched.
    bool_mask = label_img[6:22, 8:26] == 3
    expected = data.copy()
    expected[6:22, 8:26] = np.where(bool_mask, patch - 5, data[6:22, 8:26])
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_mask_transform_requires_roi():
    image, label, _, _ = _make_setup()
    mask = MaskTransform(label=label, target_image=image)

    with pytest.raises(NgioValueError, match="ROI"):
        image.get_as_numpy(transforms=[mask])


def test_mask_transform_must_be_terminal():
    image, label, _, _ = _make_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    mask = MaskTransform(label=label, target_image=image)

    with pytest.raises(NgioValueError, match="last transform"):
        image.get_roi_as_numpy(roi=roi, transforms=[mask, PlusN(5)])
    # Terminal position is fine
    image.get_roi_as_numpy(roi=roi, transforms=[PlusN(5), mask])


def test_mask_transform_rescaling_requires_target_dimensions():
    image, label, _, _ = _make_setup()

    with pytest.raises(NgioValueError, match="target_dimensions"):
        BaseMaskTransform(
            label_zarr_array=label.zarr_array,
            label_dimensions=label.dimensions,
        )
    # Without rescaling no target is needed
    MaskTransform(label=label, allow_rescaling=False)
