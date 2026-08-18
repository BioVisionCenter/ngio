"""Tests for the `merge=` slot: how a write combines with what is on disk."""

import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes import MergeRule
from ngio.transforms import MaskMerge, MaskTransform
from ngio.utils import NgioValueError


class PlusN:
    """A plain transform, to sit in the chain alongside a merge."""

    def __init__(self, n: int):
        self.n = n

    def on_get(self, array, ctx):
        return array + self.n

    def on_set(self, array, ctx):
        return array - self.n


def _image(fill: int = 100, shape: tuple[int, int] = (16, 16)):
    data = np.full(shape, fill, dtype="uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    return ome_zarr.get_image(path="0"), data


def _labelled_image():
    data = np.full((16, 16), 100, dtype="uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label_img = np.zeros((16, 16), dtype="uint32")
    label_img[4:12, 4:12] = 3
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    return image, label, data, label_img


ROI = Roi.from_values(slices={"y": (4, 8), "x": (4, 8)}, name="r", label=3)


def test_merge_reads_the_destination():
    """The merge sees what is on disk, not a zero-filled placeholder."""
    image, data = _image(fill=100)
    patch = np.full((8, 8), 50, dtype="uint16")

    image.set_roi(roi=ROI, patch=patch, merge="max")

    expected = data.copy()
    expected[4:12, 4:12] = 100  # max(100, 50) -- the patch loses everywhere
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("max", 100), ("min", 50), ("sum", 150), ("keep_nonzero", 50)],
)
def test_builtin_merge_rules(rule: MergeRule, expected: int):
    image, data = _image(fill=100)
    patch = np.full((8, 8), 50, dtype="uint16")

    image.set_roi(roi=ROI, patch=patch, merge=rule)

    want = data.copy()
    want[4:12, 4:12] = expected
    np.testing.assert_array_equal(image.get_as_numpy(), want)


def test_keep_nonzero_lets_background_through():
    image, data = _image(fill=100)
    patch = np.zeros((8, 8), dtype="uint16")
    patch[0, 0] = 7

    image.set_roi(roi=ROI, patch=patch, merge="keep_nonzero")

    expected = data.copy()
    expected[4, 4] = 7
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_custom_merge_callable():
    image, data = _image(fill=100)
    patch = np.full((8, 8), 50, dtype="uint16")

    def half(existing, patch, ctx):
        return ((existing + patch) // 2).astype(existing.dtype)

    image.set_roi(roi=ROI, patch=patch, merge=half)

    expected = data.copy()
    expected[4:12, 4:12] = 75
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_merge_does_not_require_a_roi():
    """Unlike a mask, a plain merge has nothing to select and works on set_array."""
    image, _ = _image(fill=100)
    patch = np.full((16, 16), 50, dtype="uint16")

    image.set_array(patch=patch, merge="max")

    np.testing.assert_array_equal(
        image.get_as_numpy(), np.full((16, 16), 100, dtype="uint16")
    )


def test_merge_dask_parity():
    import dask.array as da

    image, data = _image(fill=100)
    patch = da.full((8, 8), 50, dtype="uint16", chunks=(4, 4))

    image.set_roi(roi=ROI, patch=patch, merge="max")

    expected = data.copy()
    expected[4:12, 4:12] = 100
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_unknown_merge_rule_raises():
    image, _ = _image()
    with pytest.raises(NgioValueError, match="Unknown merge rule"):
        image.set_roi(
            roi=ROI,
            patch=np.zeros((8, 8), dtype="uint16"),
            merge="average",
        )


def test_merge_runs_after_the_transform_chain():
    """The merge sees disk-space arrays, so the comparison is in disk space.

    With `PlusN(5)`, a patch of 102 inverts to 97 before the merge; `max(100,
    97)` keeps the on-disk 100. If the merge ran in user space instead it would
    compare 105 against 102 and write 100 as well — so the discriminating case
    is the one below, where the two disagree.
    """
    image, data = _image(fill=100)
    patch = np.full((8, 8), 102, dtype="uint16")

    image.set_roi(roi=ROI, patch=patch, transforms=[PlusN(5)], merge="max")

    expected = data.copy()
    expected[4:12, 4:12] = 100
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_merge_policy_in_transforms_is_refused():
    """A merge policy is not a transform, and the error says where it goes."""
    image, label, _, _ = _labelled_image()
    merge = MaskMerge(label=label, target_image=image)

    with pytest.raises(NgioValueError, match="merge policy, not a transform"):
        image.get_roi_as_numpy(roi=ROI, transforms=[merge])


def test_mask_transform_on_a_write_is_refused():
    """Filling on read has no inverse; protecting on write is a merge."""
    image, label, _, _ = _labelled_image()
    mask = MaskTransform(label=label, target_image=image)

    with pytest.raises(NgioValueError, match="merge=MaskMerge"):
        image.set_roi(
            roi=ROI, patch=np.zeros((8, 8), dtype="uint16"), transforms=[mask]
        )


def test_mask_merge_protects_outside_pixels():
    image, label, data, label_img = _labelled_image()
    patch = np.full((8, 8), 250, dtype="uint16")

    image.set_roi(
        roi=ROI, patch=patch, merge=MaskMerge(label=label, target_image=image)
    )

    bool_mask = label_img[4:12, 4:12] == 3
    expected = data.copy()
    expected[4:12, 4:12] = np.where(bool_mask, patch, data[4:12, 4:12])
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_mask_merge_leaves_outside_pixels_byte_identical():
    """The exactness the old user-space merge could not promise.

    A lossy transform in the chain used to round-trip the protected pixels
    through its inverse; merging in disk space carries them through untouched.
    """
    image, label, data, label_img = _labelled_image()

    class Lossy:
        """Integer-divides on read and multiplies back, losing the remainder."""

        def on_get(self, array, ctx):
            return array // 7

        def on_set(self, array, ctx):
            return (array * 7).astype("uint16")

    image.set_roi(
        roi=ROI,
        patch=np.full((8, 8), 3, dtype="uint16"),
        transforms=[Lossy()],
        merge=MaskMerge(label=label, target_image=image),
    )

    outside = label_img[4:12, 4:12] != 3
    written = image.get_as_numpy()[4:12, 4:12]
    np.testing.assert_array_equal(
        written[outside], data[4:12, 4:12][outside], "protected pixels were resampled"
    )
