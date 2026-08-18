"""Tests for the read-modify-write transform base and `MergeTransform`."""

import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.transforms import MaskTransform, MergeTransform
from ngio.transforms._merge import MergeRule
from ngio.utils import NgioValueError


class PlusN:
    """A plain (non read-modify-write) transform, to sit before a terminal one."""

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
    image, data = _image()
    label_img = np.zeros((16, 16), dtype="uint32")
    label_img[4:12, 4:12] = 3
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    return image, label, data, label_img


ROI = Roi.from_values(slices={"y": (4, 8), "x": (4, 8)}, name="r", label=3)


def test_merge_reads_the_destination_on_set():
    """`on_set` sees what is on disk, not a zero-filled placeholder."""
    image, data = _image(fill=100)
    patch = np.full((8, 8), 50, dtype="uint16")

    image.set_roi(roi=ROI, patch=patch, transforms=[MergeTransform("max")])

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

    image.set_roi(roi=ROI, patch=patch, transforms=[MergeTransform(rule)])

    want = data.copy()
    want[4:12, 4:12] = expected
    np.testing.assert_array_equal(image.get_as_numpy(), want)


def test_keep_nonzero_lets_background_through():
    """A zero in the patch keeps the on-disk value; a nonzero overwrites it."""
    image, data = _image(fill=100)
    patch = np.zeros((8, 8), dtype="uint16")
    patch[0, 0] = 7

    image.set_roi(roi=ROI, patch=patch, transforms=[MergeTransform("keep_nonzero")])

    expected = data.copy()
    expected[4, 4] = 7
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_custom_merge_callable():
    image, data = _image(fill=100)
    patch = np.full((8, 8), 50, dtype="uint16")

    def half(existing, patch, ctx):
        return ((existing + patch) // 2).astype(existing.dtype)

    image.set_roi(roi=ROI, patch=patch, transforms=[MergeTransform(half)])

    expected = data.copy()
    expected[4:12, 4:12] = 75
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_merge_does_not_require_a_roi():
    """Unlike the mask, a merge has nothing to select and works on `set_array`."""
    image, _ = _image(fill=100)
    patch = np.full((16, 16), 50, dtype="uint16")

    image.set_array(patch=patch, transforms=[MergeTransform("max")])

    np.testing.assert_array_equal(
        image.get_as_numpy(), np.full((16, 16), 100, dtype="uint16")
    )


def test_merge_dask_parity():
    import dask.array as da

    image, data = _image(fill=100)
    patch = da.full((8, 8), 50, dtype="uint16", chunks=(4, 4))

    image.set_roi(roi=ROI, patch=patch, transforms=[MergeTransform("max")])

    expected = data.copy()
    expected[4:12, 4:12] = 100
    np.testing.assert_array_equal(image.get_as_numpy(), expected)


def test_unknown_merge_rule_raises():
    with pytest.raises(NgioValueError, match="Unknown merge rule"):
        MergeTransform("average")  # ty: ignore[invalid-argument-type]


def test_preceding_chain_is_bound_automatically():
    """The pipe replays the preceding chain without being told to.

    This is the regression on the footgun: before auto-wiring, forgetting
    `set_transforms=` made the read-back skip `PlusN` silently.
    """
    auto_image, data = _image(fill=100)
    manual_image, _ = _image(fill=100)
    # 102 is chosen so the two readings disagree: with the prefix replayed the
    # read-back is 105 and the disk keeps 100; without it the read-back is 100,
    # the patch wins, and `PlusN.on_set` writes 97.
    patch = np.full((8, 8), 102, dtype="uint16")

    plus = PlusN(5)
    auto_image.set_roi(roi=ROI, patch=patch, transforms=[plus, MergeTransform("max")])
    manual_image.set_roi(
        roi=ROI,
        patch=patch,
        transforms=[plus, MergeTransform("max", set_transforms=[plus])],
    )

    np.testing.assert_array_equal(
        auto_image.get_as_numpy(), manual_image.get_as_numpy()
    )
    expected = data.copy()
    expected[4:12, 4:12] = 100
    np.testing.assert_array_equal(auto_image.get_as_numpy(), expected)


def test_binding_does_not_leak_between_pipes():
    """One instance reused across chains must not carry the first binding."""
    bound_image, _ = _image(fill=100)
    bare_image, _ = _image(fill=100)
    merge = MergeTransform("max")
    patch = np.full((8, 8), 102, dtype="uint16")

    # Chain with a prefix: read-back is disk+5 == 105, so max(105, 102) - 5 == 100.
    bound_image.set_roi(roi=ROI, patch=patch, transforms=[PlusN(5), merge])
    # The same instance, now with no prefix: max(100, 102) == 102.
    bare_image.set_roi(roi=ROI, patch=patch, transforms=[merge])

    assert bound_image.get_as_numpy()[4, 4] == 100
    assert bare_image.get_as_numpy()[4, 4] == 102


def test_at_most_one_rmw_transform_per_chain():
    image, label, _, _ = _labelled_image()
    mask = MaskTransform(label=label, target_image=image)

    with pytest.raises(NgioValueError, match="at most one"):
        image.get_roi_as_numpy(roi=ROI, transforms=[MergeTransform("max"), mask])


def test_merge_must_be_terminal():
    image, _ = _image()

    with pytest.raises(NgioValueError, match="last transform"):
        image.get_roi_as_numpy(roi=ROI, transforms=[MergeTransform("max"), PlusN(5)])
    # Terminal position is fine.
    image.get_roi_as_numpy(roi=ROI, transforms=[PlusN(5), MergeTransform("max")])


def test_merge_is_identity_on_read():
    image, data = _image(fill=100)
    np.testing.assert_array_equal(
        image.get_roi_as_numpy(roi=ROI, transforms=[MergeTransform("max")]),
        data[4:12, 4:12],
    )
