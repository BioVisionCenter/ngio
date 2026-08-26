"""Behavior locks for the masked io pipes.

These tests pin the exact pixel-level behavior of the masked pipe shells,
which must match a bare pipe with a terminal `MaskTransform` until their
removal in ngio=1.2. The expectations are built only from stable primitives:
the bare pipes, `BaseZoomTransform`, and numpy.
"""

import dask.array as da
import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes import (
    DaskGetterMasked,
    DaskSetterMasked,
    NumpyGetter,
    NumpyGetterMasked,
    NumpyRoiGetter,
    NumpySetterMasked,
)
from ngio.io_pipes._zoom_transform import BaseZoomTransform
from ngio.utils import NgioValueError

pytestmark = pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")


class PlusN:
    """Invertible transform recording the order it is applied in."""

    def __init__(self, n: int, calls: list[str]):
        self.n = n
        self.calls = calls

    def on_get(self, array, ctx):
        self.calls.append("get")
        return array + self.n

    def on_set(self, array, ctx):
        self.calls.append("set")
        return array - self.n


def _make_same_res_setup(seed: int = 0):
    """Image and label at the same resolution, labels 3 and 5 inside the roi."""
    rng = np.random.default_rng(seed)
    data = rng.integers(10, 200, size=(32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label_img = np.zeros((32, 32), dtype="uint32")
    label_img[8:20, 10:24] = 3
    label_img[6:8, 8:12] = 5
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    return image, label, data, label_img


def _make_coarse_label_setup(seed: int = 1):
    """Image at pixel size 1.0, label at pixel size 2.0 (coarser level)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(10, 200, size=(32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=2, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label = ome_zarr.derive_label("lbl", ref_image=ome_zarr.get_image(path="1"))
    label_img = np.zeros((16, 16), dtype="uint32")
    label_img[4:10, 5:12] = 7
    label.set_array(label_img)
    return image, label, data


def _masked_kwargs(image, label, roi):
    return {
        "zarr_array": image.zarr_array,
        "dimensions": image.dimensions,
        "label_zarr_array": label.zarr_array,
        "label_dimensions": label.dimensions,
        "roi": roi,
    }


def _get_masked(mode: str, **kwargs) -> np.ndarray:
    if mode == "numpy":
        return NumpyGetterMasked(**kwargs)()
    result = DaskGetterMasked(**kwargs)()
    assert isinstance(result, da.Array)
    return np.asarray(result.compute())


def _set_masked(mode: str, patch: np.ndarray, **kwargs) -> None:
    if mode == "numpy":
        NumpySetterMasked(**kwargs)(patch)
    else:
        DaskSetterMasked(**kwargs)(da.from_array(patch, chunks=-1))


def _read_raw(image) -> np.ndarray:
    return NumpyGetter(zarr_array=image.zarr_array, dimensions=image.dimensions)()


def _zoomed_label_roi(label, image, roi) -> np.ndarray:
    """Read the label roi zoomed to the image grid, via stable primitives."""
    zoom = BaseZoomTransform(
        input_dimensions=label.dimensions,
        target_dimensions=image.dimensions,
        order="nearest",
    )
    getter = NumpyGetter(
        zarr_array=label.zarr_array,
        dimensions=label.dimensions,
        transforms=[zoom],
        slicing_dict=roi.to_slicing_dict(pixel_size=label.dimensions.pixel_size),
        remove_channel_selection=True,
    )
    return getter()


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_get_same_resolution(mode: str):
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)

    result = _get_masked(mode, **_masked_kwargs(image, label, roi))
    mask = label_img[6:22, 8:26] == 3
    np.testing.assert_array_equal(result, np.where(mask, data[6:22, 8:26], 0))


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_get_fill_value_applies_on_get_only(mode: str):
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)

    result = _get_masked(mode, fill_value=7, **_masked_kwargs(image, label, roi))
    mask = label_img[6:22, 8:26] == 3
    np.testing.assert_array_equal(result, np.where(mask, data[6:22, 8:26], 7))

    # The set path has no fill_value: outside-mask pixels keep the disk data.
    patch = np.full((16, 18), 60, dtype="uint16")
    _set_masked(mode, patch, **_masked_kwargs(image, label, roi))
    expected = data.copy()
    expected[6:22, 8:26] = np.where(mask, patch, data[6:22, 8:26])
    np.testing.assert_array_equal(_read_raw(image), expected)


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_get_unlabelled_roi_masks_nonzero(mode: str):
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=None)

    result = _get_masked(mode, **_masked_kwargs(image, label, roi))
    mask = label_img[6:22, 8:26] != 0
    # Both labels 3 and 5 intersect the roi, so this differs from any == mask
    assert {3, 5} <= set(np.unique(label_img[6:22, 8:26]))
    np.testing.assert_array_equal(result, np.where(mask, data[6:22, 8:26], 0))


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_set_same_resolution(mode: str):
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    patch = np.arange(16 * 18, dtype="uint16").reshape(16, 18)

    _set_masked(mode, patch, **_masked_kwargs(image, label, roi))

    mask = label_img[6:22, 8:26] == 3
    expected = data.copy()
    expected[6:22, 8:26] = np.where(mask, patch, data[6:22, 8:26])
    np.testing.assert_array_equal(_read_raw(image), expected)


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_get_label_at_coarser_level(mode: str):
    image, label, data = _make_coarse_label_setup()
    roi = Roi.from_values(slices={"y": (4, 20), "x": (6, 20)}, name="r", label=7)

    result = _get_masked(mode, **_masked_kwargs(image, label, roi))
    mask = _zoomed_label_roi(label, image, roi) == 7
    assert mask.any() and not mask.all()
    np.testing.assert_array_equal(result, np.where(mask, data[4:24, 6:26], 0))


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_set_label_at_coarser_level(mode: str):
    image, label, data = _make_coarse_label_setup()
    roi = Roi.from_values(slices={"y": (4, 20), "x": (6, 20)}, name="r", label=7)
    patch = np.arange(20 * 20, dtype="uint16").reshape(20, 20)

    _set_masked(mode, patch, **_masked_kwargs(image, label, roi))

    mask = _zoomed_label_roi(label, image, roi) == 7
    expected = data.copy()
    expected[4:24, 6:26] = np.where(mask, patch, data[4:24, 6:26])
    np.testing.assert_array_equal(_read_raw(image), expected)


def test_masked_get_with_user_transform():
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    calls: list[str] = []

    result = NumpyGetterMasked(
        transforms=[PlusN(5, calls)], **_masked_kwargs(image, label, roi)
    )()
    # The user transform applies to the data before masking; the fill does
    # not go through the transform.
    mask = label_img[6:22, 8:26] == 3
    np.testing.assert_array_equal(result, np.where(mask, data[6:22, 8:26] + 5, 0))
    assert calls == ["get"]


def test_masked_set_with_user_transform():
    image, label, data, label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    calls: list[str] = []
    patch = np.full((16, 18), 60, dtype="uint16")

    NumpySetterMasked(
        transforms=[PlusN(5, calls)], **_masked_kwargs(image, label, roi)
    )(patch)
    # The inverse transform applies to the patch, then the mask merge picks
    # between it and the raw on-disk data: inside the mask the written pixels
    # are patch - 5, outside they are unchanged.
    mask = label_img[6:22, 8:26] == 3
    expected = data.copy()
    expected[6:22, 8:26] = np.where(mask, patch - 5, data[6:22, 8:26])
    np.testing.assert_array_equal(_read_raw(image), expected)
    # `on_get` is *not* called on the write path. The merge runs after the
    # chain against raw disk values, so the protected pixels are never sent
    # through the transform and back — which is what keeps them exact.
    assert calls == ["set"]


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_masked_get_multichannel_broadcast(mode: str):
    rng = np.random.default_rng(2)
    data = rng.integers(10, 200, size=(2, 32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="cyx"
    )
    image = ome_zarr.get_image(path="0")
    label_img = np.zeros((32, 32), dtype="uint32")
    label_img[8:20, 10:24] = 3
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    mask = label_img[6:22, 8:26] == 3

    # Channel-free label mask broadcasts across the channel axis
    result = _get_masked(mode, **_masked_kwargs(image, label, roi))
    np.testing.assert_array_equal(
        result, np.where(mask[None, :, :], data[:, 6:22, 8:26], 0)
    )

    # Integer channel selection squeezes the axis before masking
    result = _get_masked(
        mode, slicing_dict={"c": 0}, **_masked_kwargs(image, label, roi)
    )
    np.testing.assert_array_equal(result, np.where(mask, data[0, 6:22, 8:26], 0))


def test_masked_slicing_dict_override_refuses():
    """Overriding a roi-pinned axis drops the roi, and masking needs it.

    The old silent path was only correct when the label shared the data's
    resolution — a pixel override lands on the wrong grid otherwise.
    """
    image, label, _data, _label_img = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)

    masked_getter = NumpyGetterMasked(
        slicing_dict={"x": slice(2, 10)},
        label_slicing_dict={"x": slice(2, 10)},
        **_masked_kwargs(image, label, roi),
    )
    with pytest.raises(NgioValueError, match="Masking requires a ROI-scoped pipe"):
        masked_getter()


def test_masked_pipe_properties():
    image, label, _, _ = _make_same_res_setup()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)

    masked_getter = NumpyGetterMasked(**_masked_kwargs(image, label, roi))
    assert masked_getter.label_id == 3
    assert masked_getter.roi is roi

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    assert masked_getter.slicing_ops == roi_getter.slicing_ops
