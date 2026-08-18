"""Behavior locks for the ROI io pipes.

These tests pin the exact observable behavior of the ROI pipe shells, which
must match the bare pipes' roi kwarg until their removal in ngio=1.2.
"""

import dask.array as da
import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes import (
    DaskGetter,
    DaskRoiGetter,
    DaskRoiSetter,
    NumpyGetter,
    NumpyRoiGetter,
    NumpyRoiSetter,
    SlicingInputType,
)

pytestmark = pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")


def _make_image(pixelsize: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.integers(10, 200, size=(32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=pixelsize, levels=1, axes_names="yx"
    )
    return ome_zarr.get_image(path="0"), data


def test_roi_getter_equals_bare_getter_with_slices():
    image, data = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r")

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    roi_slices: dict[str, SlicingInputType] = {
        **roi.to_slicing_dict(pixel_size=image.dimensions.pixel_size)
    }
    bare_getter = NumpyGetter(
        zarr_array=image.zarr_array,
        dimensions=image.dimensions,
        slicing_dict=roi_slices,
    )
    assert roi_getter.slicing_ops == bare_getter.slicing_ops
    np.testing.assert_array_equal(roi_getter(), data[6:22, 8:26])
    np.testing.assert_array_equal(roi_getter(), bare_getter())

    dask_roi_getter = DaskRoiGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    assert dask_roi_getter.slicing_ops == roi_getter.slicing_ops
    np.testing.assert_array_equal(dask_roi_getter().compute(), data[6:22, 8:26])


def test_roi_world_to_pixel_conversion():
    image, data = _make_image(pixelsize=0.5)
    # World coords at pixel size 0.5: y [2, 10) -> pixels [4, 20)
    roi = Roi.from_values(slices={"y": (2, 8), "x": (3, 8)}, name="r")

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    np.testing.assert_array_equal(roi_getter(), data[4:20, 6:22])


def test_explicit_slicing_dict_overrides_roi_axis():
    image, data = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r")

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array,
        dimensions=image.dimensions,
        roi=roi,
        slicing_dict={"x": slice(2, 10)},
    )
    bare_getter = NumpyGetter(
        zarr_array=image.zarr_array,
        dimensions=image.dimensions,
        slicing_dict={"y": slice(6, 22), "x": slice(2, 10)},
    )
    assert roi_getter.slicing_ops == bare_getter.slicing_ops
    np.testing.assert_array_equal(roi_getter(), data[6:22, 2:10])


def test_roi_getter_axes_order_passthrough():
    image, data = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r")

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array,
        dimensions=image.dimensions,
        roi=roi,
        axes_order=["x", "y"],
    )
    np.testing.assert_array_equal(roi_getter(), data[6:22, 8:26].T)


def test_bare_getter_roi_kwarg_is_slicing_active():
    # New in 1.1: passing roi to a bare pipe slices by it, making the Roi
    # pipe classes redundant.
    image, data = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r")

    getter = NumpyGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    np.testing.assert_array_equal(getter(), data[6:22, 8:26])
    assert getter.roi is roi


def test_roi_getter_exposes_roi():
    image, _ = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)

    roi_getter = NumpyRoiGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
    )
    assert roi_getter.roi is roi

    bare_getter = NumpyGetter(zarr_array=image.zarr_array, dimensions=image.dimensions)
    with pytest.raises(ValueError, match="No ROI"):
        _ = bare_getter.roi


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_roi_setter_round_trip(mode: str):
    image, data = _make_image()
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r")
    patch = np.arange(16 * 18, dtype="uint16").reshape(16, 18)

    if mode == "numpy":
        setter = NumpyRoiSetter(
            zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
        )
        setter(patch)
    else:
        setter = DaskRoiSetter(
            zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi
        )
        setter(da.from_array(patch, chunks=-1))

    expected = data.copy()
    expected[6:22, 8:26] = patch
    read_back = DaskGetter(
        zarr_array=image.zarr_array, dimensions=image.dimensions
    )().compute()
    np.testing.assert_array_equal(read_back, expected)
