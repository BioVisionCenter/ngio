"""Every deprecated pipe shell warns, and still behaves like its replacement."""

import numpy as np
import pytest

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes import (
    DaskGetterMasked,
    DaskRoiGetter,
    DaskRoiSetter,
    DaskSetterMasked,
    NumpyGetterMasked,
    NumpyRoiGetter,
    NumpyRoiSetter,
    NumpySetterMasked,
)
from ngio.utils import NgioDeprecationWarning


def _make_setup():
    rng = np.random.default_rng(0)
    data = rng.integers(10, 200, size=(32, 32)).astype("uint16")
    ome_zarr = create_ome_zarr_from_array(
        store={}, array=data, pixelsize=1.0, levels=1, axes_names="yx"
    )
    image = ome_zarr.get_image(path="0")
    label_img = np.zeros((32, 32), dtype="uint32")
    label_img[8:20, 10:24] = 3
    label = ome_zarr.derive_label("lbl")
    label.set_array(label_img)
    roi = Roi.from_values(slices={"y": (6, 16), "x": (8, 18)}, name="r", label=3)
    return image, label, roi


@pytest.mark.parametrize(
    "pipe_cls", [NumpyRoiGetter, DaskRoiGetter, NumpyRoiSetter, DaskRoiSetter]
)
def test_roi_pipe_shells_warn(pipe_cls):
    image, _, roi = _make_setup()
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        pipe_cls(zarr_array=image.zarr_array, dimensions=image.dimensions, roi=roi)


@pytest.mark.parametrize(
    "pipe_cls",
    [NumpyGetterMasked, DaskGetterMasked, NumpySetterMasked, DaskSetterMasked],
)
def test_masked_pipe_shells_warn(pipe_cls):
    image, label, roi = _make_setup()
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        pipe_cls(
            zarr_array=image.zarr_array,
            dimensions=image.dimensions,
            label_zarr_array=label.zarr_array,
            label_dimensions=label.dimensions,
            roi=roi,
        )


def test_shells_warn_but_keep_working():
    image, label, roi = _make_setup()
    with pytest.warns(NgioDeprecationWarning, match="NumpyGetterMasked"):
        getter = NumpyGetterMasked(
            zarr_array=image.zarr_array,
            dimensions=image.dimensions,
            label_zarr_array=label.zarr_array,
            label_dimensions=label.dimensions,
            roi=roi,
        )
    result = getter()
    assert result.shape == (16, 18)
    assert getter.label_id == 3
