from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from ngio import Roi, create_empty_ome_zarr
from ngio.transforms import ZoomTransform

# Skip this whole module for now


@pytest.mark.skip
def test_zoom_transform():
    zoom = ZoomTransform(
        scale=(1, 2, 2),
        order="nearest",
        origin_shape=(1, 10, 10),
        dest_shape=(1, 20, 20),
    )
    assert zoom.scale == (1, 2, 2)
    assert zoom.inv_scale == (1.0, 0.5, 0.5)

    x = np.ones((1, 10, 10))
    x_zoomed = zoom.apply_numpy_transform(
        array=x,
        slicing_ops=SlicingOps(),
    )

    assert x_zoomed.shape == (1, 20, 20)

    x_inverse = zoom.apply_inverse_numpy_transform(
        array=x_zoomed,
        slicing_ops=SlicingOps(),
    )
    assert x_inverse.shape == (1, 10, 10)

    x_dask = da.from_array(x)
    x_zoomed_dask = zoom.apply_dask_transform(
        array=x_dask,
        slicing_ops=SlicingOps(),
    )
    assert x_zoomed_dask.shape == (1, 20, 20)
    x_inverse_dask = zoom.apply_inverse_dask_transform(
        array=x_zoomed_dask,
        slicing_ops=SlicingOps(),
    )
    assert x_inverse_dask.shape == (1, 10, 10)


@pytest.mark.skip
def test_zoom_from_dimensions(tmp_path: Path):
    full_res_img = create_empty_ome_zarr(
        store=tmp_path / "original.zarr",
        shape=(101, 101),
        axes_names="yx",
        xy_pixelsize=1.0,
    ).get_image()

    img = create_empty_ome_zarr(
        store=tmp_path / "img.zarr",
        shape=(25, 25),
        axes_names="yx",
        xy_pixelsize=4.0,
    ).get_image()

    zoom = ZoomTransform.from_images(
        origin_image=img,
        destination_image=full_res_img,
        order="nearest",
    )
    assert np.allclose(zoom.scale, (4.0, 4.0))
    roi = Roi(name=None, x=0, y=0, x_length=21, y_length=21)

    full_res_data = full_res_img.get_roi_as_numpy(roi=roi)
    rescaled_data = img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert full_res_data.shape == rescaled_data.shape, "Failed inbound test"
    try:
        img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])
    except Exception as e:
        raise AssertionError(f"Failed inbound test: {e}") from e

    roi = Roi(name=None, x=80, y=80, x_length=21, y_length=21)

    full_res_data = full_res_img.get_roi_as_numpy(roi=roi)
    rescaled_data = img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert full_res_data.shape == rescaled_data.shape, "Failed outbound test"
    try:
        img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])
    except Exception as e:
        raise AssertionError(f"Failed outbound test: {e}") from e
