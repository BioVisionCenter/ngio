from pathlib import Path

from ngio import Roi, create_empty_ome_zarr
from ngio.transforms import ZoomTransform


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

    zoom = ZoomTransform(
        input_image=img,
        target_image=full_res_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 21), "y": (0, 21)})

    full_res_data = full_res_img.get_roi_as_numpy(roi=roi)
    rescaled_data = img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert full_res_data.shape == rescaled_data.shape, "Failed inbound test"
    try:
        img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])
    except Exception as e:
        raise AssertionError(f"Failed inbound test: {e}") from e

    roi = Roi.from_values(name=None, slices={"x": (80, 21), "y": (80, 21)})

    full_res_data = full_res_img.get_roi_as_numpy(roi=roi)
    rescaled_data = img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert full_res_data.shape == rescaled_data.shape, "Failed outbound test"
    try:
        img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])
    except Exception as e:
        raise AssertionError(f"Failed outbound test: {e}") from e
