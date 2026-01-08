from pathlib import Path

import pytest

from ngio import Image, create_empty_ome_zarr, open_image
from ngio.transforms import ZoomTransform
from ngio.utils import NgioValueError


@pytest.mark.parametrize(
    "zarr_name",
    [
        "v04/test_image_yx.zarr",
        "v04/test_image_cyx.zarr",
        "v04/test_image_zyx.zarr",
        "v04/test_image_czyx.zarr",
        "v04/test_image_c1yx.zarr",
        "v04/test_image_tyx.zarr",
        "v04/test_image_tcyx.zarr",
        "v04/test_image_tzyx.zarr",
        "v04/test_image_tczyx.zarr",
        "v05/test_image_yx.zarr",
        "v05/test_image_cyx.zarr",
        "v05/test_image_zyx.zarr",
        "v05/test_image_czyx.zarr",
        "v05/test_image_c1yx.zarr",
        "v05/test_image_tyx.zarr",
        "v05/test_image_tcyx.zarr",
        "v05/test_image_tzyx.zarr",
        "v05/test_image_tczyx.zarr",
    ],
)
def test_open_image(images_all_versions: dict[str, Path], zarr_name: str):
    path = images_all_versions[zarr_name]
    image = open_image(path)
    assert isinstance(image, Image)


@pytest.mark.parametrize(
    ("shape1", "shape2", "axes1", "axes2", "should_raise"),
    [
        ((1, 3, 10, 10), (1, 3, 12, 10), "czyx", "czyx", False),
        ((1, 10, 10), (1, 1, 10, 10), "cyx", "czyx", True),
        ((3, 10, 10), (3, 3, 10, 10), "zyx", "czyx", False),
        ((1, 1, 10, 10), (1, 1, 10, 10), "tcyx", "czyx", True),
    ],
)
def test_image_require_axes_match(
    tmp_path: Path, shape1, shape2, axes1, axes2, should_raise: bool
):
    path1 = tmp_path / "image1.zarr"
    path2 = tmp_path / "image2.zarr"
    create_empty_ome_zarr(
        store=path1, shape=shape1, axes_names=axes1, xy_pixelsize=0.5, levels=1
    )
    create_empty_ome_zarr(
        store=path2, shape=shape2, axes_names=axes2, xy_pixelsize=0.5, levels=1
    )
    img1 = open_image(path1)
    img2 = open_image(path2)

    if should_raise:
        with pytest.raises(NgioValueError):
            img1.require_axes_match(img2)
    else:
        img1.require_axes_match(img2)


@pytest.mark.parametrize(
    ("shape1", "shape2", "axes1", "axes2", "allow_singleton", "should_raise"),
    [
        ((1, 3, 10, 10), (1, 3, 10, 10), "czyx", "czyx", False, False),
        ((1, 3, 10, 10), (1, 3, 10, 10), "czyx", "czyx", True, False),
        ((1, 1, 10, 10), (1, 3, 10, 10), "czyx", "czyx", True, False),
        ((3, 10, 10), (3, 3, 10, 10), "zyx", "czyx", True, False),
        ((1, 1, 10, 10), (1, 3, 10, 10), "czyx", "czyx", False, True),
        ((1, 2, 10, 10), (1, 3, 10, 10), "czyx", "czyx", False, True),
    ],
)
def test_image_require_dimensions_match(
    tmp_path: Path, shape1, shape2, axes1, axes2, allow_singleton, should_raise: bool
):
    path1 = tmp_path / "image1.zarr"
    path2 = tmp_path / "image2.zarr"
    create_empty_ome_zarr(
        store=path1, shape=shape1, axes_names=axes1, xy_pixelsize=0.5, levels=1
    )
    create_empty_ome_zarr(
        store=path2, shape=shape2, axes_names=axes2, xy_pixelsize=0.5, levels=1
    )
    img1 = open_image(path1)
    img2 = open_image(path2)

    if should_raise:
        with pytest.raises(NgioValueError):
            img1.require_dimensions_match(img2, allow_singleton=allow_singleton)
    else:
        img1.require_dimensions_match(img2, allow_singleton=allow_singleton)


@pytest.mark.parametrize(
    ("shape1", "shape2", "axes1", "axes2", "xy_pixelsize", "should_raise"),
    [
        ((1, 3, 10, 10), (1, 3, 10, 10), "czyx", "czyx", 1.0, False),
        ((1, 3, 10, 10), (1, 3, 20, 20), "czyx", "czyx", 0.5, False),
        ((1, 3, 10, 10), (1, 3, 20, 20), "czyx", "czyx", 2.0, True),
    ],
)
def test_image_require_can_be_rescaled(
    tmp_path: Path,
    shape1,
    shape2,
    axes1,
    axes2,
    xy_pixelsize: float,
    should_raise: bool,
):
    path1 = tmp_path / "image1.zarr"
    path2 = tmp_path / "image2.zarr"
    create_empty_ome_zarr(
        store=path1, shape=shape1, axes_names=axes1, xy_pixelsize=1.0, levels=1
    )
    create_empty_ome_zarr(
        store=path2, shape=shape2, axes_names=axes2, xy_pixelsize=xy_pixelsize, levels=1
    )
    img1 = open_image(path1)
    img2 = open_image(path2)

    # Also test with transforms
    if should_raise:
        with pytest.raises(NgioValueError):
            img1.require_rescalable(img2)
        return None

    img1.require_rescalable(img2)

    # Also test with transforms
    zoom = ZoomTransform(
        input_image=img2,
        target_image=img1,
        order="nearest",
    )
    img1_data = img1.get_as_numpy()
    img2_data = img2.get_as_numpy(transforms=[zoom])
    assert img1_data.shape == img2_data.shape


@pytest.mark.parametrize(
    ("shape1", "shape2", "axes1", "axes2", "xy_pixelsize", "should_raise"),
    [
        ((111, 111), (112, 112), "yx", "yx", 0.99, False),
        ((111, 111), (111, 111), "yx", "yx", 2.0, True),
        ((111, 111), (55, 55), "yx", "yx", 2.0, False),
        ((111, 111), (56, 56), "yx", "yx", 2.0, False),
        ((111, 111), (54, 54), "yx", "yx", 2.0, True),
    ],
)
def test_image_require_can_be_rescaled2(
    tmp_path: Path,
    shape1,
    shape2,
    axes1,
    axes2,
    xy_pixelsize: float,
    should_raise: bool,
):
    path1 = tmp_path / "image1.zarr"
    path2 = tmp_path / "image2.zarr"
    create_empty_ome_zarr(
        store=path1, shape=shape1, axes_names=axes1, xy_pixelsize=1.0, levels=1
    )
    create_empty_ome_zarr(
        store=path2, shape=shape2, axes_names=axes2, xy_pixelsize=xy_pixelsize, levels=1
    )
    img1 = open_image(path1)
    img2 = open_image(path2)

    # Also test with transforms
    if should_raise:
        with pytest.raises(NgioValueError):
            img2.require_rescalable(img1)
        return None
    zoom = ZoomTransform(
        input_image=img2,
        target_image=img1,
        order="nearest",
    )

    img1_data = img1.get_as_numpy()
    img2_data = img2.get_as_numpy(transforms=[zoom])
    assert img1_data.shape == img2_data.shape, (shape1, shape2, xy_pixelsize)
    img2.set_array(patch=img2_data, transforms=[zoom])

    roi = img1.build_image_roi_table().rois()[0]
    img2_roi_data = img2.get_roi_as_numpy(roi, transforms=[zoom])
    # Roi data should match exactly
    assert img1_data.shape == img2_roi_data.shape
    img2.set_roi(roi=roi, patch=img2_roi_data, transforms=[zoom])


def test_zoom_virtual_axes(
    tmp_path: Path,
):
    path1 = tmp_path / "image1.zarr"
    path2 = tmp_path / "image2.zarr"
    create_empty_ome_zarr(
        store=path1, shape=(3, 16, 16, 16), axes_names="czyx", xy_pixelsize=1.0
    )
    create_empty_ome_zarr(
        store=path2, shape=(16, 32, 32), axes_names="zyx", xy_pixelsize=0.5
    )
    img1 = open_image(path1)
    img2 = open_image(path2)

    # Also test with transforms
    zoom = ZoomTransform(
        input_image=img2,
        target_image=img1,
        order="nearest",
    )

    img1_data = img1.get_as_numpy()
    img2_data = img2.get_as_numpy(transforms=[zoom], axes_order="czyx")
    img2.set_array(patch=img2_data, transforms=[zoom], axes_order="czyx")
    assert img2_data.shape[0] == 1  # Virtual channel axis

    roi = img1.build_image_roi_table().rois()[0]
    img2_roi_data = img2.get_roi_as_numpy(roi, transforms=[zoom], axes_order="czyx")
    # Roi data should match exactly except for virtual axis
    assert img1_data.shape[1:] == img2_roi_data.shape[1:]
    img2.set_roi(roi=roi, patch=img2_roi_data, transforms=[zoom], axes_order="czyx")


def test_derive_image_consistency(
    tmp_path: Path,
):
    path1 = tmp_path / "image1.zarr"
    ome_zarr = create_empty_ome_zarr(
        store=path1,
        shape=(3, 16, 16, 16),
        axes_names="czyx",
        xy_pixelsize=1.0,
        levels=["s0", "s1", "s2"],
    )
    ome_zarr_der = ome_zarr.derive_image(tmp_path / "derived_image.zarr")
    assert ome_zarr.channel_labels == ome_zarr_der.channel_labels
    assert ome_zarr.levels_paths == ome_zarr_der.levels_paths
    assert ome_zarr.space_unit == ome_zarr_der.space_unit
    assert ome_zarr.time_unit == ome_zarr_der.time_unit

    label = ome_zarr.derive_label("derived_label")
    assert label.path == "s0"
