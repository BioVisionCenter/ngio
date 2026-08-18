from pathlib import Path

import numpy as np
import pytest

from ngio import Roi, create_empty_ome_zarr
from ngio.transforms import ZoomTransform
from ngio.utils import NgioValueError


def _make_pair(
    tmp_path,
    *,
    target_shape,
    input_shape,
    axes_names,
    target_pix,
    input_pix,
    target_z=1.0,
    input_z=1.0,
    channels_meta=None,
):
    """Create a (input_image, target_image) pair for testing."""
    target_img = create_empty_ome_zarr(
        store=tmp_path / "target.zarr",
        shape=target_shape,
        axes_names=axes_names,
        pixelsize=target_pix,
        z_spacing=target_z,
        channels_meta=channels_meta,
        levels=1,
        overwrite=True,
    ).get_image()
    input_img = create_empty_ome_zarr(
        store=tmp_path / "input.zarr",
        shape=input_shape,
        axes_names=axes_names,
        pixelsize=input_pix,
        z_spacing=input_z,
        channels_meta=channels_meta,
        levels=1,
        overwrite=True,
    ).get_image()
    return input_img, target_img


# ---------- Original test (kept as-is) ----------


def test_zoom_from_dimensions(tmp_path: Path):
    full_res_img = create_empty_ome_zarr(
        store=tmp_path / "original.zarr",
        shape=(101, 101),
        axes_names="yx",
        pixelsize=1.0,
    ).get_image()

    img = create_empty_ome_zarr(
        store=tmp_path / "img.zarr",
        shape=(25, 25),
        axes_names="yx",
        pixelsize=4.0,
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


# ---------- Test 1: Scale factors ----------


@pytest.mark.parametrize(
    "target_shape, input_shape, target_pix, input_pix",
    [
        ((100, 100), (50, 50), 1.0, 2.0),
        ((99, 99), (33, 33), 1.0, 3.0),
        ((100, 100), (25, 25), 1.0, 4.0),
    ],
    ids=["2x", "3x", "4x"],
)
def test_zoom_scale_factors(
    tmp_path,
    target_shape,
    input_shape,
    target_pix,
    input_pix,
):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=target_shape,
        input_shape=input_shape,
        axes_names="yx",
        target_pix=target_pix,
        input_pix=input_pix,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (10, 30), "y": (10, 30)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 2: Asymmetric scale ----------


@pytest.mark.parametrize(
    "target_shape, input_shape, target_pix, input_pix",
    [
        ((100, 100), (25, 50), (1.0, 1.0), (4.0, 2.0)),
        ((60, 90), (60, 30), (1.0, 1.0), (1.0, 3.0)),
    ],
    ids=["y4x_x2x", "y1x_x3x"],
)
def test_zoom_asymmetric_scale(
    tmp_path,
    target_shape,
    input_shape,
    target_pix,
    input_pix,
):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=target_shape,
        input_shape=input_shape,
        axes_names="yx",
        target_pix=target_pix,
        input_pix=input_pix,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (5, 20), "y": (5, 20)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 3: Dimensionality ----------


@pytest.mark.parametrize(
    "target_shape, input_shape, axes_names, target_pix, input_pix, "
    "target_z, input_z, channels_meta, roi_slices, n_channels",
    [
        # 2D yx
        (
            (80, 80),
            (40, 40),
            "yx",
            1.0,
            2.0,
            1.0,
            1.0,
            None,
            {"x": (5, 20), "y": (5, 20)},
            None,
        ),
        # 3D zyx
        (
            (10, 80, 80),
            (10, 40, 40),
            "zyx",
            1.0,
            2.0,
            1.0,
            1.0,
            None,
            {"x": (5, 20), "y": (5, 20)},
            None,
        ),
        # 3D zyx with z scaling
        (
            (10, 80, 80),
            (5, 40, 40),
            "zyx",
            1.0,
            2.0,
            1.0,
            2.0,
            None,
            {"x": (5, 20), "y": (5, 20)},
            None,
        ),
        # cyx (channel must NOT be scaled)
        (
            (3, 80, 80),
            (3, 40, 40),
            "cyx",
            1.0,
            2.0,
            1.0,
            1.0,
            ["ch1", "ch2", "ch3"],
            {"x": (5, 20), "y": (5, 20)},
            3,
        ),
        # czyx
        (
            (3, 10, 80, 80),
            (3, 10, 40, 40),
            "czyx",
            1.0,
            2.0,
            1.0,
            1.0,
            ["ch1", "ch2", "ch3"],
            {"x": (5, 20), "y": (5, 20)},
            3,
        ),
        # tyx (time axis)
        (
            (4, 80, 80),
            (4, 40, 40),
            "tyx",
            1.0,
            2.0,
            1.0,
            1.0,
            None,
            {"x": (5, 20), "y": (5, 20)},
            None,
        ),
    ],
    ids=["yx", "zyx", "zyx_zscale", "cyx", "czyx", "tyx"],
)
def test_zoom_dimensionality(
    tmp_path,
    target_shape,
    input_shape,
    axes_names,
    target_pix,
    input_pix,
    target_z,
    input_z,
    channels_meta,
    roi_slices,
    n_channels,
):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=target_shape,
        input_shape=input_shape,
        axes_names=axes_names,
        target_pix=target_pix,
        input_pix=input_pix,
        target_z=target_z,
        input_z=input_z,
        channels_meta=channels_meta,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices=roi_slices)

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape

    # Channel axis must be preserved exactly
    if n_channels is not None:
        assert rescaled_data.shape[0] == n_channels

    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 4: Interpolation orders ----------


@pytest.mark.parametrize("order", ["nearest", "linear", "cubic"])
def test_zoom_interpolation_orders(tmp_path, order):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(100, 100),
        input_shape=(25, 25),
        axes_names="yx",
        target_pix=1.0,
        input_pix=4.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order=order,
    )
    roi = Roi.from_values(name=None, slices={"x": (10, 40), "y": (10, 40)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 5: ROI positions ----------


@pytest.mark.parametrize(
    "roi_slices",
    [
        {"x": (0, 20), "y": (0, 20)},
        {"x": (30, 20), "y": (30, 20)},
        {"x": (80, 20), "y": (80, 20)},
        {"x": (90, 20), "y": (90, 20)},
        {"x": (0, 100), "y": (0, 100)},
    ],
    ids=["origin", "interior", "edge_aligned", "outbound", "full_image"],
)
def test_zoom_roi_positions(tmp_path, roi_slices):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(100, 100),
        input_shape=(25, 25),
        axes_names="yx",
        target_pix=1.0,
        input_pix=4.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices=roi_slices)

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 6: Non-divisible shapes ----------


@pytest.mark.parametrize(
    "target_shape, input_shape, input_pix",
    [
        ((100, 100), (33, 33), 3.0),
        ((100, 100), (34, 34), 3.0),
        ((101, 101), (25, 25), 4.0),
        ((97, 97), (49, 49), 2.0),
    ],
    ids=["100_3x_floor", "100_3x_ceil", "101_4x", "97_2x"],
)
def test_zoom_non_divisible_shapes(tmp_path, target_shape, input_shape, input_pix):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=target_shape,
        input_shape=input_shape,
        axes_names="yx",
        target_pix=1.0,
        input_pix=input_pix,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 50), "y": (0, 50)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 7: Dask vs NumPy ----------


def test_zoom_dask_vs_numpy(tmp_path):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(80, 80),
        input_shape=(20, 20),
        axes_names="yx",
        target_pix=1.0,
        input_pix=4.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (10, 30), "y": (10, 30)})

    numpy_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    dask_data = input_img.get_roi_as_dask(roi=roi, transforms=[zoom])
    assert numpy_data.shape == dask_data.shape
    assert numpy_data.shape == dask_data.compute().shape


def test_zoom_dask_vs_numpy_multichannel(tmp_path):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(2, 80, 80),
        input_shape=(2, 20, 20),
        axes_names="cyx",
        target_pix=1.0,
        input_pix=4.0,
        channels_meta=["ch1", "ch2"],
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (10, 30), "y": (10, 30)})

    numpy_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    dask_data = input_img.get_roi_as_dask(roi=roi, transforms=[zoom])
    assert numpy_data.shape == dask_data.shape
    assert numpy_data.shape == dask_data.compute().shape
    assert numpy_data.shape[0] == 2  # channels preserved


# ---------- Test 8: Round-trip ----------


def test_zoom_roundtrip(tmp_path):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(20, 20),
        input_shape=(10, 10),
        axes_names="yx",
        target_pix=1.0,
        input_pix=2.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )

    # Write known data to input image
    known_data = np.arange(100, dtype="uint16").reshape(10, 10)
    input_img.set_array(patch=known_data)

    # Read via zoom (upscale to target resolution)
    roi = Roi.from_values(name=None, slices={"x": (0, 20), "y": (0, 20)})
    rescaled = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert rescaled.shape == (20, 20)

    # Write back via zoom (downscale back to input resolution)
    input_img.set_roi(patch=rescaled, roi=roi, transforms=[zoom])

    # Verify round-trip preserves values
    roundtrip = input_img.get_as_numpy()
    np.testing.assert_array_equal(known_data, roundtrip)


# ---------- Test 9: Identity transform ----------


def test_zoom_identity_transform(tmp_path):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(50, 50),
        input_shape=(50, 50),
        axes_names="yx",
        target_pix=1.0,
        input_pix=1.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (10, 20), "y": (10, 20)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 10: Downscaling ----------


def test_zoom_downscaling(tmp_path):
    # Input is high-res, target is low-res
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(25, 25),
        input_shape=(100, 100),
        axes_names="yx",
        target_pix=4.0,
        input_pix=1.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 80), "y": (0, 80)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 11: Large scale factors ----------


@pytest.mark.parametrize(
    "target_shape, input_shape, input_pix",
    [
        ((100, 100), (10, 10), 10.0),
        ((100, 100), (5, 5), 20.0),
    ],
    ids=["10x", "20x"],
)
def test_zoom_large_scale_factor(tmp_path, target_shape, input_shape, input_pix):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=target_shape,
        input_shape=input_shape,
        axes_names="yx",
        target_pix=1.0,
        input_pix=input_pix,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 40), "y": (0, 40)})

    target_data = target_img.get_roi_as_numpy(roi=roi)
    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert target_data.shape == rescaled_data.shape
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 12: Channel preservation ----------


@pytest.mark.parametrize("n_channels", [1, 2, 5])
def test_zoom_channel_preservation(tmp_path, n_channels):
    ch_names = [f"ch{i}" for i in range(n_channels)]
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(n_channels, 80, 80),
        input_shape=(n_channels, 20, 20),
        axes_names="cyx",
        target_pix=1.0,
        input_pix=4.0,
        channels_meta=ch_names,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 40), "y": (0, 40)})

    rescaled_data = input_img.get_roi_as_numpy(roi=roi, transforms=[zoom])
    assert rescaled_data.shape[0] == n_channels
    input_img.set_roi(patch=rescaled_data, roi=roi, transforms=[zoom])


# ---------- Test 13: Inverse shape validation ----------


def test_zoom_inverse_shape_validation(tmp_path):
    input_img, target_img = _make_pair(
        tmp_path,
        target_shape=(100, 100),
        input_shape=(25, 25),
        axes_names="yx",
        target_pix=1.0,
        input_pix=4.0,
    )
    zoom = ZoomTransform(
        input_image=input_img,
        target_image=target_img,
        order="nearest",
    )
    roi = Roi.from_values(name=None, slices={"x": (0, 40), "y": (0, 40)})

    # Pass a deliberately wrong-sized array
    wrong_data = np.zeros((30, 30), dtype="uint16")
    with pytest.raises(ValueError, match="not compatible"):
        input_img.set_roi(patch=wrong_data, roi=roi, transforms=[zoom])


def test_zoom_clamps_a_negative_slice_start(tmp_path: Path):
    """The read clamps a negative start to 0; the zoomed shape must match it."""
    full_res_img = create_empty_ome_zarr(
        store=tmp_path / "full.zarr",
        shape=(101, 101),
        axes_names="yx",
        pixelsize=1.0,
    ).get_image()
    img = create_empty_ome_zarr(
        store=tmp_path / "coarse.zarr",
        shape=(25, 25),
        axes_names="yx",
        pixelsize=4.0,
    ).get_image()
    zoom = ZoomTransform(input_image=img, target_image=full_res_img, order="nearest")

    data = img.get_as_numpy(transforms=[zoom], y=slice(-5, 20))

    # The read delivered y=0:20 of the coarse grid, i.e. 80 full-res pixels —
    # deriving the target from the unclamped -5 would stretch it to 100.
    assert data.shape == (80, 101)

    # Both backends agree, and the patch writes back through the inverse.
    as_dask = img.get_as_dask(transforms=[zoom], y=slice(-5, 20)).compute()
    np.testing.assert_array_equal(data, as_dask)
    img.set_array(patch=data, transforms=[zoom], y=slice(-5, 20))


def _delicate_pair(tmp_path: Path):
    """A 4x pair whose extents (101 vs 25) do not divide evenly on purpose."""
    full = create_empty_ome_zarr(
        store=tmp_path / "full.zarr", shape=(101, 101), axes_names="yx", pixelsize=1.0
    ).get_image()
    coarse = create_empty_ome_zarr(
        store=tmp_path / "coarse.zarr", shape=(25, 25), axes_names="yx", pixelsize=4.0
    ).get_image()
    zoom = ZoomTransform(input_image=coarse, target_image=full, order="nearest")
    return full, coarse, zoom


def test_zoom_subpixel_origin_matches_target_grid(tmp_path: Path):
    """A ROI whose origin is not divisible by the scale still lands exactly.

    On the 4x-coarser grid this ROI is `slice(5.25, 10.5)` — the sub-pixel
    bounds only the *raw* slice carries. Deriving the shape from a normalized
    (integer-rounded) slice breaks exactly this case, which is why the forward
    path must never normalize.
    """
    full, coarse, zoom = _delicate_pair(tmp_path)
    roi = Roi.from_values(name=None, slices={"y": (21, 21), "x": (21, 21)})

    on_target_grid = full.get_roi_as_numpy(roi=roi)
    rescaled = coarse.get_roi_as_numpy(roi=roi, transforms=[zoom])

    assert rescaled.shape == on_target_grid.shape == (21, 21)
    coarse.set_roi(patch=rescaled, roi=roi, transforms=[zoom])


def test_zoom_of_an_empty_selection_is_empty(tmp_path: Path):
    """Zero pixels in, zero pixels out — not a crash in the zoom kernel.

    Both the zero-length slice and the slice entirely past the array's end
    used to reach `numpy_zoom` with a zero (or negative) extent and die in a
    divide-by-zero reshape.
    """
    _, coarse, zoom = _delicate_pair(tmp_path)

    for empty in (slice(10, 10), slice(30, 40)):
        data = coarse.get_as_numpy(transforms=[zoom], y=empty)
        assert data.shape[0] == 0 and data.shape[1] == 101
        as_dask = coarse.get_as_dask(transforms=[zoom], y=empty).compute()
        assert as_dask.shape == data.shape


def test_zoom_refuses_a_noncontiguous_selection(tmp_path: Path):
    """A list selection has no geometry a zoom factor applies to.

    Silently stretching three non-contiguous rows as if adjacent resamples
    across pixels that were never neighbours.
    """
    _, coarse, zoom = _delicate_pair(tmp_path)

    with pytest.raises(NgioValueError, match="non-contiguous"):
        coarse.get_as_numpy(transforms=[zoom], y=[0, 2, 7])
