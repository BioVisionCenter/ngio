"""Tests for `with_halo`: read a grown region, write only the ROI."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import (
    FeatureExtractorIterator,
    ImageProcessingIterator,
    MaskedSegmentationIterator,
    SegmentationIterator,
    ThreadedMapper,
)
from ngio.transforms import ZoomTransform
from ngio.utils import NgioValueError


def _container(
    shape=(1, 64, 64), chunks=(1, 16, 16), axes_names="cyx", levels=1, seed=0
):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=shape).astype("uint8")
    return (
        create_ome_zarr_from_array(
            store=MemoryStore(),
            array=array,
            pixelsize=1.0,
            axes_names=axes_names,
            levels=levels,
            chunks=chunks,
            consolidation_mode="dask",
        ),
        array,
    )


def _processing_iterator(**kwargs):
    ome_zarr, array = _container(**kwargs)
    out = ome_zarr.derive_image(store=MemoryStore()).get_image()
    iterator = ImageProcessingIterator(
        ome_zarr.get_image(),
        out,
        axes_order="cyx",
        consolidation_mode="dask",
    )
    return ome_zarr, array, iterator.by_write_units()


def _identity(patch):
    return patch


def _sum_over_patch(patch):
    """Position-dependent: every pixel takes the patch's own total.

    Any leak of halo pixels into the written core changes the value, so this
    separates "the halo was read" from "the halo was written".
    """
    return np.full_like(patch, patch.sum() % 251)


def test_halo_refuses_in_place_iteration():
    """Same array in and out: a grown read covers a neighbour tile's write."""
    ome_zarr, _ = _container()
    image = ome_zarr.get_image()
    iterator = ImageProcessingIterator(
        image, image, axes_order="cyx", consolidation_mode="dask"
    ).by_write_units()
    with pytest.raises(NgioValueError, match="In-place"):
        iterator.with_halo(x=4, y=4)


def test_halo_refuses_readonly_verbs_on_a_writer():
    """`reduce`/readonly `iter` would silently measure the grown regions."""
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)
    with pytest.raises(NgioValueError, match="grown regions"):
        haloed.reduce(lambda patch: float(patch.mean()))
    with pytest.raises(NgioValueError, match="grown regions"):
        haloed.iter(lazy=False, data_mode="numpy", iterator_mode="readonly")


def test_halo_is_transparent_for_a_pointwise_function():
    """A function that ignores its neighbours must write the same bytes."""
    _, _, plain = _processing_iterator()
    plain.map(_identity)
    plain_result = plain.output_image.get_as_numpy()

    _, _, haloed = _processing_iterator()
    haloed.with_halo(x=4, y=4).map(_identity)
    haloed_result = haloed.output_image.get_as_numpy()

    np.testing.assert_array_equal(plain_result, haloed_result)


def test_halo_changes_a_neighbourhood_dependent_function():
    """The context is genuinely read, not silently dropped."""
    _, _, plain = _processing_iterator()
    plain.map(_sum_over_patch)
    plain_result = plain.output_image.get_as_numpy()

    _, _, haloed = _processing_iterator()
    haloed.with_halo(x=4, y=4).map(_sum_over_patch)
    haloed_result = haloed.output_image.get_as_numpy()

    assert not np.array_equal(plain_result, haloed_result)


def test_halo_covers_the_whole_output():
    """Cropping must leave no gaps: every pixel is still written."""
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)
    haloed.map(lambda patch: np.full_like(patch, 42))

    np.testing.assert_array_equal(
        haloed.output_image.get_as_numpy(),
        np.full((1, 64, 64), 42, dtype="uint8"),
    )


def test_halo_read_region_grows_and_clips_at_borders():
    """An edge tile grows only on the sides where there is room."""
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)

    corner = haloed.rois[0]  # y=0, x=0 tile of a 16x16 grid
    core = haloed.build_numpy_getter(corner)
    assert core.get().shape == (1, 20, 20)  # 16 + 0 before + 4 after

    interior = next(
        roi
        for roi in haloed.rois
        if roi.to_pixel(haloed.ref_image.pixel_size)["y"].start == 16
        and roi.to_pixel(haloed.ref_image.pixel_size)["x"].start == 16
    )
    assert haloed.build_numpy_getter(interior).get().shape == (1, 24, 24)


def test_halo_leaves_write_footprints_untouched():
    """The whole point: the write region does not move, so parallelism holds."""
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)

    assert not iterator.check_if_write_units_overlap()
    assert not haloed.check_if_write_units_overlap()

    plain_setter = iterator.build_numpy_setter(iterator.rois[0])
    haloed_setter = haloed.build_numpy_setter(haloed.rois[0])
    assert (
        plain_setter.slicing_ops.normalized_slicing_tuple
        == haloed_setter.slicing_ops.normalized_slicing_tuple
    )


def test_haloed_iterator_still_maps_in_parallel():
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)
    haloed.map(lambda patch: np.full_like(patch, 7), mapper=ThreadedMapper(4))

    np.testing.assert_array_equal(
        haloed.output_image.get_as_numpy(), np.full((1, 64, 64), 7, dtype="uint8")
    )


def test_halo_survives_a_reshaping_call():
    """`_new_from_rois` carries the halo, so ordering of the calls is free."""
    _, _, iterator = _processing_iterator()
    assert iterator.with_halo(x=4).by_write_units().halo == {"x": 4}
    assert iterator.by_write_units().with_halo(x=4).halo == {"x": 4}


def test_core_shaped_return_is_refused():
    """Returning the core instead of the grown region is a loud error."""
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo(x=4, y=4)

    with pytest.raises(NgioValueError, match="core-sized array"):
        haloed.map(lambda patch: np.full_like(patch, 1)[:, 4:-4, 4:-4])


def test_halo_rejects_negative_margins():
    _, _, iterator = _processing_iterator()
    with pytest.raises(NgioValueError, match="must be >= 0"):
        iterator.with_halo(x=-1)


def test_feature_halo_grows_both_patches_in_step():
    """The feature halo is a read margin: image and label grow identically."""
    ome_zarr, _ = _container()
    label = ome_zarr.derive_label("lbl")
    label.set_array(np.ones((64, 64), dtype=label.zarr_array.dtype))
    iterator = FeatureExtractorIterator(
        ome_zarr.get_image(), label, axes_order="yx"
    ).by_grid(size_x=32, size_y=32)
    haloed = iterator.with_halo(x=4, y=4)

    core, _, _ = iterator.build_numpy_getter(iterator.rois[0]).get()
    image_patch, label_patch, roi = haloed.build_numpy_getter(haloed.rois[0]).get()
    # An interior-corner tile grows only into the image: +4 px on one side.
    assert image_patch.shape == (core.shape[0] + 4, core.shape[1] + 4)
    assert label_patch.shape == image_patch.shape
    # The reported roi covers the grown patch, not the core tile.
    x_slice = roi.get("x")
    assert x_slice is not None
    assert x_slice.length == core.shape[1] + 4


def test_zero_halo_is_a_no_op():
    _, _, iterator = _processing_iterator()
    haloed = iterator.with_halo()
    assert haloed.halo == {}
    assert (
        haloed.build_numpy_getter(haloed.rois[0]).get().shape
        == iterator.build_numpy_getter(iterator.rois[0]).get().shape
    )


def test_halo_refuses_a_shape_changing_transform():
    """A zoom on the data path puts the patch on a different grid than the halo.

    The margins are counted in reference-image pixels, so a rescaled patch
    cannot be cropped by them. This must refuse, not guess — an off-by-one here
    would be resampled into place by the zoom's own inverse and never noticed.
    """
    ome_zarr, _ = _container(levels=2)
    image = ome_zarr.get_image(path="0")
    coarse = ome_zarr.get_image(path="1")
    out = ome_zarr.derive_image(store=MemoryStore()).get_image()

    iterator = ImageProcessingIterator(
        image,
        out,
        axes_order="cyx",
        input_transforms=[ZoomTransform(input_image=image, target_image=coarse)],
        consolidation_mode="dask",
    ).by_grid(size_x=16, size_y=16)

    haloed = iterator.with_halo(x=4, y=4)
    roi = haloed.rois[5]
    patch = haloed.build_numpy_getter(roi).get()
    with pytest.raises(NgioValueError, match="shape-changing transform"):
        haloed.build_numpy_setter(roi).set(patch)


def test_mask_transform_composes_with_a_halo():
    """The crop runs before the transforms, so the mask still sees the core.

    Includes a label at a coarser level, so the mask's internal zoom is
    non-identity — that zoom rescales the label it reads, never the data
    array, which is why it does not disturb the halo arithmetic.
    """
    ome_zarr, _ = _container(shape=(64, 64), chunks=(16, 16), axes_names="yx", levels=2)
    label = ome_zarr.derive_label("m")
    mask_img = np.zeros((64, 64), dtype=label.zarr_array.dtype)
    mask_img[8:56, 8:56] = 1
    label.set_array(mask_img)
    label.consolidate(mode="dask")

    seg = ome_zarr.derive_label("seg")
    masked = ome_zarr.get_masked_image("m")
    iterator = MaskedSegmentationIterator(
        masked, seg, axes_order="yx", consolidation_mode="dask"
    )

    iterator.with_halo(x=4, y=4).map(lambda patch: np.full_like(patch, 7))

    written = ome_zarr.get_label("seg").get_as_numpy()
    assert (written[mask_img == 1] == 7).all()
    assert (written[mask_img == 0] == 0).all(), "the halo leaked past the mask"


def test_halo_on_segmentation_iterator():
    """The label path crops too, and singleton axes are left alone."""
    ome_zarr, _ = _container()
    label = ome_zarr.derive_label("seg")
    iterator = SegmentationIterator(
        ome_zarr.get_image(),
        label,
        channel_selection=0,
        axes_order="yx",
        consolidation_mode="dask",
    ).by_write_units()

    iterator.with_halo(x=4, y=4).map(lambda patch: np.full_like(patch, 9))

    np.testing.assert_array_equal(
        ome_zarr.get_label("seg").get_as_numpy(),
        np.full((64, 64), 9, dtype=label.zarr_array.dtype),
    )
