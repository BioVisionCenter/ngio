import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import (
    BatchedMapper,
    FeatureExtractorIterator,
    ImageProcessingIterator,
    SegmentationIterator,
)
from ngio.utils import NgioValueError


def _build_ome_zarr():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 16, 16)).astype("uint8")
    return create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
    )


def _build_segmentation_iterator(ome_zarr, label_name="seg_label", **grid_kwargs):
    label = ome_zarr.derive_label(name=label_name)
    iterator = SegmentationIterator(
        ome_zarr.get_image(), label, channel_selection=0, axes_order="yx"
    )
    return iterator.by_grid(**grid_kwargs), label


def test_map_matches_basic_mapper_on_ragged_grid():
    # size 6 over 16 clips the border tiles to 4: three ragged shapes per axis
    ome_zarr = _build_ome_zarr()
    batched_it, batched_label = _build_segmentation_iterator(
        ome_zarr, "batched", size_x=6, size_y=6
    )
    basic_it, basic_label = _build_segmentation_iterator(
        ome_zarr, "basic", size_x=6, size_y=6
    )
    assert len(batched_it.rois) == 9

    func = lambda patch: patch // 2  # noqa: E731 - works per patch and per batch
    batched_it.map(func, mapper=BatchedMapper(batch_size=4))
    basic_it.map(func)

    np.testing.assert_array_equal(
        batched_label.zarr_array[...], basic_label.zarr_array[...]
    )


def test_map_matches_basic_mapper_with_border_halo():
    # The halo clips at image borders, so grown patches are ragged even on a
    # uniform tiling.
    ome_zarr = _build_ome_zarr()
    batched_it, batched_label = _build_segmentation_iterator(
        ome_zarr, "batched", size_x=8, size_y=8
    )
    basic_it, basic_label = _build_segmentation_iterator(
        ome_zarr, "basic", size_x=8, size_y=8
    )

    func = lambda patch: patch // 2  # noqa: E731
    batched_it.with_halo(x=2, y=2).map(func, mapper=BatchedMapper(batch_size=4))
    basic_it.with_halo(x=2, y=2).map(func)

    np.testing.assert_array_equal(
        batched_label.zarr_array[...], basic_label.zarr_array[...]
    )


def test_batch_sizes_and_padding():
    ome_zarr = _build_ome_zarr()
    iterator, _label = _build_segmentation_iterator(ome_zarr, size_x=6, size_y=6)

    seen_shapes = []

    def recording_func(batch):
        seen_shapes.append(batch.shape)
        return batch

    iterator.map(recording_func, mapper=BatchedMapper(batch_size=4, pad_values=7))

    # 9 ragged tiles in batches of 4, 4, 1 — each stacked (B, y, x), padded to
    # the batch's per-axis maximum.
    assert [shape[0] for shape in seen_shapes] == [4, 4, 1]
    assert all(len(shape) == 3 for shape in seen_shapes)


def test_constant_pad_value_reaches_the_batch():
    ome_zarr = _build_ome_zarr()
    iterator, _label = _build_segmentation_iterator(ome_zarr, size_x=6, size_y=6)

    padded_corners = []

    def recording_func(batch):
        padded_corners.append(batch[-1, -1, -1])
        return batch

    # One batch of all 9 ragged tiles: the last tile is the 4x4 corner, padded
    # up to 6x6, so its bottom-right pixel is fill.
    iterator.map(recording_func, mapper=BatchedMapper(batch_size=16, pad_values=7))
    assert padded_corners == [7]


def test_image_processing_batched_inference():
    ome_zarr = _build_ome_zarr()
    batched_out = ome_zarr.derive_image(store=MemoryStore())
    basic_out = ome_zarr.derive_image(store=MemoryStore())

    def fake_model(batch):
        # An NN-style func: one (B, c, y, x) input, one call per batch.
        assert batch.ndim == 4
        return batch // 2

    batched_it = ImageProcessingIterator(
        ome_zarr.get_image(), batched_out.get_image()
    ).by_grid(size_x=6, size_y=6)
    basic_it = ImageProcessingIterator(
        ome_zarr.get_image(), basic_out.get_image()
    ).by_grid(size_x=6, size_y=6)

    batched_it.map(fake_model, mapper=BatchedMapper(batch_size=4))
    basic_it.map(lambda patch: patch // 2)

    np.testing.assert_array_equal(
        batched_out.get_image().zarr_array[...],
        basic_out.get_image().zarr_array[...],
    )


def test_reduce_with_per_item_reduction():
    # A uniform tiling, so no padding leaks into the per-item means.
    ome_zarr = _build_ome_zarr()
    iterator, _label = _build_segmentation_iterator(ome_zarr, size_x=8, size_y=8)

    batched = iterator.reduce(
        lambda batch: batch.reshape(len(batch), -1).mean(axis=1),
        mapper=BatchedMapper(batch_size=3),
    )
    basic = iterator.reduce(lambda patch: patch.mean())

    np.testing.assert_allclose(batched, basic)


def test_reduce_trims_shape_preserving_results():
    ome_zarr = _build_ome_zarr()
    iterator, _label = _build_segmentation_iterator(ome_zarr, size_x=6, size_y=6)

    results = iterator.reduce(lambda batch: batch, mapper=BatchedMapper(batch_size=4))

    expected = [patch for patch, _ in iterator.iter_as_numpy()]
    assert len(results) == len(expected)
    for result, patch in zip(results, expected, strict=True):
        np.testing.assert_array_equal(result, patch)


def test_written_units_collect_none():
    ome_zarr = _build_ome_zarr()
    iterator, label = _build_segmentation_iterator(ome_zarr, size_x=8, size_y=8)
    units = list(iterator._numpy_units_generator())

    results = BatchedMapper(batch_size=3)(lambda batch: batch * 0 + 3, units)

    assert results == [None] * len(units)
    np.testing.assert_array_equal(
        label.zarr_array[...], np.full(label.shape, 3, dtype=label.zarr_array.dtype)
    )


def test_wrong_leading_axis_raises():
    ome_zarr = _build_ome_zarr()
    iterator, _label = _build_segmentation_iterator(ome_zarr, size_x=8, size_y=8)

    with pytest.raises(NgioValueError, match="leading axis"):
        iterator.map(lambda batch: batch[:1], mapper=BatchedMapper(batch_size=4))


def test_tuple_units_refused():
    ome_zarr = _build_ome_zarr()
    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.derive_label(name="feature_label"),
        channel_selection=0,
        axes_order="yx",
    ).by_grid(size_x=8, size_y=8)

    with pytest.raises(NgioValueError, match="bare-array"):
        # The type error is the point: tuple-unit iterators cannot be batched,
        # and the runtime refusal below is the net for untyped callers.
        iterator.reduce(
            lambda batch: batch,
            mapper=BatchedMapper(),  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(NgioValueError, match="batch_size"):
        BatchedMapper(batch_size=batch_size)


def test_invalid_read_workers():
    with pytest.raises(NgioValueError, match="max_workers"):
        BatchedMapper(read_workers=0)


def test_serial_reads_match_threaded_reads():
    ome_zarr = _build_ome_zarr()
    threaded_it, threaded_label = _build_segmentation_iterator(
        ome_zarr, "threaded", size_x=6, size_y=6
    )
    serial_it, serial_label = _build_segmentation_iterator(
        ome_zarr, "serial", size_x=6, size_y=6
    )

    func = lambda patch: patch // 2  # noqa: E731
    threaded_it.map(func, mapper=BatchedMapper(batch_size=4, read_workers=4))
    serial_it.map(func, mapper=BatchedMapper(batch_size=4, read_workers=1))

    np.testing.assert_array_equal(
        threaded_label.zarr_array[...], serial_label.zarr_array[...]
    )


def test_iter_batched_matches_map():
    """The manual batched loop lands the same pixels as `map`, ragged and haloed."""
    for halo in (0, 2):
        ome_zarr = _build_ome_zarr()
        manual_it, manual_label = _build_segmentation_iterator(
            ome_zarr, f"manual_{halo}", size_x=6, size_y=6
        )
        mapped_it, mapped_label = _build_segmentation_iterator(
            ome_zarr, f"mapped_{halo}", size_x=6, size_y=6
        )
        if halo:
            manual_it = manual_it.with_halo(x=halo, y=halo)
            mapped_it = mapped_it.with_halo(x=halo, y=halo)

        func = lambda patch: (patch > 128).astype("uint32")  # noqa: E731
        for patches, writers in manual_it.iter_batched(3):
            assert len(patches) == len(writers) <= 3
            for patch, writer in zip(patches, writers, strict=True):
                writer(func(patch))
        mapped_it.map(func)

        np.testing.assert_array_equal(
            manual_label.zarr_array[...], mapped_label.zarr_array[...]
        )


def test_iter_batched_lengths_and_order():
    """Batches follow ROI order, full batches first, the tail smaller."""
    ome_zarr = _build_ome_zarr()
    iterator, _ = _build_segmentation_iterator(ome_zarr, "order", size_x=8, size_y=8)
    n_rois = len(iterator.rois)

    lengths = []
    seen = []
    batches = iterator.iter_batched(3)
    for patches, writers in batches:
        lengths.append(len(patches))
        for patch, writer in zip(patches, writers, strict=True):
            seen.append(patch.shape)
            writer(np.zeros_like(patch, dtype="uint32"))
    assert sum(lengths) == n_rois
    assert lengths[:-1] == [3] * (len(lengths) - 1)
    assert seen == [
        tuple(  # ROI order: the same shapes iter_as_numpy yields.
            patch.shape
        )
        for patch, _ in iterator.iter_as_numpy()
    ]


def test_iter_batched_finalizes_on_drain():
    """Fully draining the generator consolidates the pyramid, like `iter`."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 32, 32)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=2,
    )
    label = ome_zarr.derive_label(name="pyr")
    iterator = SegmentationIterator(
        ome_zarr.get_image(),
        label,
        channel_selection=0,
        axes_order="yx",
        consolidation_mode="dask",
    ).by_grid(size_x=16, size_y=16)

    for patches, writers in iterator.iter_batched(2):
        for patch, writer in zip(patches, writers, strict=True):
            writer(np.full_like(patch, 5, dtype="uint32"))

    level_1 = ome_zarr.get_label("pyr", path="1").get_as_numpy()
    assert (level_1 == 5).all(), "finalize did not rebuild the pyramid"


def test_iter_batched_feature_payloads():
    """The read-only feature iterator yields payload lists, no writers."""
    ome_zarr = _build_ome_zarr()
    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.derive_label(name="fb_label"),
        channel_selection=0,
        axes_order="yx",
    ).by_grid(size_x=8, size_y=8)

    count = 0
    for batch in iterator.iter_batched(3):
        assert len(batch) <= 3
        for image, seg, roi in batch:
            assert image.shape == seg.shape
            assert roi is not None
            count += 1
    assert count == len(iterator.rois)


def test_iter_batched_detection_payloads():
    """The detection iterator yields haloed `(patch, roi)` payload lists."""
    from ngio.iterators import ObjectDetectionIterator

    ome_zarr = _build_ome_zarr()
    iterator = (
        ObjectDetectionIterator(
            ome_zarr.get_image(), channel_selection=0, axes_order="yx"
        )
        .by_grid(size_x=8, size_y=8)
        .with_halo(x=2, y=2)
    )

    count = 0
    for batch in iterator.iter_batched(4):
        for patch, roi in batch:
            assert patch.ndim >= 2
            assert roi is not None
            count += 1
    assert count == len(iterator.rois)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_iter_batched_invalid_batch_size(batch_size):
    ome_zarr = _build_ome_zarr()
    iterator, _ = _build_segmentation_iterator(ome_zarr, "bad", size_x=8, size_y=8)
    with pytest.raises(NgioValueError, match="batch_size"):
        next(iterator.iter_batched(batch_size))
    feature = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.derive_label(name="bad_label"),
        channel_selection=0,
        axes_order="yx",
    )
    with pytest.raises(NgioValueError, match="batch_size"):
        feature.iter_batched(batch_size)
