from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.io_pipes._ops_slices_utils import chunk_rects_intersect
from ngio.iterators import (
    FeatureExtractorIterator,
    IterUnit,
    SegmentationIterator,
)
from ngio.utils import NgioValueError


def _build_ome_zarr(chunks="auto", ngff_version="0.4"):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 16, 16)).astype("uint8")
    return create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        chunks=chunks,
        ngff_version=ngff_version,
    )


def _build_feature_iterator(ome_zarr) -> FeatureExtractorIterator:
    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.derive_label(name="feature_label"),
        channel_selection=0,
        axes_order="yx",
    )
    return iterator.grid(size_x=8, size_y=8)


def _build_segmentation_iterator(ome_zarr, label_name="seg_label", **derive_kwargs):
    label = ome_zarr.derive_label(name=label_name, **derive_kwargs)
    iterator = SegmentationIterator(
        ome_zarr.get_image(), label, channel_selection=0, axes_order="yx"
    )
    return iterator.grid(size_x=8, size_y=8), label


def test_reduce_as_numpy_roi_order_and_equivalence():
    iterator = _build_feature_iterator(_build_ome_zarr())
    assert len(iterator.rois) == 4

    results = iterator.reduce_as_numpy(
        lambda data: (data[2].name, float(data[0].mean()))
    )

    assert len(results) == len(iterator.rois)
    for result, roi in zip(results, iterator.rois, strict=True):
        assert result[0] == roi.name

    # Equivalent to the hand-rolled tutorial idiom
    expected = [
        (roi.name, float(image.mean()))
        for image, _, roi in iterator.iter_as_numpy()
    ]
    assert results == expected


def test_reduce_returns_dataframe():
    iterator = _build_feature_iterator(_build_ome_zarr())

    def to_features(data) -> pd.DataFrame:
        image, label, roi = data
        return pd.DataFrame({"roi": [roi.name], "mean": [image.mean()]})

    results = iterator.reduce_as_numpy(to_features)
    assert all(isinstance(result, pd.DataFrame) for result in results)
    table = pd.concat(results, ignore_index=True)
    assert list(table["roi"]) == [roi.name for roi in iterator.rois]


def test_reduce_as_dask():
    iterator = _build_feature_iterator(_build_ome_zarr())
    results = iterator.reduce_as_dask(
        lambda data: (data[2].name, float(data[0].mean().compute()))
    )
    assert [name for name, _ in results] == [roi.name for roi in iterator.rois]


def test_reduce_never_writes():
    iterator, label = _build_segmentation_iterator(_build_ome_zarr())
    before = label.zarr_array[...].copy()

    results = iterator.reduce_as_numpy(lambda patch: np.full_like(patch, 7))

    assert len(results) == len(iterator.rois)
    np.testing.assert_array_equal(label.zarr_array[...], before)


def test_map_as_numpy_still_writes_and_returns_none():
    iterator, label = _build_segmentation_iterator(_build_ome_zarr())

    out = iterator.map_as_numpy(lambda patch: np.full_like(patch, 7))

    assert out is None
    np.testing.assert_array_equal(
        label.zarr_array[...], np.full(label.shape, 7, dtype=label.zarr_array.dtype)
    )


def test_map_readonly_raises_before_func_runs():
    iterator = _build_feature_iterator(_build_ome_zarr())
    calls: list[int] = []

    def recording_func(data):
        calls.append(1)
        return data

    with pytest.raises(NgioValueError, match="read-only"):
        iterator.map_as_numpy(recording_func)
    with pytest.raises(NgioValueError, match="read-only"):
        iterator.map_as_dask(recording_func)
    assert calls == []


def test_iter_unit_write_footprint():
    iterator, label = _build_segmentation_iterator(_build_ome_zarr())

    units = list(iterator._numpy_units_generator())
    assert [unit.index for unit in units] == list(range(len(iterator.rois)))
    for unit in units:
        footprint = unit.write_footprint
        assert footprint is not None
        # Rank of the output (yx label), inclusive (first, last) per axis
        assert len(footprint) == 2
        assert all(first <= last for first, last in footprint)

    readonly_units = list(iterator._numpy_units_generator(with_setters=False))
    assert all(unit.write_footprint is None for unit in readonly_units)

    feature_units = list(
        _build_feature_iterator(_build_ome_zarr())._numpy_units_generator()
    )
    assert all(unit.write_footprint is None for unit in feature_units)


def test_iter_unit_write_footprint_shard_granularity():
    ome_zarr = _build_ome_zarr(chunks=(2, 8, 8), ngff_version="0.5")

    iterator, _ = _build_segmentation_iterator(
        ome_zarr, label_name="sharded", chunks=(2, 8, 8), shards=(2, 16, 16)
    )
    footprints = [unit.write_footprint for unit in iterator._numpy_units_generator()]
    # All four 8x8 tiles land in the single 16x16 shard
    assert all(chunk_rects_intersect(footprints[0], fp) for fp in footprints[1:])

    iterator, _ = _build_segmentation_iterator(
        ome_zarr, label_name="unsharded", chunks=(2, 8, 8)
    )
    footprints = [unit.write_footprint for unit in iterator._numpy_units_generator()]
    # Without sharding each tile owns its own chunk
    assert not any(chunk_rects_intersect(footprints[0], fp) for fp in footprints[1:])


class ShuffledMapper:
    """A MapperProtocol implementation that runs units in reverse order."""

    def __call__(self, func: Callable, units: Iterable[IterUnit]) -> list:
        materialized = list(units)
        results = []
        for unit in reversed(materialized):
            results.append((unit.index, func(unit.getter())))
        results.sort(key=lambda item: item[0])
        return [result for _, result in results]


def test_custom_shuffled_mapper_pins_roi_order():
    iterator = _build_feature_iterator(_build_ome_zarr())
    func = lambda data: (data[2].name, float(data[0].mean()))  # noqa: E731

    shuffled = iterator.reduce_as_numpy(func, mapper=ShuffledMapper())
    basic = iterator.reduce_as_numpy(func)

    assert shuffled == basic
    assert [name for name, _ in shuffled] == [roi.name for roi in iterator.rois]
