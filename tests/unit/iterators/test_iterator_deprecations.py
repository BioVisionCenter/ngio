import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import SegmentationIterator
from ngio.utils import NgioDeprecationWarning, NgioFutureWarning


def _build_iterator() -> SegmentationIterator:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        chunks=(1, 16, 16),
    )
    label = ome_zarr.derive_label("label")
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx"
    )
    return iterator.by_chunks()


def test_bare_iter_warns_and_still_yields_dask():
    iterator = _build_iterator()
    with pytest.warns(NgioFutureWarning, match="ngio=1.2"):
        gen = iterator.iter()
    for patch, _writer in gen:
        assert isinstance(patch, da.Array)


def test_explicit_dask_mode_warns_deprecation():
    iterator = _build_iterator()
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        gen = iterator.iter(lazy=False, data_mode="dask")
    for patch, _writer in gen:
        assert isinstance(patch, da.Array)


def test_numpy_spellings_are_silent():
    # The suite runs with filterwarnings=error, so any warning below fails loudly
    iterator = _build_iterator()
    for patch, writer in iterator.iter(lazy=False, data_mode="numpy"):
        writer(np.zeros_like(patch, dtype=np.uint8))
    for patch, writer in iterator.iter_as_numpy():
        writer(np.zeros_like(patch, dtype=np.uint8))
    iterator.map(lambda x: np.zeros_like(x, dtype=np.uint8))
    iterator.map_as_numpy(lambda x: np.zeros_like(x, dtype=np.uint8))
    iterator.reduce(lambda x: float(x.mean()))
    iterator.reduce_as_numpy(lambda x: float(x.mean()))


def test_dask_trio_warns_deprecation():
    iterator = _build_iterator()
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        iterator.iter_as_dask()
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.2"):
        iterator.reduce_as_dask(lambda x: float(x.mean().compute()))


def test_map_is_equivalent_to_alias():
    iterator = _build_iterator()
    label = iterator.output_image

    iterator.map(lambda x: np.full_like(x, 3))
    via_map = label.zarr_array[...].copy()
    iterator.map_as_numpy(lambda x: np.full_like(x, 3))
    np.testing.assert_array_equal(label.zarr_array[...], via_map)


def test_reduce_is_equivalent_to_alias():
    iterator = _build_iterator()
    assert iterator.reduce(lambda x: float(x.mean())) == iterator.reduce_as_numpy(
        lambda x: float(x.mean())
    )


def test_map_with_max_workers():
    iterator = _build_iterator()
    label = iterator.output_image

    iterator.map(lambda x: np.full_like(x, 5), max_workers=2)
    np.testing.assert_array_equal(
        label.zarr_array[...],
        np.full(label.shape, 5, dtype=label.zarr_array.dtype),
    )
