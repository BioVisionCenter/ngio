import dask.array as da
import numpy as np
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import FeatureExtractorIterator


def _build_iterator() -> FeatureExtractorIterator:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 16, 16)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
    )
    label = ome_zarr.derive_label(name="label")
    image = ome_zarr.get_image()
    iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label,
        channel_selection=0,
        axes_order="yx",
    )
    return iterator.by_yx()


def test_feature_iterator_numpy():
    iterator = _build_iterator()

    n_items = 0
    for data, seg, roi in iterator.iter_as_numpy():
        assert isinstance(data, np.ndarray)
        assert isinstance(seg, np.ndarray)
        assert data.shape == seg.shape
        assert roi is not None
        n_items += 1
    assert n_items == len(iterator.rois)

    # Lazy iteration yields the getter objects with image/label properties
    for getter in iterator.iter(lazy=True, data_mode="numpy", iterator_mode="readonly"):
        assert isinstance(getter.image, np.ndarray)  # ty: ignore[unresolved-attribute]
        assert isinstance(getter.label, np.ndarray)  # ty: ignore[unresolved-attribute]
        assert getter.image.shape == getter.label.shape  # ty: ignore[unresolved-attribute]


def test_feature_iterator_dask():
    iterator = _build_iterator()

    n_items = 0
    for data, seg, roi in iterator.iter_as_dask():
        assert isinstance(data, da.Array)
        assert isinstance(seg, da.Array)
        assert data.shape == seg.shape
        assert roi is not None
        n_items += 1
    assert n_items == len(iterator.rois)

    # Lazy iteration yields the getter objects with image/label properties
    for getter in iterator.iter(lazy=True, data_mode="dask", iterator_mode="readonly"):
        assert isinstance(getter.image, da.Array)  # ty: ignore[unresolved-attribute]
        assert isinstance(getter.label, da.Array)  # ty: ignore[unresolved-attribute]
        assert getter.image.shape == getter.label.shape  # ty: ignore[unresolved-attribute]


def test_feature_iterator_is_readonly():
    iterator = _build_iterator()

    # The feature extractor is a read-only iterator: no setters are built
    roi = iterator.rois[0]
    assert iterator.build_numpy_setter(roi) is None
    assert iterator.build_dask_setter(roi) is None
    assert iterator.post_consolidate() is None
