import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import FeatureExtractorIterator
from ngio.utils import NgioDeprecationWarning


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
    with pytest.warns(NgioDeprecationWarning):
        dask_iter = iterator.iter_as_dask()
    for data, seg, roi in dask_iter:
        assert isinstance(data, da.Array)
        assert isinstance(seg, da.Array)
        assert data.shape == seg.shape
        assert roi is not None
        n_items += 1
    assert n_items == len(iterator.rois)

    # Lazy iteration yields the getter objects with image/label properties
    with pytest.warns(NgioDeprecationWarning):
        lazy_dask_iter = iterator.iter(
            lazy=True, data_mode="dask", iterator_mode="readonly"
        )
    for getter in lazy_dask_iter:
        assert isinstance(getter.image, da.Array)  # ty: ignore[unresolved-attribute]
        assert isinstance(getter.label, da.Array)  # ty: ignore[unresolved-attribute]
        assert getter.image.shape == getter.label.shape  # ty: ignore[unresolved-attribute]


def test_feature_iterator_is_readonly():
    iterator = _build_iterator()

    # The feature extractor is a read-only iterator: no setters are built
    roi = iterator.rois[0]
    assert iterator.build_numpy_setter(roi) is None
    assert iterator.build_dask_setter(roi) is None
    assert iterator.finalize() is None


def _build_container_and_iterator():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 16, 16)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
    )
    label = ome_zarr.derive_label(name="nuclei")
    label_data = np.zeros((16, 16), dtype="uint32")
    label_data[2:6, 2:6] = 1
    label_data[10:14, 10:14] = 2
    label.set_array(label_data)
    label.consolidate()
    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.get_label("nuclei"),
        channel_selection=0,
        axes_order="yx",
    ).by_grid(size_x=8, size_y=8)
    return ome_zarr, iterator


def _measure_as_dict(image, label, roi):
    ids = [int(value) for value in np.unique(label) if value]
    return {
        "label": ids,
        "mean": [float(image[label == i].mean()) for i in ids],
    }


def _measure_as_frame(image, label, roi):
    import pandas as pd

    return pd.DataFrame(_measure_as_dict(image, label, roi))


def test_measure_returns_a_feature_table():
    """Both result shapes coalesce into the same table; the write is the caller's."""
    ome_zarr, iterator = _build_container_and_iterator()

    from_dicts = iterator.measure(_measure_as_dict)
    assert sorted(from_dicts.dataframe.index.tolist()) == [1, 2]
    assert from_dicts.dataframe.index.name == "label"
    assert "feats" not in ome_zarr.list_tables(), "nothing is written implicitly"

    from_frames = iterator.measure(_measure_as_frame)
    assert from_frames.dataframe.equals(from_dicts.dataframe)

    # The caller stores the table; a round-trip through the container works.
    ome_zarr.add_table("feats", from_dicts)
    read_back = ome_zarr.get_table("feats")
    assert read_back.dataframe.equals(from_dicts.dataframe)


def test_measure_parallel_matches_serial():
    from ngio.iterators import ThreadedMapper

    _, iterator = _build_container_and_iterator()

    serial = iterator.measure(_measure_as_dict)
    threaded = iterator.measure(_measure_as_dict, mapper=ThreadedMapper(4))
    assert threaded.dataframe.equals(serial.dataframe)


def test_measure_custom_coalesce():
    import pandas as pd

    from ngio.tables import GenericTable

    _, iterator = _build_container_and_iterator()

    def totals(results):
        joined = pd.concat([pd.DataFrame(r) for r in results if r["label"]])
        return GenericTable(pd.DataFrame({"total_objects": [len(joined)]}))

    table = iterator.measure(_measure_as_dict, coalesce=totals)
    assert table.dataframe["total_objects"].tolist() == [2]


def test_measure_all_empty_raises():
    from ngio.utils import NgioValueError

    _, iterator = _build_container_and_iterator()

    def nothing(image, label, roi):
        return {"label": [], "mean": []}

    with pytest.raises(NgioValueError, match="zero rows"):
        iterator.measure(nothing)


def test_measure_requires_a_label_key():
    from ngio.utils import NgioValueError

    _, iterator = _build_container_and_iterator()

    def unlabelled(image, label, roi):
        return {"mean": [1.0]}

    with pytest.raises(NgioValueError, match="no 'label' column"):
        iterator.measure(unlabelled)


def test_feature_getter_reads_once_and_releases():
    """`.image`/`.label` then `get()` share one read; `get()` drops the cache."""
    _, iterator = _build_container_and_iterator()

    getter = iterator.build_numpy_getter(iterator.rois[0])
    image = getter.image
    label = getter.label
    got_image, got_label, _ = getter.get()
    assert got_image is image
    assert got_label is label
    # A consumed getter must not retain its patches: `reduce` keeps every
    # unit alive until the whole run returns.
    assert getter._image_data is None
    assert getter._label_data is None
