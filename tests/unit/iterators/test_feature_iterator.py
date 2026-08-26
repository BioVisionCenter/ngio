import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import FeatureExtractorIterator
from ngio.utils import NgioDeprecationWarning, NgioValueError


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
        assert isinstance(getter.image, np.ndarray)
        assert isinstance(getter.label, np.ndarray)
        assert getter.image.shape == getter.label.shape


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
        assert isinstance(getter.image, da.Array)
        assert isinstance(getter.label, da.Array)
        assert getter.image.shape == getter.label.shape


def test_feature_iterator_is_readonly():
    iterator = _build_iterator()

    # The feature extractor is read-only: the writer surface does not exist.
    assert not hasattr(iterator, "build_numpy_setter")
    assert not hasattr(iterator, "map")
    # finalize is the distributed gather: without banked partials it refuses
    # loudly instead of quietly doing nothing.
    with pytest.raises(NgioValueError, match="No partials"):
        iterator.finalize()


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
    """Both result shapes join into the same table; the write is the caller's."""
    ome_zarr, iterator = _build_container_and_iterator()

    from_dicts = iterator.measure(_measure_as_dict)
    assert from_dicts is not None
    assert sorted(from_dicts.dataframe.index.tolist()) == [1, 2]
    assert from_dicts.dataframe.index.name == "label"
    assert "feats" not in ome_zarr.list_tables(), "nothing is written implicitly"

    from_frames = iterator.measure(_measure_as_frame)
    assert from_frames is not None
    assert from_frames.dataframe.equals(from_dicts.dataframe)

    # The caller stores the table; a round-trip through the container works.
    # The storage backend regroups columns by dtype, so compare ignoring
    # column order — this also pins that roi_index/roi_name survive storage.
    import pandas as pd

    ome_zarr.add_table("feats", from_dicts)
    read_back = ome_zarr.get_table("feats")
    pd.testing.assert_frame_equal(
        read_back.dataframe, from_dicts.dataframe, check_like=True
    )


def test_measure_parallel_matches_serial():
    from ngio.iterators import ThreadedMapper

    _, iterator = _build_container_and_iterator()

    serial = iterator.measure(_measure_as_dict)
    threaded = iterator.measure(_measure_as_dict, mapper=ThreadedMapper(4))
    assert serial is not None and threaded is not None
    assert threaded.dataframe.equals(serial.dataframe)


def test_measure_with_a_declared_join():
    import pandas as pd

    from ngio.tables import GenericTable

    _, iterator = _build_container_and_iterator()

    def totals(results):
        # The join receives normalized DataFrames (empty ROIs are
        # empty frames), never the function's raw dicts.
        joined = pd.concat([r for r in results if len(r)])
        return GenericTable(pd.DataFrame({"total_objects": [len(joined)]}))

    table = iterator.with_join(totals).measure(_measure_as_dict)
    assert table is not None
    assert table.dataframe["total_objects"].tolist() == [2]


def test_measure_all_empty_returns_empty_table():
    """Zero objects is a legitimate outcome — same contract as `detect`."""
    _, iterator = _build_container_and_iterator()

    def nothing(image, label, roi):
        return {"label": [], "mean": []}

    table = iterator.measure(nothing)
    assert table is not None
    assert len(table.dataframe) == 0


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


# --- provenance columns and the read-only halo -------------------------------


def test_measure_default_table_carries_provenance_columns():
    import pandas as pd

    _, iterator = _build_container_and_iterator()

    table = iterator.measure(_measure_as_dict)
    assert table is not None
    frame = table.dataframe
    assert pd.api.types.is_integer_dtype(frame["roi_index"])
    names = {iterator.rois[i].get_name() for i in frame["roi_index"]}
    assert set(frame["roi_name"]) == names


def test_measure_normalizes_results_for_a_declared_join():
    """Every join sees DataFrames with a label column and provenance."""
    import pandas as pd

    from ngio.tables import GenericTable

    _, iterator = _build_container_and_iterator()

    def label_indexed(image, label, roi):
        # A label-indexed frame: normalization must reset it to a column.
        frame = pd.DataFrame(_measure_as_dict(image, label, roi))
        return frame.set_index("label") if len(frame) else frame

    captured: list = []

    def capture(results):
        captured.extend(results)
        return GenericTable(pd.DataFrame({"n": [len(results)]}))

    iterator.with_join(capture).measure(label_indexed)
    assert len(captured) == len(iterator.rois)
    for index, frame in enumerate(captured):
        assert isinstance(frame, pd.DataFrame)
        if not len(frame):
            assert list(frame.columns) == []  # empty ROIs stay column-less
            continue
        assert list(frame.columns[-2:]) == ["roi_index", "roi_name"]
        assert (frame["roi_index"] == index).all()
        assert (frame["roi_name"] == iterator.rois[index].get_name()).all()
        assert "label" in frame.columns


@pytest.mark.parametrize("reserved", ["_ngio_index", "roi_index", "roi_name"])
def test_measure_refuses_reserved_columns(reserved):
    _, iterator = _build_container_and_iterator()

    def shadowing(image, label, roi):
        return {"label": [1], reserved: [0]}

    with pytest.raises(NgioValueError, match="reserved column"):
        iterator.measure(shadowing)


def test_measure_with_halo_duplicates_then_join_dedups():
    """A border object is measured by every grown region; roi_index dedups."""
    import pandas as pd

    from ngio import create_ome_zarr_from_array

    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(2, 16, 16)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=array, pixelsize=1.0, axes_names="cyx", levels=1
    )
    label = ome_zarr.derive_label(name="nuclei")
    label_data = np.zeros((16, 16), dtype="uint32")
    label_data[2:6, 2:6] = 1
    label_data[6:10, 6:10] = 3  # straddles the 8-px grid boundary
    label.set_array(label_data)
    label.consolidate()
    iterator = (
        FeatureExtractorIterator(
            input_image=ome_zarr.get_image(),
            input_label=ome_zarr.get_label("nuclei"),
            channel_selection=0,
            axes_order="yx",
        )
        .by_grid(size_x=8, size_y=8)
        .with_halo(x=4, y=4)
    )

    def measure(image, label_patch, roi):
        ids = [int(v) for v in np.unique(label_patch) if v]
        return {
            "label": ids,
            "pixel_count": [int((label_patch == i).sum()) for i in ids],
            "mean": [float(image[label_patch == i].mean()) for i in ids],
        }

    # The default join keeps the duplicates (status quo), silently.
    table = iterator.measure(measure)
    assert table is not None
    counts = table.dataframe.index.value_counts()
    assert counts.loc[3] > 1, "the border object must be measured repeatedly"

    # The documented recipe: keep, per object, the row from the region that
    # saw the most of it — roi provenance makes the choice explicit.
    def dedup(results):
        from ngio.tables import FeatureTable

        joined = pd.concat([r for r in results if len(r)])
        joined = (
            joined.sort_values("pixel_count", ascending=False)
            .drop_duplicates("label")
            .set_index("label")
            .sort_index()
        )
        return FeatureTable(table_data=joined, reference_label="nuclei")

    deduped = iterator.with_join(dedup).measure(measure)
    assert deduped is not None
    frame = deduped.dataframe
    assert frame.index.is_unique
    assert sorted(frame.index.tolist()) == [1, 3]
    # With a 4-px halo every 8-px tile sees object 3 whole: 16 pixels.
    assert frame.loc[3, "pixel_count"] == 16


def test_with_join_carries_through_the_chain_and_refuses_on_a_slice():
    from ngio.iterators import ConcatJoin

    _, iterator = _build_container_and_iterator()
    declared = iterator.with_join(ConcatJoin(reference_label="nuclei"))
    reshaped = declared.by_grid(size_x=4, size_y=4).with_halo(x=2, y=2)
    assert isinstance(reshaped._join, ConcatJoin), "the declaration survives"

    with pytest.raises(NgioValueError, match="callable"):
        iterator.with_join("not-a-join")  # type: ignore[arg-type]


def test_concat_join_is_public_and_delegable():
    """A declared ConcatJoin equals the undeclared default, column for column."""
    import pandas as pd

    from ngio.iterators import ConcatJoin

    _, iterator = _build_container_and_iterator()
    default = iterator.measure(_measure_as_dict)
    declared = iterator.with_join(ConcatJoin(reference_label="nuclei")).measure(
        _measure_as_dict
    )
    assert default is not None and declared is not None
    pd.testing.assert_frame_equal(default.dataframe, declared.dataframe)
