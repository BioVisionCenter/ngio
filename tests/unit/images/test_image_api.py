"""Coverage tests for channel metadata and channel selection in ngio images."""

from typing import Any

import dask.array as da
import numpy as np
import pytest
from pydantic import ValidationError
from zarr.storage import MemoryStore

from ngio import create_empty_ome_zarr
from ngio.images import ChannelSelectionModel
from ngio.ome_zarr_meta.ngio_specs import Channel, ChannelsMeta
from ngio.utils import NgioUserWarning, NgioValueError


def _make_container():
    """Create a small two-channel OME-Zarr container in memory."""
    channels = [
        Channel.default_init(label="DAPI", wavelength_id="A01_C01"),
        Channel.default_init(label="GFP", wavelength_id="A02_C02"),
    ]
    return create_empty_ome_zarr(
        MemoryStore(),
        shape=(2, 8, 8),
        axes_names=("c", "y", "x"),
        pixelsize=0.5,
        levels=2,
        channels_meta=channels,
    )


def test_channel_selection_model_index_must_be_int():
    with pytest.raises(ValidationError, match="must be an integer"):
        ChannelSelectionModel(mode="index", identifier="not_an_int")


def test_set_channel_meta_default_init():
    ome_zarr = _make_container()
    ome_zarr.set_channel_meta()
    assert ome_zarr.channel_labels == ["channel_0", "channel_1"]


def test_set_channel_meta_with_windows():
    ome_zarr = _make_container()
    ome_zarr.set_channel_meta(
        channel_meta=ChannelsMeta.default_init(
            labels=["c1", "c2"], start=[0.0, 5.0], end=[10.0, 20.0]
        )
    )
    assert ome_zarr.channel_labels == ["c1", "c2"]
    channels = ome_zarr.images_container.channels_meta.channels
    assert channels[0].channel_visualisation.start == 0.0
    assert channels[0].channel_visualisation.end == 10.0
    assert channels[1].channel_visualisation.start == 5.0
    assert channels[1].channel_visualisation.end == 20.0


def test_set_channel_labels_wrong_length():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="number of labels"):
        ome_zarr.set_channel_labels(["only_one"])


def test_set_channel_colors_wrong_length():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="number of colors"):
        ome_zarr.set_channel_colors(["FF0000"])


def test_set_channel_windows_wrong_lengths():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="start-end pairs"):
        ome_zarr.set_channel_windows(starts_ends=[(0.0, 1.0)])
    with pytest.raises(NgioValueError, match="min-max pairs"):
        ome_zarr.set_channel_windows(
            starts_ends=[(0.0, 1.0), (0.0, 1.0)], min_max=[(0.0, 1.0)]
        )


def test_percentiles_input_validation():
    ome_zarr = _make_container()
    # Tuple with wrong number of entries
    with pytest.raises(NgioValueError, match="tuple of two floats"):
        ome_zarr.set_channel_windows_with_percentiles(percentiles=(0.1, 50.0, 99.9))
    # Tuple with non-float entries
    with pytest.raises(NgioValueError, match="tuple of two floats"):
        ome_zarr.set_channel_windows_with_percentiles(percentiles=(1, 99))
    # List with a length different from the number of channels
    with pytest.raises(NgioValueError, match="number of channels"):
        ome_zarr.set_channel_windows_with_percentiles(percentiles=[(0.1, 99.9)])


def test_channel_selection_negative_index():
    image = _make_container().get_image()
    with pytest.raises(NgioValueError, match="non-negative"):
        image.get_array(channel_selection=-1)


def test_channel_selection_index_out_of_range():
    image = _make_container().get_image()
    with pytest.raises(NgioValueError, match="less than the number"):
        image.get_array(channel_selection=5)


def test_channel_selection_by_wavelength_id():
    image = _make_container().get_image()
    selection = ChannelSelectionModel(mode="wavelength_id", identifier="A02_C02")
    array = image.get_array(channel_selection=selection)
    np.testing.assert_array_equal(array, image.get_array(c=1))


@pytest.mark.parametrize("mode", ["numpy", "dask"])
def test_channel_selection_order_is_the_requested_one(mode):
    """Adjacent channels are read and written in the requested order (#255)."""
    image = _make_container().get_image()
    plane = np.ones((8, 8), dtype=image.dtype)
    image.set_array(plane * 1, channel_selection="DAPI")
    image.set_array(plane * 2, channel_selection="GFP")

    swapped = image.get_array(channel_selection=["GFP", "DAPI"], mode=mode)
    assert np.asarray(swapped)[:, 0, 0].tolist() == [2, 1]
    in_order = image.get_array(channel_selection=["DAPI", "GFP"], mode=mode)
    assert np.asarray(in_order)[:, 0, 0].tolist() == [1, 2]

    # The write follows the same order: patch position 0 lands on GFP.
    patch = np.stack([plane * 20, plane * 10])
    image.set_array(
        da.from_array(patch) if mode == "dask" else patch,
        channel_selection=["GFP", "DAPI"],
    )
    assert image.get_as_numpy(channel_selection="DAPI")[0, 0] == 10
    assert image.get_as_numpy(channel_selection="GFP")[0, 0] == 20


def test_channel_selection_invalid_type_in_sequence():
    image = _make_container().get_image()
    with pytest.raises(NgioValueError, match="Invalid channel selection"):
        image.get_array(channel_selection=[0.5])


def test_channel_selection_invalid_type():
    image = _make_container().get_image()
    with pytest.raises(NgioValueError, match="Invalid channel selection"):
        image.get_array(channel_selection=1.5)


def test_channel_selection_ambiguous_with_c_kwarg():
    image = _make_container().get_image()
    with pytest.raises(NgioValueError, match="ambiguous"):
        image.get_array(channel_selection=0, c=0)


def test_resolve_channel_selection():
    """The public resolver mirrors what the `get_*` methods accept.

    Resolution touches only metadata — this is the supported way to validate
    a selection (e.g. a task's `skip_if_missing`) before loading any data.
    """
    image = _make_container().get_image()

    assert image.resolve_channel_selection(None) == {}
    assert image.resolve_channel_selection(1) == {"c": 1}
    assert image.resolve_channel_selection("GFP") == {"c": 1}
    assert image.resolve_channel_selection(["DAPI", "GFP"]) == {"c": [0, 1]}
    assert image.resolve_channel_selection(
        [
            ChannelSelectionModel(mode="wavelength_id", identifier="A02_C02"),
            ChannelSelectionModel(mode="index", identifier="0"),
        ]
    ) == {"c": [1, 0]}

    with pytest.raises(NgioValueError):
        image.resolve_channel_selection("not_a_channel")
    with pytest.raises(NgioValueError):
        image.resolve_channel_selection(["DAPI", 7])


def _raw_windows(store) -> list[tuple[float, float]]:
    """Read the channel windows from the on-disk JSON, bypassing ngio's models."""
    import zarr

    attrs: dict[str, Any] = dict(zarr.open_group(store, mode="r").attrs)
    channels = attrs["omero"]["channels"]
    return [(c["window"]["start"], c["window"]["end"]) for c in channels]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # empty: no nonzero data at all
        (np.zeros((64, 64), dtype="uint16"), (0.0, 1.0)),
        # constant: percentiles and max coincide, drop the start to 0
        (np.full((64, 64), 7, dtype="uint16"), (0.0, 7.0)),
        # binary mask: the nonzero data is constant
        (np.indices((64, 64)).sum(axis=0) % 2, (0.0, 1.0)),
        # float constant
        (np.full((64, 64), 0.3, dtype="float32"), (0.0, np.float32(0.3))),
    ],
)
def test_percentile_window_never_degenerate(data, expected):
    """Viewers need start < end; the raw JSON must never carry start == end."""
    store = MemoryStore()
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(1, 64, 64),
        axes_names=("c", "y", "x"),
        pixelsize=1.0,
        levels=1,
        dtype=str(data.dtype),
    )
    ome_zarr.get_image().set_array(data[None])
    ome_zarr.set_channel_windows_with_percentiles()
    (window,) = _raw_windows(store)
    assert window[0] < window[1]
    assert window == pytest.approx(expected)


def test_percentile_window_sparse_opens_to_max():
    """A single outlier is beyond the 99.9th percentile: the window opens to it."""
    data = np.ones((1, 64, 64), dtype="uint8")
    data[0, 3, 5] = 2
    store = MemoryStore()
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=data.shape,
        axes_names=("c", "y", "x"),
        pixelsize=1.0,
        levels=1,
        dtype="uint8",
    )
    ome_zarr.get_image().set_array(data)
    ome_zarr.set_channel_windows_with_percentiles()
    assert _raw_windows(store) == [(1.0, 2.0)]


def test_set_channel_windows_warns_but_writes_as_given():
    """A non-increasing pair is spec-legal: warn, but write exactly what was asked."""
    store = MemoryStore()
    ome_zarr = create_empty_ome_zarr(
        store,
        shape=(2, 8, 8),
        axes_names=("c", "y", "x"),
        pixelsize=1.0,
        levels=1,
    )
    with pytest.warns(NgioUserWarning, match="start/end = \\(5.0, 5.0\\)"):
        ome_zarr.set_channel_windows(starts_ends=[(5.0, 5.0), (0.0, 1.0)])
    with pytest.warns(NgioUserWarning, match="start/end = \\(3.0, 1.0\\)"):
        ome_zarr.set_channel_windows(starts_ends=[(0.0, 1.0), (3.0, 1.0)])
    assert _raw_windows(store) == [(0.0, 1.0), (3.0, 1.0)]
    with pytest.warns(NgioUserWarning, match="min/max = \\(7, 7\\)"):
        ome_zarr.set_channel_windows(
            starts_ends=[(0.0, 1.0), (0.0, 1.0)], min_max=[(7, 7), (0, 255)]
        )
    # The model reflects the disk: no silent repair on read either.
    (bad, good) = ome_zarr.get_image().channels_meta.channels
    assert (bad.channel_visualisation.min, bad.channel_visualisation.max) == (7, 7)
    assert (good.channel_visualisation.min, good.channel_visualisation.max) == (0, 255)
