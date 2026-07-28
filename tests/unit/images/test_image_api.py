"""Coverage tests for channel metadata and channel selection in ngio images."""

import numpy as np
import pytest
from pydantic import ValidationError
from zarr.storage import MemoryStore

from ngio import create_empty_ome_zarr
from ngio.images import ChannelSelectionModel
from ngio.ome_zarr_meta.ngio_specs import Channel
from ngio.utils import NgioDeprecationWarning, NgioValueError


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


def test_images_container_levels_paths_deprecated():
    container = _make_container().images_container
    with pytest.warns(NgioDeprecationWarning, match="levels_paths"):
        paths = container.levels_paths
    assert paths == container.level_paths


def test_set_channel_meta_default_init():
    ome_zarr = _make_container()
    ome_zarr.set_channel_meta()
    assert ome_zarr.channel_labels == ["channel_0", "channel_1"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start": [0.0, 0.0]}, "end must be provided"),
        ({"end": [1.0, 1.0]}, "start must be provided"),
        (
            {
                "start": [0.0, 0.0],
                "end": [1.0, 1.0],
                "percentiles": (0.1, 99.9),
            },
            "percentiles must be None",
        ),
        ({"start": [0.0], "end": [1.0, 2.0]}, "same length"),
        ({"start": [0.0], "end": [1.0]}, "number of channels"),
    ],
)
def test_set_channel_meta_legacy_errors(kwargs, match):
    ome_zarr = _make_container()
    with (
        pytest.warns(NgioDeprecationWarning, match="deprecated"),
        pytest.raises(NgioValueError, match=match),
    ):
        ome_zarr.set_channel_meta(**kwargs)


def test_set_channel_meta_legacy_start_end():
    ome_zarr = _make_container()
    with pytest.warns(NgioDeprecationWarning, match="deprecated"):
        ome_zarr.set_channel_meta(
            labels=["c1", "c2"], start=[0.0, 5.0], end=[10.0, 20.0]
        )
    assert ome_zarr.channel_labels == ["c1", "c2"]
    channels = ome_zarr.images_container.channels_meta.channels
    assert channels[0].channel_visualisation.start == 0.0
    assert channels[0].channel_visualisation.end == 10.0
    assert channels[1].channel_visualisation.start == 5.0
    assert channels[1].channel_visualisation.end == 20.0


def test_set_channel_meta_legacy_percentiles():
    ome_zarr = _make_container()
    with pytest.warns(NgioDeprecationWarning, match="deprecated"):
        ome_zarr.set_channel_meta(percentiles=(0.1, 99.9))
    assert ome_zarr.num_channels == 2


def test_set_channel_labels_wrong_length():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="number of labels"):
        ome_zarr.set_channel_labels(["only_one"])


def test_set_channel_colors_wrong_length():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="number of colors"):
        ome_zarr.set_channel_colors(["FF0000"])


def test_images_container_set_channel_percentiles_deprecated():
    container = _make_container().images_container
    with pytest.warns(NgioDeprecationWarning, match="set_channel_percentiles"):
        container.set_channel_percentiles(start_percentile=1.0, end_percentile=99.0)


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
