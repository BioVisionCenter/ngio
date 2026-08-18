"""The units API: split setters, the deprecated batch form, and issue #231/#232."""

import pytest

from ngio import create_empty_ome_zarr
from ngio.utils import NgioDeprecationWarning


def _units(container):
    meta = container.get_image()._meta_handler.get_meta()
    return {axis.name: axis.unit for axis in meta.axes_handler.axes}


def _container(ngff_version="0.4", channels=None):
    return create_empty_ome_zarr(
        store={},
        shape=(3, 2, 64, 64) if channels else (3, 64, 64),
        axes_names="tcyx" if channels else "tyx",
        pixelsize=1.0,
        time_spacing=1.0,
        levels=1,
        channels_meta=channels,
        ngff_version=ngff_version,
    )


def test_split_setters_touch_only_their_unit():
    """Issue #232: changing one unit must not silently reset the other."""
    container = _container()
    container.set_time_unit("hour")
    container.set_space_unit("nanometer")

    units = _units(container)
    assert units["t"] == "hour", "setting space reset time (issue #232)"
    assert units["x"] == units["y"] == "nanometer"

    container.set_time_unit("minute")
    assert _units(container)["x"] == "nanometer", "setting time reset space"


def test_batch_setter_is_deprecated_and_sets_both():
    """The batch form keeps its documented behavior: both units, every call."""
    container = _container()
    container.set_time_unit("hour")

    with pytest.warns(NgioDeprecationWarning, match="set_space_unit"):
        container.set_axes_units(space_unit="nanometer")

    units = _units(container)
    assert units["x"] == "nanometer"
    assert units["t"] == "second", "the batch form sets BOTH units by design"


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_metadata_updates_preserve_channel_labels(ngff_version):
    """Issue #231: unit and axes-name updates must not drop OMERO channels."""
    container = _container(ngff_version=ngff_version, channels=["H2B", "mem9"])

    container.set_space_unit("micrometer")
    assert container.channel_labels == ["H2B", "mem9"], "set_space_unit dropped them"

    container.set_axes_names(["t", "c", "y", "x"])
    assert container.channel_labels == ["H2B", "mem9"], "set_axes_names dropped them"


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_label_source_survives_metadata_updates(ngff_version):
    """The same reconstruction bug dropped a label's image-label source."""
    container = _container(ngff_version=ngff_version)
    label = container.derive_label("seg")
    before = label._meta_handler.get_meta().image_label

    label.set_space_unit("micrometer")
    after = label._meta_handler.get_meta().image_label
    assert after == before
