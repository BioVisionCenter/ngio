"""Coverage tests for error paths in OmeZarrContainer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr.storage import MemoryStore

from ngio import (
    Roi,
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    open_label,
    open_ome_zarr_container,
)
from ngio.images import Label
from ngio.ome_zarr_meta.ngio_specs import Channel
from ngio.tables import ConditionTable, FeatureTable, MaskingRoiTable, RoiTable
from ngio.utils import NgioValidationError, NgioValueError


def _make_container(store=None, channels=("DAPI", "GFP")):
    """Create a small multi-channel OME-Zarr container in memory."""
    return create_empty_ome_zarr(
        store if store is not None else MemoryStore(),
        shape=(len(channels), 8, 8),
        axes_names=("c", "y", "x"),
        pixelsize=0.5,
        levels=2,
        channels_meta=list(channels),
    )


def _add_label(ome_zarr, name: str) -> None:
    label = ome_zarr.derive_label(name)
    data = np.zeros((8, 8), dtype=np.uint32)
    data[:4, :4] = 1
    label.set_array(data)
    label.consolidate()


def test_repr_few_labels_and_tables():
    ome_zarr = _make_container()
    _add_label(ome_zarr, "lbl1")
    ome_zarr.add_table("t1", ome_zarr.build_image_roi_table())

    repr_str = repr(ome_zarr)
    assert "labels=['lbl1']" in repr_str
    assert "tables=['t1']" in repr_str


def test_repr_many_labels_and_tables():
    ome_zarr = _make_container()
    for name in ["lbl1", "lbl2", "lbl3"]:
        ome_zarr.derive_label(name)
    for name in ["t1", "t2", "t3"]:
        ome_zarr.add_table(name, ome_zarr.build_image_roi_table())

    repr_str = repr(ome_zarr)
    assert "#labels=3" in repr_str
    assert "#tables=3" in repr_str


def test_axes_setup_property():
    ome_zarr = _make_container()
    assert ome_zarr.axes_setup == ome_zarr.images_container.axes_setup


def test_set_axes_units_updates_labels():
    ome_zarr = _make_container()
    _add_label(ome_zarr, "lbl1")
    ome_zarr.set_axes_units(space_unit="micrometer", set_labels=True)
    assert ome_zarr.space_unit == "micrometer"
    assert ome_zarr.get_label("lbl1").space_unit == "micrometer"


def test_tables_container_raises_on_readonly_without_tables(tmp_path: Path):
    store = tmp_path / "no_tables.zarr"
    create_empty_ome_zarr(store, shape=(8, 8), pixelsize=0.5, levels=2)
    ome_zarr = open_ome_zarr_container(store, mode="r")

    assert ome_zarr.list_tables() == []
    assert ome_zarr.list_labels() == []
    with pytest.raises(NgioValidationError, match="No tables found"):
        _ = ome_zarr.tables_container


def test_get_masked_image_without_names_raises():
    ome_zarr = _make_container()
    with pytest.raises(NgioValueError, match="Neither masking_label_name"):
        ome_zarr.get_masked_image()


def test_get_masked_image_table_without_reference_label():
    ome_zarr = _make_container()
    _add_label(ome_zarr, "lbl1")
    roi = Roi.from_values(slices={"y": (0.0, 2.0), "x": (0.0, 2.0)}, name="1", label=1)
    table = MaskingRoiTable(rois=[roi])
    ome_zarr.add_table("no_ref", table)

    with pytest.raises(NgioValueError, match="does not have a reference"):
        ome_zarr.get_masked_image(masking_table_name="no_ref")


@pytest.fixture()
def container_with_tables():
    ome_zarr = _make_container()
    ome_zarr.add_table("roi", ome_zarr.build_image_roi_table())
    feature = FeatureTable(pd.DataFrame({"label": [1, 2], "feat": [0.1, 0.2]}))
    ome_zarr.add_table("feat", feature)
    condition = ConditionTable(table_data=pd.DataFrame({"condition": ["a", "b"]}))
    ome_zarr.add_table("cond", condition)
    return ome_zarr


def test_get_roi_table_wrong_type(container_with_tables):
    with pytest.raises(NgioValueError, match="is not a ROI table"):
        container_with_tables.get_roi_table("feat")


def test_get_masking_roi_table_wrong_type(container_with_tables):
    with pytest.raises(NgioValueError, match="is not a masking ROI table"):
        container_with_tables.get_masking_roi_table("roi")


def test_get_feature_table(container_with_tables):
    table = container_with_tables.get_feature_table("feat")
    assert isinstance(table, FeatureTable)
    with pytest.raises(NgioValueError, match="is not a feature table"):
        container_with_tables.get_feature_table("roi")


def test_get_generic_roi_table_wrong_type(container_with_tables):
    with pytest.raises(NgioValueError, match="is not a generic ROI table"):
        container_with_tables.get_generic_roi_table("feat")


def test_get_condition_table(container_with_tables):
    table = container_with_tables.get_condition_table("cond")
    assert isinstance(table, ConditionTable)
    with pytest.raises(NgioValueError, match="is not a condition table"):
        container_with_tables.get_condition_table("roi")


def test_get_table_as(container_with_tables):
    table = container_with_tables.get_table_as("roi", RoiTable)
    assert isinstance(table, RoiTable)


def test_build_masking_roi_table_from_label():
    ome_zarr = _make_container()
    _add_label(ome_zarr, "lbl1")
    table = ome_zarr.build_masking_roi_table("lbl1")
    assert isinstance(table, MaskingRoiTable)
    assert len(table.rois()) == 1


@pytest.mark.parametrize(
    "zarr_key", ["v04/test_image_yx.zarr", "v05/test_image_yx.zarr"]
)
def test_open_label(images_all_versions_readonly: dict[str, Path], zarr_key: str):
    path = images_all_versions_readonly[zarr_key]

    # Open a label group directly (name=None branch)
    label = open_label(path / "labels" / "label", mode="r")
    assert isinstance(label, Label)

    # Open through the labels container (name provided branch)
    label_by_name = open_label(path / "labels", name="label", mode="r")
    assert isinstance(label_by_name, Label)
    assert label_by_name.path == label.path
    assert label_by_name.shape == label.shape


def test_create_from_array_bad_percentiles():
    with pytest.raises(NgioValueError, match="tuple of two values"):
        create_ome_zarr_from_array(
            MemoryStore(),
            array=np.zeros((8, 8), dtype=np.uint16),
            pixelsize=0.5,
            levels=2,
            percentiles=(1.0, 50.0, 99.0),  # ty: ignore[invalid-argument-type]
        )


def test_create_empty_with_channel_objects():
    channels = [
        Channel.default_init(label="DAPI", wavelength_id="A01_C01"),
        Channel.default_init(label="GFP", wavelength_id="A02_C02"),
    ]
    ome_zarr = _make_container(channels=channels)
    assert ome_zarr.wavelength_ids == ["A01_C01", "A02_C02"]


def test_labels_missing_on_read_only_container(tmp_path: Path):
    from ngio.utils import NgioValidationError

    store = tmp_path / "img.zarr"
    create_empty_ome_zarr(store=store, shape=(16, 16), pixelsize=1.0)
    container = open_ome_zarr_container(store, mode="r")

    # a read-only container without a labels group degrades gracefully
    assert container.list_labels() == []
    with pytest.raises(NgioValidationError):
        _ = container.labels_container
