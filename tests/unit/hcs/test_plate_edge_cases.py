"""Edge-case coverage tests for `ngio.hcs._plate`."""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from ngio import (
    Roi,
    create_empty_plate,
    open_ome_zarr_plate,
)
from ngio.tables import (
    ConditionTable,
    FeatureTable,
    GenericRoiTable,
    MaskingRoiTable,
    RoiTable,
)
from ngio.utils import NgioValueError


def _make_roi(name: str, label: int | None = None) -> Roi:
    return Roi.from_values(
        name=name,
        slices={"x": (0, 10), "y": (0, 10), "z": (0, 10)},
        label=label,
    )


def test_get_image_missing_in_well(cardiomyocyte_tiny_path_readonly: Path):
    plate = open_ome_zarr_plate(cardiomyocyte_tiny_path_readonly, mode="r")
    with pytest.raises(ValueError, match="does not exist in well"):
        plate.get_image("B", "03", "not_an_image")


def test_wells_and_images_cache(cardiomyocyte_tiny_path_readonly: Path):
    plate = open_ome_zarr_plate(cardiomyocyte_tiny_path_readonly, cache=True, mode="r")
    # First get_image populates the cache, later calls return the cached object
    image = plate.get_image("B", "03", "0")
    assert plate.get_image("B", "03", "0") is image

    # First get_well populates the cache, later calls return the cached object
    _ = plate.get_well("B", "03")
    well = plate.get_well("B", "03")
    assert plate.get_well("B", "03") is well


def test_add_image_none_path_raises(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate")
    with pytest.raises(ValueError, match="Image path cannot be None"):
        plate.add_image(row="B", column="03", image_path=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Image path cannot be None"):
        plate.atomic_add_image(row="B", column="03", image_path=None)  # ty: ignore[invalid-argument-type]


def test_atomic_remove_image(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate")
    plate.add_image(row="B", column="03", image_path="0")
    plate.add_image(row="B", column="03", image_path="1")

    plate.atomic_remove_image(row="B", column="03", image_path="1")
    assert plate.images_paths() == ["B/03/0"]
    assert plate.wells_paths() == ["B/03"]

    # Removing the last image of a well also removes the well
    plate.atomic_remove_image(row="B", column="03", image_path="0")
    assert plate.images_paths() == []
    assert plate.wells_paths() == []


def test_tables_container_missing_on_readonly_plate(tmp_path: Path):
    create_empty_plate(tmp_path / "plate.zarr", name="plate")
    plate = open_ome_zarr_plate(tmp_path / "plate.zarr", mode="r")
    # No tables group exists and it cannot be created in read-only mode
    assert plate.list_tables() == []
    with pytest.raises(NgioValueError, match="No tables container found"):
        _ = plate.tables_container


def test_typed_table_getters(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate")
    plate.add_table("roi", RoiTable(rois=[_make_roi("roi_1")]))
    plate.add_table("masking", MaskingRoiTable(rois=[_make_roi("1", label=1)]))
    plate.add_table("feature", FeatureTable(pd.DataFrame({"label": [1], "x": [0.5]})))
    plate.add_table(
        "condition", ConditionTable(table_data=pd.DataFrame({"cond": ["a"]}))
    )

    assert isinstance(plate.get_roi_table("roi"), RoiTable)
    with pytest.raises(NgioValueError, match="is not a ROI table"):
        plate.get_roi_table("masking")

    assert isinstance(plate.get_masking_roi_table("masking"), MaskingRoiTable)
    with pytest.raises(NgioValueError, match="is not a masking ROI table"):
        plate.get_masking_roi_table("roi")

    assert isinstance(plate.get_feature_table("feature"), FeatureTable)
    with pytest.raises(NgioValueError, match="is not a feature table"):
        plate.get_feature_table("roi")

    # RoiTable is a GenericRoiTable subclass, so it passes the generic getter
    assert isinstance(plate.get_generic_roi_table("roi"), GenericRoiTable)
    with pytest.raises(NgioValueError, match="is not a generic ROI table"):
        plate.get_generic_roi_table("feature")

    assert isinstance(plate.get_condition_table("condition"), ConditionTable)
    with pytest.raises(NgioValueError, match="is not a condition table"):
        plate.get_condition_table("roi")


def test_get_table_as(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate")
    plate.add_table("roi", RoiTable(rois=[_make_roi("roi_1")]))
    table = plate.get_table_as("roi", RoiTable)
    assert isinstance(table, RoiTable)
    assert len(table.rois()) == 1


def test_concatenate_image_tables_as(cardiomyocyte_small_mip_path_readonly: Path):
    plate = open_ome_zarr_plate(cardiomyocyte_small_mip_path_readonly, mode="r")
    table = plate.concatenate_image_tables_as(
        name="regionprops_DAPI", table_cls=FeatureTable
    )
    assert isinstance(table, FeatureTable)

    async_table = asyncio.run(
        plate.concatenate_image_tables_as_async(
            name="regionprops_DAPI", table_cls=FeatureTable
        )
    )
    assert isinstance(async_table, FeatureTable)
    assert set(table.dataframe.columns) == set(async_table.dataframe.columns)


def test_get_well_returns_cached_instance(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate", cache=True)
    plate.add_well(row="A", column=1)

    well_1 = plate.get_well(row="A", column=1)
    well_2 = plate.get_well(row="A", column=1)
    assert well_1 is well_2
