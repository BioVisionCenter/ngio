import asyncio
from pathlib import Path

import pandas.testing as pdt
import pytest
import zarr
from pandas import DataFrame

from ngio import (
    ImageInWellPath,
    OmeZarrContainer,
    OmeZarrWell,
    Roi,
    create_empty_plate,
    open_ome_zarr_plate,
)
from ngio.tables import GenericTable, RoiTable
from ngio.utils import NgioValueError


def test_open_real_ome_zarr_plate(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path
    ome_zarr_plate = open_ome_zarr_plate(cardiomyocyte_tiny_path)

    assert isinstance(ome_zarr_plate.__repr__(), str)
    assert ome_zarr_plate.columns == ["03"]
    assert ome_zarr_plate.rows == ["B"]
    assert ome_zarr_plate.acquisition_ids == [0]
    assert ome_zarr_plate.acquisitions_names == [
        "20200812-CardiomyocyteDifferentiation14-Cycle1"
    ]

    well_path = ome_zarr_plate._well_path("B", "03")
    well_path2 = ome_zarr_plate._well_path("B", 3)
    assert well_path == well_path2
    well = ome_zarr_plate.get_well("B", "03")
    assert well.paths() == ["0"]

    image_path = ome_zarr_plate._image_path("B", "03", "0")
    assert image_path == "B/03/0"

    images_plate = ome_zarr_plate.get_images()
    assert len(images_plate) == 1
    images_plate_async = asyncio.run(ome_zarr_plate.get_images_async())
    assert len(images_plate_async) == 1
    assert images_plate.keys() == images_plate_async.keys()

    _ = ome_zarr_plate.get_image("B", "03", "0")

    images_well = ome_zarr_plate.get_well_images("B", "03")
    assert len(images_well) == 1

    well = ome_zarr_plate.get_well("B", "03")
    assert isinstance(well, OmeZarrWell)
    assert well.paths() == ["0"]

    image = well.get_image("0")
    assert isinstance(image, OmeZarrContainer)
    assert well.get_image_acquisition_id("0") is None


def test_create_and_edit_plate(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    assert test_plate.columns == []
    assert test_plate.rows == []
    assert test_plate.acquisition_ids == []

    test_plate.add_image(row="B", column="03", image_path="0", acquisition_id=0)
    test_plate.add_image(row="B", column="03", image_path="1", acquisition_id=0)

    assert test_plate.meta.plate.name == "test_plate"

    with pytest.raises(NgioValueError):
        test_plate.add_image(row="B", column="03", image_path="1", acquisition_id=1)

    test_plate.atomic_add_image(row="C", column="02", image_path="1", acquisition_id=1)

    assert test_plate.columns == ["02", "03"]
    assert test_plate.rows == ["B", "C"]
    assert test_plate.acquisition_ids == [0, 1]
    assert (
        test_plate.get_image_acquisition_id(row="B", column="03", image_path="0") == 0
    )

    assert len(test_plate.wells_paths()) == 2

    test_plate.remove_image(row="C", column="02", image_path="1")
    assert len(test_plate.wells_paths()) == 1


def test_create_and_edit_plate_path_normalization(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    test_plate.add_image(row="B", column="03", image_path="0_mip", acquisition_id=0)
    test_plate.add_image(
        row="B", column="03", image_path="1_illumination_correction", acquisition_id=0
    )
    assert test_plate.images_paths() == ["B/03/0_mip", "B/03/1_illumination_correction"]


def test_derive_plate_from_ome_zarr(cardiomyocyte_tiny_path: Path, tmp_path: Path):
    ome_zarr_plate = open_ome_zarr_plate(cardiomyocyte_tiny_path)
    test_plate = ome_zarr_plate.derive_plate(
        tmp_path / "test_plate.zarr", keep_acquisitions=True
    )
    assert test_plate.columns == ["03"]
    assert test_plate.rows == ["B"]
    assert test_plate.acquisition_ids == [0]


def test_add_well(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    well = test_plate.add_well(row="B", column="03")
    assert isinstance(well, OmeZarrWell)
    assert test_plate.columns == ["03"]
    assert test_plate.rows == ["B"]
    assert test_plate.acquisition_ids == []
    assert test_plate.wells_paths() == ["B/03"]
    assert test_plate.meta.plate.name == "test_plate"

    test_plate.add_column("04")
    test_plate.add_row("C")
    assert test_plate.columns == ["03", "04"]
    assert test_plate.rows == ["B", "C"]
    # No well added in this step
    assert test_plate.wells_paths() == ["B/03"]


def test_add_well_with_acquisition(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")
    test_plate.add_acquisition(acquisition_id=0, acquisition_name="test_acquisition")
    test_plate.add_acquisition(acquisition_id=1, acquisition_name="test_acquisition1")
    assert test_plate.acquisition_ids == [0, 1]
    assert test_plate.acquisitions_names == ["test_acquisition", "test_acquisition1"]


def test_create_plate_with_wells(tmp_path: Path):
    images = [
        ImageInWellPath(row="B", column="03", path="0", acquisition_id=0),
        ImageInWellPath(row="B", column="03", path="1", acquisition_id=1),
        ImageInWellPath(row="C", column="02", path="0"),
    ]

    test_plate = create_empty_plate(
        tmp_path / "test_plate.zarr", name="test_plate", images=images
    )
    assert test_plate.columns == ["02", "03"]
    assert test_plate.rows == ["B", "C"]
    assert test_plate.acquisition_ids == [0, 1]
    assert test_plate.wells_paths() == ["B/03", "C/02"]
    assert test_plate.images_paths() == ["B/03/0", "B/03/1", "C/02/0"]
    assert test_plate.get_well("B", "03").acquisition_ids == [0, 1]
    assert test_plate.get_well("C", "02").acquisition_ids == []
    well = test_plate.get_well("B", "03")
    assert isinstance(well, OmeZarrWell)
    assert well.paths() == ["0", "1"]
    assert well.get_image_acquisition_id("0") == 0
    assert well.get_image_acquisition_id("1") == 1

    store = test_plate.get_image_store(row="B", column="03", image_path="0")
    assert isinstance(store, zarr.Group)


def test_tables_api(tmp_path: Path):
    test_plate = create_empty_plate(tmp_path / "test_plate.zarr", name="test_plate")

    test_df = DataFrame({"a": [1, 2], "b": [3, 4]})
    test_table = GenericTable(test_df)
    test_plate.add_table("test_table", test_table, backend="csv")

    test_roi_table = RoiTable(
        rois=[Roi(name="roi_1", x_length=10, y_length=10, z_length=10)]  # type: ignore
    )
    test_plate.add_table("test_roi_table", test_roi_table)
    assert test_plate.list_tables() == ["test_table", "test_roi_table"]
    assert test_plate.list_roi_tables() == ["test_roi_table"]

    pdt.assert_frame_equal(
        test_plate.get_table("test_table").dataframe,  # type: ignore
        test_df,
        check_names=False,
    )


@pytest.mark.filterwarnings("ignore::anndata._warnings.ImplicitModificationWarning")
def test_plate_table_aggregations(cardiomyocyte_small_mip_path: Path):
    ome_zarr_plate = open_ome_zarr_plate(cardiomyocyte_small_mip_path)
    expected_tables = [
        "FOV_ROI_table",
        "nuclei_ROI_table",
        "well_ROI_table",
        "regionprops_DAPI",
        "nuclei_measurements_wf3",
        "nuclei_measurements_wf4",
        "nuclei_lamin_measurements_wf4",
    ]
    tables = ome_zarr_plate.list_image_tables()
    assert set(tables) == set(expected_tables)
    async_tables = asyncio.run(ome_zarr_plate.list_image_tables_async())
    assert set(async_tables) == set(expected_tables)
    roi_tables = ome_zarr_plate.list_image_tables(filter_types="roi_table")
    assert set(roi_tables) == {"FOV_ROI_table", "well_ROI_table"}

    t1 = ome_zarr_plate.concatenate_image_tables(name="regionprops_DAPI")
    t2 = asyncio.run(
        ome_zarr_plate.concatenate_image_tables_async(name="regionprops_DAPI")
    )
    pdt.assert_frame_equal(t1.dataframe, t2.dataframe)
