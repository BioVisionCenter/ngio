"""Edge-case coverage tests for `ngio.hcs._plate`."""

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
from ngio.utils import NgioValidationError, NgioValueError


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


@pytest.mark.parametrize("add", ["add_image", "atomic_add_image"])
def test_cached_plate_sees_its_own_add_image(tmp_path: Path, add: str):
    """A `cache=True` plate lists an image it just added, without `refresh()`.

    `clean_cache()` in the add path used to orphan the cached well's handler,
    so `_wells_cache` kept serving the pre-write image list from every
    listing API.
    """
    create_empty_plate(tmp_path / "plate.zarr", name="plate")
    plate = open_ome_zarr_plate(tmp_path / "plate.zarr", cache=True)
    plate.add_image(row="B", column="03", image_path="0")

    assert plate.images_paths() == ["B/03/0"]  # populate the well cache
    getattr(plate, add)(row="B", column="03", image_path="1")

    assert sorted(plate.images_paths()) == ["B/03/0", "B/03/1"]
    assert sorted(plate.get_well("B", "03").paths()) == ["0", "1"]

    # A well added after the cache was populated is visible too.
    plate.add_well(row="C", column="03")
    assert "C/03" in plate.wells_paths()


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
    with pytest.raises(NgioValidationError, match="No tables found"):
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


@pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")
def test_concatenate_image_tables_as(cardiomyocyte_small_mip_path_readonly: Path):
    plate = open_ome_zarr_plate(cardiomyocyte_small_mip_path_readonly, mode="r")
    table = plate.concatenate_image_tables_as(
        name="regionprops_DAPI", table_cls=FeatureTable
    )
    assert isinstance(table, FeatureTable)

    assert "label" in table.dataframe.reset_index().columns


def test_get_well_returns_cached_instance(tmp_path: Path):
    plate = create_empty_plate(tmp_path / "plate.zarr", name="plate", cache=True)
    plate.add_well(row="A", column=1)

    well_1 = plate.get_well(row="A", column=1)
    well_2 = plate.get_well(row="A", column=1)
    assert well_1 is well_2


@pytest.mark.parametrize("max_workers", [None, 1, 4, "auto"])
def test_plate_fan_out_agrees_with_serial(tmp_path: Path, max_workers):
    """Reading the wells concurrently must return exactly the serial answer.

    The fan-out is round-trip bound, so on a remote store it is worth several
    times its serial cost — but only if the results are identical and ordered
    the same way, since callers index into them by position.
    """
    from ngio.ome_zarr_meta import ImageInWellPath

    images = [
        ImageInWellPath(row=row, column=f"{col + 1:02d}", path="0")
        for row in ("A", "B", "C")
        for col in range(4)
    ]
    plate = create_empty_plate(
        tmp_path / "fanout.zarr", name="plate", images=images, overwrite=True
    )

    assert plate.images_paths(max_workers=max_workers) == plate.images_paths()
    assert list(plate.get_wells(max_workers=max_workers)) == list(plate.get_wells())
    assert len(plate.images_paths(max_workers=max_workers)) == len(images)


def test_concurrent_gets_return_the_identical_object(
    cardiomyocyte_tiny_path_readonly: Path,
):
    """The cache must hand every racing thread the same well and image.

    `get_well_images` relies on sharing `_images_cache` with `get_images`,
    and the fan-out builds cache entries from worker threads — so a
    check-then-act insert would quietly hand two threads two objects.
    """
    from concurrent.futures import ThreadPoolExecutor

    plate = open_ome_zarr_plate(cardiomyocyte_tiny_path_readonly, cache=True, mode="r")

    with ThreadPoolExecutor(max_workers=8) as pool:
        wells = list(pool.map(lambda _: plate._get_well("B/03"), range(32)))
        imgs = list(pool.map(lambda _: plate._get_image("B/03/0"), range(32)))

    assert all(w is wells[0] for w in wells)
    assert all(i is imgs[0] for i in imgs)


def test_a_well_at_a_different_version_than_its_plate_still_decodes(tmp_path: Path):
    """The version the plate hands down is a fast path, not a constraint.

    A well rewritten at another NGFF version than its plate used to decode via
    the registry walk; handing the resolved version down must not turn that
    tolerance into a raise.
    """
    import zarr

    from ngio.ome_zarr_meta import ImageInWellPath

    images = [ImageInWellPath(row="A", column="01", path="0")]
    store = tmp_path / "mixed.zarr"
    create_empty_plate(store, name="plate", images=images, overwrite=True)

    # Rewrite the 0.4 well document as 0.5 behind ngio's back.
    well = zarr.open_group(str(store / "A" / "01"), mode="r+")
    well.attrs.clear()
    well.attrs.update({"ome": {"version": "0.5", "well": {"images": [{"path": "0"}]}}})

    plate = open_ome_zarr_plate(store, mode="r")
    assert plate.images_paths() == ["A/01/0"]


def test_a_malformed_well_still_raises_just_later(tmp_path: Path):
    """Well validation is deferred, not dropped.

    `WellMetaHandler` used to read and decode in its constructor purely to
    validate, which a plate walking 384 wells paid 384 times for documents it
    was about to read again. With the version handed down there is nothing left
    to resolve, so the check moved to first use — but it must still happen.
    """
    import zarr

    from ngio.ome_zarr_meta import ImageInWellPath
    from ngio.utils import NgioValidationError

    images = [ImageInWellPath(row="A", column="01", path="0")]
    store = tmp_path / "broken.zarr"
    create_empty_plate(store, name="plate", images=images, overwrite=True)

    # Corrupt the well document behind ngio's back.
    well = zarr.open_group(str(store / "A" / "01"), mode="r+")
    well.attrs.clear()
    well.attrs.update({"well": {"images": "not-a-list"}})

    plate = open_ome_zarr_plate(store, mode="r")
    with pytest.raises(NgioValidationError):
        plate.images_paths()
