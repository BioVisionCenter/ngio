"""Unit tests for the plate/well metadata models in ngio_specs._ngio_hcs.

These focus on the functional API contract: add/remove operations must
return an updated copy and leave the receiver untouched.
"""

from ngio.ome_zarr_meta.ngio_specs import NgioPlateMeta, NgioWellMeta


def test_well_add_image_does_not_mutate_receiver():
    well = NgioWellMeta.default_init()
    well2 = well.add_image(path="0")

    assert well.paths() == []
    assert well2.paths() == ["0"]


def test_well_remove_image_does_not_mutate_receiver():
    well = NgioWellMeta.default_init().add_image(path="0").add_image(path="1")
    well2 = well.remove_image(path="0")

    assert well.paths() == ["0", "1"]
    assert well2.paths() == ["1"]


def test_plate_add_well_does_not_mutate_receiver():
    plate = NgioPlateMeta.default_init()
    plate2 = plate.add_well(row="A", column=1)

    assert plate.wells_paths == []
    assert plate2.wells_paths == ["A/01"]


def test_plate_add_well_existing_row_column_does_not_mutate_receiver():
    # Removing a well keeps its row/column, so re-adding takes the
    # short-circuit path in add_row/add_column.
    plate = NgioPlateMeta.default_init().add_well(row="A", column=1)
    plate_empty = plate.remove_well(row="A", column=1)

    assert plate.wells_paths == ["A/01"]
    assert plate_empty.wells_paths == []

    plate_again = plate_empty.add_well(row="A", column=1)
    assert plate_empty.wells_paths == []
    assert plate_again.wells_paths == ["A/01"]


def test_plate_add_acquisition_does_not_mutate_receiver():
    plate = NgioPlateMeta.default_init().add_acquisition(acquisition_id=0)
    plate2 = plate.add_acquisition(acquisition_id=1)

    assert plate.acquisition_ids == [0]
    assert plate2.acquisition_ids == [0, 1]


def test_plate_remove_well_does_not_mutate_receiver():
    plate = NgioPlateMeta.default_init().add_well(row="A", column=1)
    plate2 = plate.remove_well(row="A", column=1)

    assert plate.wells_paths == ["A/01"]
    assert plate2.wells_paths == []
