from pathlib import Path

import pytest
import zarr

from ngio import create_empty_well, open_ome_zarr_well
from ngio.utils import NgioValueError


def test_open_real_ome_zarr_well(cardiomyocyte_tiny_path: Path):
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03"
    ome_zarr_well = open_ome_zarr_well(cardiomyocyte_tiny_path)
    assert isinstance(ome_zarr_well.__repr__(), str)
    assert ome_zarr_well.paths() == ["0"]
    assert ome_zarr_well.acquisition_ids == []


def test_create_and_edit_well(tmp_path: Path):
    test_well = create_empty_well(tmp_path / "test_well.zarr")
    assert test_well.paths() == []
    assert test_well.acquisition_ids == []

    test_well.add_image(image_path="0", acquisition_id=0, strict=False)
    test_well.add_image(image_path="1", acquisition_id=0, strict=True)

    with pytest.raises(NgioValueError):
        test_well.add_image(image_path="1", acquisition_id=1)

    test_well.atomic_add_image(image_path="2", acquisition_id=1, strict=False)
    assert len(test_well.paths()) == 3
    assert test_well.acquisition_ids == [0, 1], test_well.acquisition_ids

    store = test_well.get_image_store("0")
    assert isinstance(store, zarr.Group)
