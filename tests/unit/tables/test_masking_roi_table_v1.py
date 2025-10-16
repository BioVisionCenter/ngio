from pathlib import Path

import pytest

from ngio.tables._tables_container import open_table, write_table
from ngio.tables.v1._roi_table import MaskingRoiTableV1, Roi
from ngio.utils import NgioValueError


def test_masking_roi_table_v1(tmp_path: Path):
    rois = {
        1: Roi(
            name="1",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        )
    }

    table = MaskingRoiTableV1(rois=rois.values(), reference_label="label")
    assert isinstance(table.__repr__(), str)
    assert table.reference_label == "label"
    assert table.meta.region is not None
    assert table.meta.region.path == "../labels/label"

    table.add(
        roi=Roi(
            name="2",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        )
    )

    with pytest.raises(NgioValueError):
        table.add(
            roi=Roi(
                name="2",
                x=0.0,
                y=0.0,
                z=0.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            )
        )

    write_table(store=tmp_path / "roi_table.zarr", table=table, backend="anndata")

    loaded_table = open_table(store=tmp_path / "roi_table.zarr")
    assert isinstance(loaded_table, MaskingRoiTableV1)

    assert loaded_table.meta.backend == "anndata"
    meta_dict = loaded_table._meta.model_dump()
    assert meta_dict.get("table_version") == loaded_table.version()
    assert meta_dict.get("type") == loaded_table.table_type()
