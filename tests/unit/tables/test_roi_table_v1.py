from pathlib import Path

import pandas as pd
import pytest

from ngio.common import Roi
from ngio.tables import RoiTable
from ngio.tables._tables_container import open_table, write_table
from ngio.tables.backends import AnnDataBackend
from ngio.tables.v1 import RoiTableV1
from ngio.tables.v1._roi_table import RoiTableV1Meta
from ngio.utils import NgioValueError, ZarrGroupHandler


def test_roi_table_v1(tmp_path: Path):
    rois = [
        Roi(
            name="roi1",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        )
    ]

    table = RoiTableV1(rois=rois)
    assert isinstance(table.__repr__(), str)

    table.add(
        roi=Roi(
            name="roi2",
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
        # ROI name already exists
        table.add(
            roi=Roi(
                name="roi2",
                x=0.0,
                y=0.0,
                z=0.0,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                unit="micrometer",  # type: ignore
            )
        )

    table.add(
        roi=Roi(
            name="roi2",
            x=0.0,
            y=0.0,
            z=0.0,
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
            unit="micrometer",  # type: ignore
        ),
        overwrite=True,
    )
    assert len(table.rois()) == 2
    write_table(store=tmp_path / "roi_table.zarr", table=table, backend="anndata")

    loaded_table = open_table(store=tmp_path / "roi_table.zarr")
    assert isinstance(loaded_table, RoiTableV1)
    assert len(loaded_table.rois()) == 2
    assert loaded_table.get("roi1") == table.get("roi1")
    assert loaded_table.get("roi2") == table.get("roi2")

    with pytest.raises(NgioValueError):
        loaded_table.get("roi3")

    assert loaded_table.meta.backend == "anndata"
    meta_dict = loaded_table._meta.model_dump()
    assert meta_dict.get("table_version") == loaded_table.version()
    assert meta_dict.get("type") == loaded_table.table_type()


def test_roi_no_index(tmp_path: Path):
    """ngio needs to support reading a table without an index. for legacy reasons"""
    handler = ZarrGroupHandler(tmp_path / "roi_table.zarr")
    backend = AnnDataBackend()
    backend.set_group_handler(handler)

    roi_table = pd.DataFrame(
        {
            "x_micrometer": [0.0, 1.0],
            "y_micrometer": [0.0, 1.0],
            "z_micrometer": [0.0, 1.0],
            "len_x_micrometer": [1.0, 1.0],
            "len_y_micrometer": [1.0, 1.0],
            "len_z_micrometer": [1.0, 1.0],
        }
    )
    roi_table.index = pd.Index(["roi_1", "roi_2"])

    backend.write(
        roi_table,
        metadata=RoiTableV1Meta().model_dump(exclude_none=True),
    )

    roi_table = RoiTable.from_handler(handler=handler)
    assert isinstance(roi_table, RoiTable)
    assert len(roi_table.rois()) == 2
