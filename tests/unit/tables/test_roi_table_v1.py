from pathlib import Path

import pandas as pd
import pytest

from ngio.common import Roi
from ngio.tables import GenericRoiTable, RoiTable
from ngio.tables._tables_container import open_table, open_table_as, write_table
from ngio.tables.backends import AnnDataBackend
from ngio.tables.v1 import RoiTableV1
from ngio.tables.v1._roi_table import RoiTableV1Meta
from ngio.utils import NgioValueError, ZarrGroupHandler


def test_roi_table_v1(tmp_path: Path):
    rois = [
        Roi.from_values(
            name="roi1",
            slices={"x": slice(0, 1), "y": slice(0, 1), "z": slice(0, 1)},
        )
    ]

    table = RoiTableV1(rois=rois)
    assert isinstance(table.__repr__(), str)

    table.add(
        roi=Roi.from_values(
            name="roi2",
            slices={"x": slice(0, 1), "y": slice(0, 1), "z": slice(0, 1)},
        )
    )

    with pytest.raises(NgioValueError):
        # ROI name already exists
        table.add(
            roi=Roi.from_values(
                name="roi2",
                slices={"x": slice(0, 1), "y": slice(0, 1), "z": slice(0, 1)},
            )
        )

    table.add(
        roi=Roi.from_values(
            name="roi2",
            slices={"x": slice(0, 1), "y": slice(0, 1), "z": slice(0, 1)},
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


def test_generic_roi_table_is_constructible_and_loadable(tmp_path: Path):
    """`GenericRoiTable` is a public export: it must construct and load.

    `from_handler` was left abstract with a `pass` body and never overridden,
    so the class could not be instantiated and — because `abstractmethod` only
    guards instantiation, not classmethod calls — `open_table_as`/`get_as`
    silently returned `None` instead of a table.
    """
    rois = [
        Roi.from_values(
            name="roi1",
            slices={"x": slice(0, 2), "y": slice(0, 2), "z": slice(0, 1)},
        ),
        Roi.from_values(
            name="roi2",
            slices={"x": slice(2, 4), "y": slice(2, 4), "z": slice(0, 1)},
        ),
    ]
    table = GenericRoiTable(rois=rois)
    write_table(store=tmp_path / "generic.zarr", table=table, backend="anndata")

    loaded = open_table_as(store=tmp_path / "generic.zarr", table_cls=GenericRoiTable)
    assert isinstance(loaded, GenericRoiTable)
    assert loaded.get("roi1") == table.get("roi1")
    assert loaded.get("roi2") == table.get("roi2")

    # The written attrs carry the type, and the registry resolves it: the
    # typed `open_table` (strict path) returns a GenericRoiTable, not the
    # untyped GenericTable fallback.
    handler = ZarrGroupHandler(tmp_path / "generic.zarr")
    assert handler.load_attrs()["type"] == "generic_roi_table"
    reopened = open_table(store=tmp_path / "generic.zarr")
    assert isinstance(reopened, GenericRoiTable)
    assert reopened.get("roi1") == table.get("roi1")


def test_generic_roi_table_opens_a_foreign_table_without_index_attrs(tmp_path: Path):
    """The lax reader must not impose its own index on a foreign ROI table.

    A ROI-typed table written by another tool can carry no `index_key` attrs
    (only type/region/instance_key); the meta's index fields default to `None`
    so the stored index is used as-is instead of failing to find `FieldIndex`.
    """
    import zarr

    from ngio.tables import MaskingRoiTable

    store = tmp_path / "foreign.zarr"
    table = MaskingRoiTable(
        rois=[
            Roi.from_values(
                name="1",
                slices={"x": slice(0, 4), "y": slice(0, 4), "z": slice(0, 1)},
                label=1,
            )
        ]
    )
    write_table(store=store, table=table, backend="anndata")
    # Simulate the foreign writer: the type survives, the index attrs do not.
    group = zarr.open_group(store, mode="r+")
    attrs = dict(group.attrs)
    attrs.pop("index_key", None)
    attrs.pop("index_type", None)
    group.attrs.put(attrs)

    loaded = open_table_as(store=store, table_cls=GenericRoiTable)
    assert [roi.label for roi in loaded.rois()] == [1]


def test_foreign_table_add_then_consolidate_does_not_destroy_the_store(
    tmp_path: Path,
):
    """`add()` + `consolidate()` on a lax-opened foreign table round-trips.

    With `index_key=None` the dirty rebuild produced a literal `None`-named
    column, and the failed anndata write truncated the group before raising —
    the original table was destroyed on disk.
    """
    import zarr

    from ngio.tables import MaskingRoiTable

    store = tmp_path / "foreign.zarr"
    table = MaskingRoiTable(
        rois=[
            Roi.from_values(
                name="1",
                slices={"x": slice(0, 4), "y": slice(0, 4), "z": slice(0, 1)},
                label=1,
            )
        ]
    )
    write_table(store=store, table=table, backend="anndata")
    group = zarr.open_group(store, mode="r+")
    attrs = dict(group.attrs)
    attrs.pop("index_key", None)
    attrs.pop("index_type", None)
    group.attrs.put(attrs)

    loaded = open_table_as(store=store, table_cls=GenericRoiTable)
    loaded.add(
        Roi.from_values(
            name="2",
            slices={"x": slice(4, 8), "y": slice(4, 8), "z": slice(0, 1)},
            label=2,
        )
    )
    loaded.consolidate()

    reopened = open_table_as(store=store, table_cls=GenericRoiTable)
    assert sorted(roi.name for roi in reopened.rois() if roi.name) == ["1", "2"]
    # The foreign table's own index name survives the rebuild.
    assert reopened.dataframe.index.name == "label"
