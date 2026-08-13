"""Edge-case coverage tests for `ngio.tables.v1._roi_table`."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from ngio.common import Roi
from ngio.tables._tables_container import open_table, write_table
from ngio.tables.v1._roi_table import (
    GenericRoiTableV1,
    MaskingRoiTableV1,
    MaskingRoiTableV1Meta,
    RoiDictWrapper,
    RoiTableV1,
    RoiTableV1Meta,
)
from ngio.utils import NgioTableValidationError, NgioValueError


def _make_roi(name: str, label: int | None = None, **extras) -> Roi:
    return Roi.from_values(
        name=name,
        slices={"x": (0, 10), "y": (0, 10), "z": (0, 10)},
        label=label,
        **extras,
    )


def _valid_roi_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x_micrometer": [0.0, 1.0],
            "y_micrometer": [0.0, 1.0],
            "z_micrometer": [0.0, 1.0],
            "len_x_micrometer": [1.0, 1.0],
            "len_y_micrometer": [1.0, 1.0],
            "len_z_micrometer": [1.0, 1.0],
        }
    )
    df.index = pd.Index(["roi_1", "roi_2"], name="FieldIndex")
    return df


def test_set_table_data_missing_required_columns():
    table = RoiTableV1()
    df = _valid_roi_df().drop(columns=["len_x_micrometer", "len_y_micrometer"])
    with pytest.raises(NgioTableValidationError):
        table.set_table_data(df)


def test_set_table_data_invalid_type():
    table = RoiTableV1()
    with pytest.raises(NgioValueError):
        table.set_table_data([1, 2, 3])  # ty: ignore[invalid-argument-type]


def test_set_table_data_valid_dataframe():
    table = RoiTableV1()
    table.set_table_data(_valid_roi_df())
    assert len(table.rois()) == 2
    assert table.get("roi_1").name == "roi_1"


def test_set_table_data_refresh_from_backend(tmp_path: Path):
    table = RoiTableV1(rois=[_make_roi("roi_1"), _make_roi("roi_2")])
    write_table(store=tmp_path / "roi.zarr", table=table, backend="anndata")

    loaded = open_table(store=tmp_path / "roi.zarr")
    assert isinstance(loaded, RoiTableV1)
    assert len(loaded.rois()) == 2

    # Reload the in-memory state from the backend
    loaded.set_table_data(refresh=True)
    assert len(loaded.rois()) == 2
    assert loaded.get("roi_1").name == "roi_1"


def test_serialization_requires_x_slice():
    roi = Roi.from_values(name="roi", slices={"y": (0, 10), "z": (0, 10)})
    with pytest.raises(NgioValueError, match="missing 'x' slice"):
        RoiTableV1(rois=[roi])


def test_serialization_requires_y_slice():
    roi = Roi.from_values(name="roi", slices={"x": (0, 10), "z": (0, 10)})
    with pytest.raises(NgioValueError, match="missing 'y' slice"):
        RoiTableV1(rois=[roi])


def test_label_written_as_column_when_index_is_not_label():
    # RoiTableV1 uses "FieldIndex" as index, so the label must be
    # serialized as a dedicated "label" column.
    table = RoiTableV1(rois=[_make_roi("roi_1", label=5)])
    df = table.dataframe
    assert "label" in df.columns
    assert df.loc["roi_1", "label"] == 5


def test_extra_columns_roundtrip_and_unknown_column_warning(caplog):
    roi = _make_roi("roi_1", plate_name="plate_a", not_a_known_column="foo")
    with caplog.at_level(logging.WARNING):
        table = RoiTableV1(rois=[roi])
        df = table.dataframe
    assert df.loc["roi_1", "plate_name"] == "plate_a"
    assert df.loc["roi_1", "not_a_known_column"] == "foo"
    # Unknown (non-optional) columns are reported via a logger warning
    assert "not_a_known_column" in caplog.text
    assert "is not in the optional columns" in caplog.text


def test_read_unknown_extra_column_warning(caplog):
    df = _valid_roi_df()
    df["strange_extra"] = ["a", "b"]
    table = RoiTableV1()
    with caplog.at_level(logging.WARNING):
        table.set_table_data(df)
    assert "strange_extra" in caplog.text
    rois = table.rois()
    assert len(rois) == 2


def test_duplicate_roi_names_are_deduplicated():
    # Two ROIs with the same name: the second one is stored under a
    # uuid-suffixed key so both survive.
    wrapper = RoiDictWrapper([_make_roi("roi"), _make_roi("roi")])
    assert len(wrapper.to_list()) == 2
    assert wrapper.get_by_name("roi") is not None

    table = RoiTableV1(rois=[_make_roi("roi"), _make_roi("roi")])
    assert len(table.rois()) == 2


def test_roi_dict_wrapper_add_single_roi():
    wrapper = RoiDictWrapper([])
    wrapper.add_rois(_make_roi("roi_1", label=3))
    assert len(wrapper.to_list()) == 1
    assert wrapper.get_by_name("roi_1") is not None
    assert wrapper.get_by_label(3) is not None
    assert wrapper.get_by_label(99) is None


def test_generic_roi_table_type():
    assert GenericRoiTableV1.table_type() == "generic_roi_table"


def test_from_table_data():
    table = RoiTableV1.from_table_data(_valid_roi_df(), meta=RoiTableV1Meta())
    assert isinstance(table, RoiTableV1)
    assert len(table.rois()) == 2
    assert table.get("roi_2").name == "roi_2"


def test_roi_table_meta_none_defaults():
    meta = RoiTableV1Meta(index_key=None, index_type=None)
    table = RoiTableV1(meta=meta)
    assert table.meta.index_key == "FieldIndex"
    assert table.meta.index_type == "str"


def test_masking_roi_table_meta_none_defaults():
    meta = MaskingRoiTableV1Meta(index_key=None, index_type=None)
    table = MaskingRoiTableV1(meta=meta)
    assert table.meta.index_key == "label"
    assert table.meta.index_type == "int"
    assert table.meta.instance_key == "label"


def test_masking_roi_table_repr_without_reference_label():
    table = MaskingRoiTableV1(rois=[_make_roi("1", label=1)])
    assert table.reference_label is None
    assert repr(table) == "MaskingRoiTableV1(num_rois=1)"


def test_roi_without_z_and_with_time_roundtrip():
    # No z slice: serialized with defaults z=0.0, len_z=1.0.
    # A t slice is serialized to t_second / len_t_second and read back.
    roi = Roi.from_values(
        name="roi_1",
        slices={"x": (0, 10), "y": (0, 10), "t": (2, 5)},
    )
    table = RoiTableV1(rois=[roi])
    df = table.dataframe
    assert df.loc["roi_1", "z_micrometer"] == 0.0
    assert df.loc["roi_1", "len_z_micrometer"] == 1.0
    assert df.loc["roi_1", "t_second"] == 2.0
    assert df.loc["roi_1", "len_t_second"] == 5.0

    reloaded = RoiTableV1()
    reloaded.set_table_data(df)
    t_slice = reloaded.get("roi_1").get("t")
    assert t_slice is not None
    assert t_slice.start == 2.0
    assert t_slice.length == 5.0


def test_masking_roi_table_label_from_index():
    df = _valid_roi_df()
    df.index = pd.Index([10, 20], name="label")
    table = MaskingRoiTableV1.from_table_data(df, meta=MaskingRoiTableV1Meta())
    assert isinstance(table, MaskingRoiTableV1)
    assert table.get_label(10).label == 10
    assert table.get_label(20).label == 20


def test_masking_roi_table_get_label():
    table = MaskingRoiTableV1(rois=[])
    table.add(_make_roi("1", label=1))
    roi = table.get_label(1)
    assert roi.label == 1
    with pytest.raises(NgioValueError, match="label 2 not found"):
        table.get_label(2)


def test_empty_roi_table_without_backend_is_usable():
    table = RoiTableV1()
    assert table.rois() == []

    table.add(_make_roi("r", label=1))
    assert [roi.name for roi in table.rois()] == ["r"]


def test_table_data_is_rebuilt_only_after_add():
    """Repeated reads serve the same DataFrame; `add()` invalidates it.

    The rebuild iterates every ROI into a fresh DataFrame, and it used to run
    on every `table_data`/`dataframe` access — for a masking table with tens
    of thousands of labels that is hundreds of milliseconds per property read.
    """
    table = RoiTableV1(rois=[_make_roi("a"), _make_roi("b")])

    first = table.table_data
    assert table.table_data is first

    table.add(_make_roi("c"))
    rebuilt = table.table_data
    assert rebuilt is not first
    assert isinstance(rebuilt, pd.DataFrame) and len(rebuilt) == 3
    assert table.table_data is rebuilt


def test_duplicate_roi_names_survive_roundtrip(tmp_path: Path):
    table = RoiTableV1(rois=[_make_roi("roi"), _make_roi("roi")])
    names = [roi.name for roi in table.rois()]
    assert len(set(names)) == 2

    store = tmp_path / "rois.zarr"
    write_table(store=store, table=table)
    reloaded = open_table(store=store)
    reloaded_names = [roi.name for roi in reloaded.rois()]  # ty: ignore[unresolved-attribute]
    assert len(set(reloaded_names)) == 2
