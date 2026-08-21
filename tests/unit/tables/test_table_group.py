from pathlib import Path

import pytest
from pandas import DataFrame

from ngio.tables._tables_container import (
    FeatureTable,
    TablesContainer,
    open_tables_container,
)
from ngio.utils import NgioFileNotFoundError, NgioValueError


def test_table_container(tmp_path: Path):
    table_group = open_tables_container(tmp_path / "test.zarr", mode="a")
    assert isinstance(table_group, TablesContainer)
    assert table_group.list() == []

    # Create a feature table
    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [1.0, 1.3, 0.0]})
    )
    table_group.add(name="feat_table", table=table)
    assert table_group.list() == ["feat_table"]

    with pytest.raises(NgioValueError):
        table_group.add(name="feat_table", table=table)

    table = table_group.get("feat_table")
    assert isinstance(table, FeatureTable)

    expected = DataFrame({"label": [1, 2, 3], "a": [1.0, 1.3, 0.0]})
    expected = expected.set_index("label")
    assert table.dataframe.equals(expected)


def test_a_stale_table_name_does_not_break_typed_listing(tmp_path: Path):
    """A name in the `tables` attribute with no group behind it is tolerated.

    Another writer (or a crashed one) can leave a dangling entry. A typed
    listing must still return the tables that do exist; only a direct `get`
    of the stale name reports the problem.
    """
    table_group = open_tables_container(tmp_path / "test.zarr", mode="a")
    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [1.0, 1.3, 0.0]})
    )
    table_group.add(name="feat_table", table=table)

    # Dangle a name behind the container's back.
    handler = table_group._group_handler
    handler.write_attrs({"tables": ["feat_table", "stale"]})

    assert table_group.list() == ["feat_table", "stale"]
    assert table_group.list(filter_types="feature_table") == ["feat_table"]
    with pytest.raises(NgioFileNotFoundError):
        table_group.get("stale")


def test_uncached_typed_listing_sees_a_type_change(tmp_path: Path):
    """Under `cache=False` the type memo must not outlive the table."""
    from ngio.tables import GenericTable

    store = tmp_path / "test.zarr"
    reader = open_tables_container(store, cache=False, mode="a")
    writer = open_tables_container(store, mode="r+")

    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [1.0, 1.3, 0.0]})
    )
    writer.add(name="t", table=table)
    assert reader.list(filter_types="feature_table") == ["t"]

    writer.delete("t")
    writer.add(name="t", table=GenericTable(table_data=DataFrame({"a": [1.0]})))
    assert reader.list(filter_types="feature_table") == []
    assert reader.list(filter_types="generic_table") == ["t"]


def test_add_explicit_backend_overrides(tmp_path: Path):
    """An explicit backend argument still converts the table."""
    src_group = open_tables_container(tmp_path / "src.zarr", mode="a")
    dst_group = open_tables_container(tmp_path / "dst.zarr", mode="a")

    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [4.0, 5.0, 6.0]})
    )
    src_group.add(name="table", table=table, backend="parquet")

    loaded_table = src_group.get("table")
    dst_group.add(name="table", table=loaded_table, backend="anndata")
    assert dst_group.get("table").backend_name == "anndata"


def test_add_in_memory_table_uses_default_backend(tmp_path: Path):
    """In-memory tables are still written with the default backend."""
    table_group = open_tables_container(tmp_path / "test.zarr", mode="a")

    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [4.0, 5.0, 6.0]})
    )
    assert table.backend_name == "anndata_v1"

    table_group.add(name="table", table=table)
    assert table_group.get("table").backend_name == "anndata_v1"

    # Explicitly passing backend=None is equivalent to the default
    table_group.add(name="table_none", table=table, backend=None)
    assert table_group.get("table_none").backend_name == "anndata_v1"
