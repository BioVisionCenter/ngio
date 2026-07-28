from pathlib import Path

import pytest
from pandas import DataFrame

from ngio.tables._tables_container import (
    FeatureTable,
    TablesContainer,
    open_tables_container,
)
from ngio.utils import NgioValueError


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


@pytest.mark.parametrize("backend", ["anndata", "json", "csv", "parquet"])
def test_add_preserves_table_backend(tmp_path: Path, backend: str):
    """Re-adding a loaded table without an explicit backend keeps its backend.

    Testing for #207.
    """
    src_group = open_tables_container(tmp_path / "src.zarr", mode="a")
    dst_group = open_tables_container(tmp_path / "dst.zarr", mode="a")

    table = FeatureTable(
        table_data=DataFrame({"label": [1, 2, 3], "a": [4.0, 5.0, 6.0]})
    )
    src_group.add(name="table", table=table, backend=backend)

    loaded_table = src_group.get("table")
    assert loaded_table.backend_name == backend

    dst_group.add(name="table", table=loaded_table)
    copied_table = dst_group.get("table")
    assert copied_table.backend_name == backend
    assert copied_table.meta.backend == backend


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
