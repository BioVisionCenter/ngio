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
