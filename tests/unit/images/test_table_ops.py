import asyncio
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from ngio import OmeZarrContainer, create_empty_ome_zarr
from ngio.images import (
    concatenate_image_tables,
    concatenate_image_tables_as,
    concatenate_image_tables_as_async,
    concatenate_image_tables_async,
    list_image_tables,
    list_image_tables_async,
)
from ngio.tables import FeatureTable, GenericTable


def create_sample_ome_zarr(
    tmp_path: Path, name: str, tables: list[str]
) -> OmeZarrContainer:
    store = tmp_path / f"{name}.zarr"
    ome_zarr_container = create_empty_ome_zarr(
        store=store,
        shape=(32, 32),
        xy_pixelsize=0.1,
    )
    for table_name in tables:
        table_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "label": [1, 2, 3]})
        table = FeatureTable(table_data=table_data)
        ome_zarr_container.add_table(table_name, table, backend="json")
    return ome_zarr_container


def test_list_sync_api(tmp_path: Path):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    assert list_image_tables([ome_zarr_1, ome_zarr_2], mode="common") == ["table1"]
    assert list_image_tables([ome_zarr_1, ome_zarr_2], mode="all") == [
        "table1",
        "table2",
    ]


def test_list_async_api(tmp_path: Path):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    assert asyncio.run(
        list_image_tables_async([ome_zarr_1, ome_zarr_2], mode="common")
    ) == ["table1"]
    assert asyncio.run(
        list_image_tables_async([ome_zarr_1, ome_zarr_2], mode="all")
    ) == [
        "table1",
        "table2",
    ]


@pytest.mark.parametrize(
    "table, mode, strict",
    [
        ("table1", "eager", True),
        ("table1", "lazy", True),
        ("table2", "eager", False),
        ("table2", "eager", True),
    ],
)
def test_cat_sync_api(
    tmp_path: Path, table: str, mode: Literal["eager", "lazy"], strict: bool
):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    extras1 = {"column1": "value1"}
    extras2 = {"column1": "value2"}
    if strict and table == "table2":
        with pytest.raises(ValueError):
            concatenate_image_tables(
                [ome_zarr_1, ome_zarr_2],
                extras=[extras1, extras2],
                name=table,
                mode=mode,
                strict=strict,
            )
        return None

    concatenated_table = concatenate_image_tables(
        [ome_zarr_1, ome_zarr_2],
        extras=[extras1, extras2],
        name=table,
        mode=mode,
        strict=strict,
    )
    assert isinstance(concatenated_table, FeatureTable)

    df = concatenated_table.dataframe
    df = df.reset_index()
    assert set(df.columns) == {"x", "y", "label", "column1"}
    if "table2" in table:
        assert df.shape == (3, 4), df.shape
    else:
        assert df.shape == (6, 4), df.shape


def test_cat_as_sync(tmp_path: Path):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    extras1 = {"column1": "value1"}
    extras2 = {"column1": "value2"}

    concatenated_table = concatenate_image_tables_as(
        [ome_zarr_1, ome_zarr_2],
        extras=[extras1, extras2],
        name="table1",
        table_cls=GenericTable,
    )

    assert isinstance(concatenated_table, GenericTable)


def test_set_index(tmp_path: Path):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    extras1 = {"column1": "value1"}
    extras2 = {"column1": "value2"}

    concatenated_table = concatenate_image_tables_as(
        [ome_zarr_1, ome_zarr_2],
        extras=[extras1, extras2],
        name="table1",
        table_cls=GenericTable,
        index_key="Index",
    )
    df = concatenated_table.dataframe
    assert set(df.columns) == {"x", "y", "label", "column1"}
    assert df.index.name == "Index"


def test_cat_async_api(tmp_path: Path):
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])

    extras1 = {"column1": "value1"}
    extras2 = {"column1": "value2"}

    concatenated_table = asyncio.run(
        concatenate_image_tables_async(
            [ome_zarr_1, ome_zarr_2],
            extras=[extras1, extras2],
            name="table1",
        )
    )
    assert isinstance(concatenated_table, FeatureTable)

    df = concatenated_table.dataframe
    df = df.reset_index()
    assert set(df.columns) == {"x", "y", "label", "column1"}
    assert df.shape == (6, 4), df.shape

    concatenate_table = asyncio.run(
        concatenate_image_tables_as_async(
            [ome_zarr_1, ome_zarr_2],
            extras=[extras1, extras2],
            name="table1",
            table_cls=GenericTable,
        )
    )
    assert isinstance(concatenate_table, GenericTable)
