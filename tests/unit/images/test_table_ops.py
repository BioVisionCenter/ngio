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
        pixelsize=0.1,
    )
    for table_name in tables:
        table_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "label": [1, 2, 3]})
        table = FeatureTable(table_data=table_data)
        ome_zarr_container.add_table(table_name, table, backend="json")
    return ome_zarr_container


@pytest.fixture(scope="module")
def sample_ome_zarrs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[OmeZarrContainer, OmeZarrContainer]:
    """Two containers with tables, shared by all (read-only) tests here."""
    tmp_path = tmp_path_factory.mktemp("table_ops")
    ome_zarr_1 = create_sample_ome_zarr(tmp_path, "test1", ["table1", "table2"])
    ome_zarr_2 = create_sample_ome_zarr(tmp_path, "test2", ["table1"])
    return ome_zarr_1, ome_zarr_2


def test_list_sync_api(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

    assert list_image_tables([ome_zarr_1, ome_zarr_2], mode="common") == ["table1"]
    assert list_image_tables([ome_zarr_1, ome_zarr_2], mode="all") == [
        "table1",
        "table2",
    ]


@pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")
def test_list_async_api(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

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
    sample_ome_zarrs, table: str, mode: Literal["eager", "lazy"], strict: bool
):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

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


def test_cat_as_sync(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

    extras1 = {"column1": "value1"}
    extras2 = {"column1": "value2"}

    concatenated_table = concatenate_image_tables_as(
        [ome_zarr_1, ome_zarr_2],
        extras=[extras1, extras2],
        name="table1",
        table_cls=GenericTable,
    )

    assert isinstance(concatenated_table, GenericTable)


def test_set_index(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

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


@pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")
def test_cat_async_api(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

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


@pytest.mark.filterwarnings("ignore::ngio.utils.NgioDeprecationWarning")
def test_cat_eager_lazy_index_parity(sample_ome_zarrs):
    ome_zarr_1, ome_zarr_2 = sample_ome_zarrs

    extras = [{"column1": "value1"}, {"column1": "value2"}]
    eager_df = concatenate_image_tables_as(
        [ome_zarr_1, ome_zarr_2],
        extras=extras,
        name="table1",
        table_cls=GenericTable,
        index_key="Index",
        mode="eager",
    ).dataframe
    lazy_df = concatenate_image_tables_as(
        [ome_zarr_1, ome_zarr_2],
        extras=extras,
        name="table1",
        table_cls=GenericTable,
        index_key="Index",
        mode="lazy",
    ).dataframe

    assert eager_df.index.name == "Index"
    assert lazy_df.index.name == "Index"
    # every row must get a unique index derived from extras + original index
    assert eager_df.index.is_unique
    assert lazy_df.index.is_unique
    assert sorted(eager_df.index) == sorted(lazy_df.index)

    async_lazy_df = asyncio.run(
        concatenate_image_tables_as_async(
            [ome_zarr_1, ome_zarr_2],
            extras=extras,
            name="table1",
            table_cls=GenericTable,
            index_key="Index",
            mode="lazy",
        )
    ).dataframe
    assert sorted(async_lazy_df.index) == sorted(eager_df.index)
