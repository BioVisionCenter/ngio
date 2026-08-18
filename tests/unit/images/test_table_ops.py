from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from ngio import OmeZarrContainer, create_empty_ome_zarr
from ngio.images import (
    concatenate_image_tables,
    concatenate_image_tables_as,
    list_image_tables,
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

