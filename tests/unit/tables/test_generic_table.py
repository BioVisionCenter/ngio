from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from ngio.tables._tables_container import open_table, write_table
from ngio.tables.backends import CsvTableBackend
from ngio.tables.v1 import GenericTable
from ngio.utils import NgioValueError


@pytest.mark.parametrize("backend", ["json", "anndata"])
def test_generic_df_table(tmp_path: Path, backend: str):
    store = tmp_path / "test.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table = GenericTable(table_data=test_df)
    assert isinstance(table.__repr__(), str)

    write_table(store=store, table=table, backend=backend)

    loaded_table = open_table(store=store)
    assert isinstance(loaded_table, GenericTable)
    assert table.backend_name == backend

    assert set(loaded_table.dataframe.columns) == {"a", "b"}
    for column in loaded_table.dataframe.columns:
        pd.testing.assert_series_equal(
            loaded_table.dataframe[column], test_df[column], check_index=False
        )

    loaded_table.load_as_pandas_df()
    loaded_table.load_as_polars_lf()
    loaded_table.load_as_anndata()


@pytest.mark.parametrize(
    "src_backend, dst_backend",
    # anndata -> json is unsupported (json cannot serialize an AnnData object);
    # the other combinations exercise both lazy-load paths.
    [("json", "json"), ("json", "anndata"), ("anndata", "anndata")],
)
def test_write_table_from_opened_table(
    tmp_path: Path, src_backend: str, dst_backend: str
):
    """write_table on a lazily-opened table must copy its data, not write empty.

    Mirrors a table copy done via the low-level write_table API: open a table from
    one store (its data is loaded lazily) and write it to another store.
    """
    src_store = tmp_path / "src.zarr"
    dst_store = tmp_path / "dst.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    write_table(
        store=src_store, table=GenericTable(table_data=test_df), backend=src_backend
    )

    # Freshly opened -> table_data is loaded lazily (not yet materialized).
    loaded_table = open_table(store=src_store)
    write_table(store=dst_store, table=loaded_table, backend=dst_backend)

    copied = open_table(store=dst_store).dataframe
    assert set(copied.columns) == {"a", "b"}
    for column in ["a", "b"]:
        pd.testing.assert_series_equal(
            copied[column], test_df[column], check_index=False, check_dtype=False
        )


def test_set_backend_preference_without_handler(tmp_path: Path):
    """A backend preference can be declared before the table is stored anywhere."""
    store = tmp_path / "test.zarr"
    table = GenericTable(table_data=pd.DataFrame({"a": [1, 2, 3]}))
    # In-memory tables report the default backend from their metadata
    assert table.backend_name == "anndata_v1"

    table.set_backend(backend="csv")
    assert table.backend_name == "csv"
    assert table.meta.backend == "csv"

    # Aliases are normalized to canonical names
    table.set_backend(backend="experimental_parquet_v1")
    assert table.backend_name == "parquet"

    # No handler and no backend is a no-op
    table.set_backend()
    assert table.backend_name == "parquet"

    with pytest.raises(NgioValueError):
        table.set_backend(backend="non_existent_backend")

    # A backend instance cannot be attached without a handler
    with pytest.raises(NgioValueError):
        table.set_backend(backend=CsvTableBackend())

    # The declared preference is used when the table is written
    write_table(store=store, table=table)
    loaded_table = open_table(store=store)
    assert loaded_table.backend_name == "parquet"


@pytest.mark.parametrize("backend", ["anndata", "json", "csv", "parquet"])
def test_write_table_preserves_backend(tmp_path: Path, backend: str):
    """Copying an opened table without an explicit backend keeps its backend.

    Testing for #207.
    """
    src_store = tmp_path / "src.zarr"
    dst_store = tmp_path / "dst.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    write_table(
        store=src_store, table=GenericTable(table_data=test_df), backend=backend
    )

    loaded_table = open_table(store=src_store)
    write_table(store=dst_store, table=loaded_table)

    copied_table = open_table(store=dst_store)
    assert copied_table.backend_name == backend
    assert copied_table.meta.backend == backend


@pytest.mark.parametrize("backend", ["anndata"])
def test_generic_anndata_table(tmp_path: Path, backend: str):
    store = tmp_path / "test.zarr"
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    test_obs = pd.DataFrame({"c": ["a", "b", "c"]})
    test_obs.index = test_obs.index.astype(str)
    test_obsm = np.random.normal(0, 1, size=(3, 2))

    anndata = AnnData(X=test_df, obs=test_obs)
    anndata.obsm["test"] = test_obsm

    table = GenericTable(table_data=anndata)

    assert isinstance(table.table_data, AnnData)

    write_table(store=store, table=table, backend=backend)

    loaded_table = open_table(store=store)
    assert isinstance(loaded_table, GenericTable)

    loaded_ad = loaded_table.load_as_anndata()
    loaded_df = loaded_table.dataframe
    assert set(loaded_df.columns) == {"a", "b", "c"}

    np.testing.assert_allclose(loaded_ad.obsm["test"], test_obsm)  # type: ignore
