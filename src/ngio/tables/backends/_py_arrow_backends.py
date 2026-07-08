from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.dataset as pa_ds
import pyarrow.fs as pa_fs
import pyarrow.parquet as pa_parquet
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._utils import normalize_pandas_df, normalize_polars_lf
from ngio.utils import NgioStore, NgioValueError, retry_io


class PyArrowBackend(AbstractTableBackend):
    """A class to load and write small tables in CSV format."""

    def __init__(
        self,
        table_name: str,
        table_format: Literal["csv", "parquet"] = "parquet",
    ):
        self.table_name = table_name
        self.table_format = table_format

    @staticmethod
    def implements_anndata() -> bool:
        """Whether the handler implements the anndata protocol."""
        return False

    @staticmethod
    def implements_pandas() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    @staticmethod
    def implements_polars() -> bool:
        """Whether the handler implements the polars protocol."""
        return True

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        raise NotImplementedError(
            "The backend_name method must be implemented in the subclass."
        )

    def _raise_store_type_not_supported(self):
        """Raise an error for unsupported store types."""
        ext = self.table_name.split(".")[-1]
        store = self._group_handler.store
        raise NgioValueError(
            f"Ngio does not support reading a {ext} table from a "
            f"store of type {type(store)}. "
            "Please make sure to use a compatible "
            "store like a LocalStore, or "
            "FsspecStore, or MemoryStore, or ZipStore."
        )

    def _load_from_local_store(self, store: NgioStore, path: str) -> pa_ds.Dataset:
        """Load the table from a directory store."""
        table_path = f"{store.full_url()}/{path}/{self.table_name}"
        dataset = pa_ds.dataset(table_path, format=self.table_format)
        return dataset

    def _load_from_fsspec_store(self, store: NgioStore, path: str) -> pa_ds.Dataset:
        """Load the table from an FS store."""
        fs, table_path = store.sync_fs_and_path(f"{path}/{self.table_name}")
        dataset = pa_ds.dataset(table_path, format=self.table_format, filesystem=fs)
        return dataset

    def _load_from_in_memory_store(self, store: NgioStore, path: str) -> pa_ds.Dataset:
        """Load the table from an in-memory store."""
        table_path = f"{path}/{self.table_name}"
        table = store.memory_dict.get(table_path, None)
        if table is None:
            raise NgioValueError(
                f"Table {self.table_name} not found in the in-memory store at "
                f"path {path}."
            )
        assert isinstance(table, pa.Table)
        dataset = pa_ds.dataset(table)
        return dataset

    def _load_from_zip_store(self, store: NgioStore, path: str) -> pa_ds.Dataset:
        """Load the table from a zip store."""
        raise NotImplementedError("Zip store loading is not implemented yet.")

    @retry_io
    def _load_pyarrow_dataset(self) -> pa_ds.Dataset:
        """Load the table as a pyarrow Dataset.

        This IO bypasses the zarr store, so the `io_retry` policy is applied
        here. Retry covers the eager dataset/schema construction; lazy scans
        performed later (e.g. through a polars `LazyFrame`) are not retried.
        """
        store = NgioStore.ensure(self._group_handler.store)
        path = self._group_handler.group.path
        if store.store_type == "local":
            return self._load_from_local_store(store, path)
        elif store.store_type == "fsspec":
            return self._load_from_fsspec_store(store, path)
        elif store.store_type == "memory":
            return self._load_from_in_memory_store(store, path)
        elif store.store_type == "zip":
            return self._load_from_zip_store(store, path)
        self._raise_store_type_not_supported()

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        dataset = self._load_pyarrow_dataset()
        dataframe = dataset.to_table().to_pandas()
        dataframe = normalize_pandas_df(
            dataframe,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=False,
        )
        return dataframe

    def load(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        return self.load_as_pandas_df()

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame."""
        dataset = self._load_pyarrow_dataset()
        lazy_frame = pl.scan_pyarrow_dataset(dataset)
        if not isinstance(lazy_frame, LazyFrame):
            raise NgioValueError(
                "Table is not a lazy frame. Please report this issue as an ngio bug."
                f" {type(lazy_frame)}"
            )

        lazy_frame = normalize_polars_lf(
            lazy_frame,
            index_key=self.index_key,
            index_type=self.index_type,
        )
        return lazy_frame

    def _write_to_stream(self, stream, table: pa.Table) -> None:
        """Write the table to a stream."""
        if self.table_format == "parquet":
            pa_parquet.write_table(table, stream)
        elif self.table_format == "csv":
            pa_csv.write_csv(table, stream)
        else:
            raise NgioValueError(
                f"Unsupported table format: {self.table_format}. "
                "Supported formats are 'parquet' and 'csv'."
            )

    def _write_to_local_store(
        self, store: NgioStore, path: str, table: pa.Table
    ) -> None:
        """Write the table to a directory store."""
        table_path = f"{store.full_url()}/{path}/{self.table_name}"
        self._write_to_stream(table_path, table)

    def _write_to_fsspec_store(
        self, store: NgioStore, path: str, table: pa.Table
    ) -> None:
        """Write the table to an FS store."""
        fs, table_path = store.sync_fs_and_path(f"{path}/{self.table_name}")
        fs = pa_fs.PyFileSystem(pa_fs.FSSpecHandler(fs))
        with fs.open_output_stream(table_path) as out_stream:
            self._write_to_stream(out_stream, table)

    def _write_to_in_memory_store(
        self, store: NgioStore, path: str, table: pa.Table
    ) -> None:
        """Write the table to an in-memory store."""
        table_path = f"{path}/{self.table_name}"
        store.memory_dict[table_path] = table

    def _write_to_zip_store(self, store: NgioStore, path: str, table: pa.Table) -> None:
        """Write the table to a zip store."""
        raise NotImplementedError("Writing to zip store is not implemented yet.")

    @retry_io
    def _write_pyarrow_dataset(self, dataset: pa.Table) -> None:
        """Write the table from a pyarrow Dataset.

        This IO bypasses the zarr store, so the `io_retry` policy is
        applied here.
        """
        store = NgioStore.ensure(self._group_handler.store)
        path = self._group_handler.group.path
        if store.store_type == "local":
            return self._write_to_local_store(store=store, path=path, table=dataset)
        elif store.store_type == "fsspec":
            return self._write_to_fsspec_store(store=store, path=path, table=dataset)
        elif store.store_type == "memory":
            return self._write_to_in_memory_store(store=store, path=path, table=dataset)
        elif store.store_type == "zip":
            return self._write_to_zip_store(store=store, path=path, table=dataset)
        self._raise_store_type_not_supported()

    def write_from_pandas(self, table: DataFrame) -> None:
        """Write the table from a pandas DataFrame."""
        table = normalize_pandas_df(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=True,
        )
        table = pa.Table.from_pandas(table, preserve_index=False)
        self._write_pyarrow_dataset(table)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Write the table from a polars DataFrame or LazyFrame."""
        table = normalize_polars_lf(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )

        if isinstance(table, LazyFrame):
            table = table.collect()
        table = table.to_arrow()
        self._write_pyarrow_dataset(table)
