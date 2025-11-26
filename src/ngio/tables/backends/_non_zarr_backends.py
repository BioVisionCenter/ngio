import io
from collections.abc import Callable
from typing import Any

from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import FsspecStore, LocalStore, MemoryStore, ZipStore

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._utils import normalize_pandas_df, normalize_polars_lf
from ngio.utils import NgioFileNotFoundError, NgioValueError


class NonZarrBaseBackend(AbstractTableBackend):
    """A class to load and write small tables in CSV format."""

    def __init__(
        self,
        df_reader: Callable[[Any], DataFrame],
        lf_reader: Callable[[Any], LazyFrame],
        df_writer: Callable[[str, DataFrame], None],
        lf_writer: Callable[[str, PolarsDataFrame], None],
        table_name: str,
    ):
        self.df_reader = df_reader
        self.lf_reader = lf_reader
        self.df_writer = df_writer
        self.lf_writer = lf_writer
        self.table_name = table_name

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

    def _load_from_local_store(self, reader):
        """Load the table from a directory store."""
        url = self._group_handler.full_url
        assert url is not None
        table_path = f"{url}/{self.table_name}"
        dataframe = reader(table_path)
        return dataframe

    def _load_from_fs_store_df(self, reader):
        """Load the table from an FS store."""
        path = self._group_handler.group.path
        table_path = f"{path}/{self.table_name}"
        bytes_table = self._group_handler.store.get(table_path)
        if bytes_table is None:
            raise NgioFileNotFoundError(f"No table found at {table_path}. ")
        dataframe = reader(io.BytesIO(bytes_table))
        return dataframe

    def _load_from_fs_store_lf(self, reader):
        """Load the table from an FS store."""
        full_url = self._group_handler.full_url
        parquet_path = f"{full_url}/{self.table_name}"
        store_fs = self._group_handler.store.fs  # type: ignore (in this context, store_fs is a fs.FSStore)
        with store_fs.open(parquet_path, "rb") as f:
            dataframe = reader(f)
        return dataframe

    def _load_from_in_memory_store(self, reader):
        """Load the table from an in-memory store."""
        raise NotImplementedError("In-memory store loading is not implemented yet.")

    def _load_from_zip_store(self, reader):
        """Load the table from a zip store."""
        raise NotImplementedError("Zip store loading is not implemented yet.")

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        store = self._group_handler.store
        if isinstance(store, LocalStore):
            dataframe = self._load_from_local_store(reader=self.df_reader)
        elif isinstance(store, FsspecStore):
            dataframe = self._load_from_fs_store_df(reader=self.df_reader)
        elif isinstance(store, MemoryStore):
            dataframe = self._load_from_in_memory_store(reader=self.df_reader)
        elif isinstance(store, ZipStore):
            dataframe = self._load_from_zip_store(reader=self.df_reader)
        else:
            self._raise_store_type_not_supported()

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
        store = self._group_handler.store
        if isinstance(store, LocalStore):
            lazy_frame = self._load_from_local_store(reader=self.lf_reader)
        elif isinstance(store, FsspecStore):
            lazy_frame = self._load_from_fs_store_lf(reader=self.lf_reader)
        elif isinstance(store, MemoryStore):
            lazy_frame = self._load_from_in_memory_store(reader=self.lf_reader)
        elif isinstance(store, ZipStore):
            lazy_frame = self._load_from_zip_store(reader=self.lf_reader)
        else:
            self._raise_store_type_not_supported()

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

    def _write_to_local_store(self, writer, table):
        """Write the table to a directory store."""
        url = self._group_handler.full_url
        assert url is not None
        table_path = f"{url}/{self.table_name}"
        writer(table_path, table)

    def _write_to_fs_store(self, writer, table):
        """Write the table to an FS store."""
        raise NotImplementedError("Writing to FS store is not implemented yet.")

    def _write_to_in_memory_store(self, writer, table):
        """Write the table to an in-memory store."""
        raise NotImplementedError("Writing to in-memory store is not implemented yet.")

    def _write_to_zip_store(self, writer, table):
        """Write the table to a zip store."""
        raise NotImplementedError("Writing to zip store is not implemented yet.")

    def write_from_pandas(self, table: DataFrame) -> None:
        """Write the table from a pandas DataFrame."""
        table = normalize_pandas_df(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=True,
        )
        if isinstance(self._group_handler.store, LocalStore):
            self._write_to_local_store(writer=self.df_writer, table=table)
        elif isinstance(self._group_handler.store, FsspecStore):
            self._write_to_fs_store(writer=self.df_writer, table=table)
        elif isinstance(self._group_handler.store, MemoryStore):
            self._write_to_in_memory_store(writer=self.df_writer, table=table)
        elif isinstance(self._group_handler.store, ZipStore):
            self._write_to_zip_store(writer=self.df_writer, table=table)
        else:
            self._raise_store_type_not_supported()

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Write the table from a polars DataFrame or LazyFrame."""
        table = normalize_polars_lf(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )

        if isinstance(table, LazyFrame):
            table = table.collect()

        if isinstance(self._group_handler.store, LocalStore):
            self._write_to_local_store(writer=self.lf_writer, table=table)
        elif isinstance(self._group_handler.store, FsspecStore):
            self._write_to_fs_store(writer=self.lf_writer, table=table)
        elif isinstance(self._group_handler.store, MemoryStore):
            self._write_to_in_memory_store(writer=self.lf_writer, table=table)
        elif isinstance(self._group_handler.store, ZipStore):
            self._write_to_zip_store(writer=self.lf_writer, table=table)
        else:
            self._raise_store_type_not_supported()
