import io
from collections.abc import Callable
from typing import Any

from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import DirectoryStore, FSStore

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

    def _load_from_directory_store(self, reader):
        """Load the table from a directory store."""
        url = self._group_handler.full_url
        if url is None:
            ext = self.table_name.split(".")[-1]
            raise NgioValueError(
                f"Ngio does not support reading a {ext} table from a "
                f"store of type {type(self._group_handler)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore."
            )
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

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            dataframe = self._load_from_directory_store(reader=self.df_reader)
        elif isinstance(store, FSStore):
            dataframe = self._load_from_fs_store_df(reader=self.df_reader)
        else:
            ext = self.table_name.split(".")[-1]
            raise NgioValueError(
                f"Ngio does not support reading a {ext} table from a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )

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
        if isinstance(store, DirectoryStore):
            lazy_frame = self._load_from_directory_store(reader=self.lf_reader)
        elif isinstance(store, FSStore):
            lazy_frame = self._load_from_fs_store_lf(reader=self.lf_reader)
        else:
            ext = self.table_name.split(".")[-1]
            raise NgioValueError(
                f"Ngio does not support reading a {ext} from a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )
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

    def _get_store_url(self) -> str:
        """Get the store URL."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            full_url = self._group_handler.full_url
        else:
            ext = self.table_name.split(".")[-1]
            raise NgioValueError(
                f"Ngio does not support writing a {ext} file to a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )
        if full_url is None:
            ext = self.table_name.split(".")[-1]
            raise NgioValueError(
                f"Ngio does not support writing a {ext} file to a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )
        return full_url

    def write_from_pandas(self, table: DataFrame) -> None:
        """Write the table from a pandas DataFrame."""
        table = normalize_pandas_df(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=True,
        )
        full_url = self._get_store_url()
        table_path = f"{full_url}/{self.table_name}"
        self.df_writer(table_path, table)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Write the table from a polars DataFrame or LazyFrame."""
        table = normalize_polars_lf(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )

        if isinstance(table, LazyFrame):
            table = table.collect()

        full_url = self._get_store_url()
        table_path = f"{full_url}/{self.table_name}"
        self.lf_writer(table_path, table)
