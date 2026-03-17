import io

import pandas as pd
import polars as pl
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import DirectoryStore, FSStore

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._utils import normalize_pandas_df, normalize_polars_lf
from ngio.utils import NgioFileNotFoundError, NgioValueError


class CsvTableBackend(AbstractTableBackend):
    """A class to load and write small tables in CSV format."""

    csv_name = "table.csv"

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "experimental_csv_v1"

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

    def _load_from_directory_store(self, reader):
        """Load the table from a directory store."""
        url = self._group_handler.full_url
        if url is None:
            raise NgioValueError(
                f"Ngio does not support reading a CSV file from a "
                f"store of type {type(self._group_handler)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore."
            )
        csv_path = f"{url}/{self.csv_name}"
        dataframe = reader(csv_path)
        return dataframe

    def _load_from_fs_store(self, reader):
        """Load the table from an FS store."""
        bytes_table = self._group_handler.store.get(self.csv_name)
        if bytes_table is None:
            raise NgioFileNotFoundError(f"No table found at {self.csv_name}. ")
        dataframe = reader(io.BytesIO(bytes_table))
        return dataframe

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            dataframe = self._load_from_directory_store(reader=pd.read_csv)
        elif isinstance(store, FSStore):
            dataframe = self._load_from_fs_store(reader=pd.read_csv)
        else:
            raise NgioValueError(
                f"Ngio does not support reading a CSV file from a "
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

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame."""
        store = self._group_handler.store
        if isinstance(store, DirectoryStore):
            lazy_frame = self._load_from_directory_store(reader=pl.scan_csv)
        elif isinstance(store, FSStore):
            lazy_frame = self._load_from_fs_store(reader=pl.scan_csv)
        else:
            raise NgioValueError(
                f"Ngio does not support reading a CSV file from a "
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
            raise NgioValueError(
                f"Ngio does not support writing a CSV file to a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore or "
                "zarr.FSStore."
            )
        if full_url is None:
            raise NgioValueError(
                f"Ngio does not support writing a CSV file to a "
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
        csv_path = f"{full_url}/{self.csv_name}"
        table.to_csv(csv_path, index=False)

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
        csv_path = f"{full_url}/{self.csv_name}"
        table.write_csv(csv_path)


class CsvTableBackendFC(CsvTableBackend):
    """Forwards compatible version of the CSV backend."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "csv"
