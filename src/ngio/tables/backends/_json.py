import pandas as pd
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._utils import (
    normalize_pandas_df,
    normalize_polars_lf,
)
from ngio.utils import NgioError


class JsonTableBackend(AbstractTableBackend):
    """A class to load and write small tables in the zarr group .attrs (json)."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "json"

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

    def _get_table_group(self):
        """Get the table group, creating it if it doesn't exist."""
        try:
            table_group = self._group_handler.get_group(path="table", create_mode=True)
        except NgioError as e:
            raise NgioError(
                "Could not get or create a 'table' group in the store "
                f"{self._group_handler.store} path "
                f"{self._group_handler.group.path}/table."
            ) from e
        return table_group

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        table_dict = self._get_table_group().attrs.asdict()
        data_frame = pd.DataFrame.from_dict(table_dict)
        data_frame = normalize_pandas_df(
            data_frame,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=False,
        )
        return data_frame

    def load(self) -> DataFrame:
        return self.load_as_pandas_df()

    def _write_from_dict(self, table: dict) -> None:
        """Write the table from a dictionary to the store."""
        table_group = self._get_table_group()
        table_group.attrs.clear()
        table_group.attrs.update(table)

    def write_from_pandas(self, table: DataFrame) -> None:
        """Write the table from a pandas DataFrame."""
        table = normalize_pandas_df(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
            reset_index=True,
        )
        table_dict = table.to_dict(orient="list")
        self._write_from_dict(table=table_dict)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Write the table from a polars DataFrame or LazyFrame."""
        table = normalize_polars_lf(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )
        if isinstance(table, LazyFrame):
            table = table.collect()

        table_dict = table.to_dict(as_series=False)
        self._write_from_dict(table=table_dict)
