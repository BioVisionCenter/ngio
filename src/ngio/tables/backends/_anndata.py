from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._anndata_utils import (
    custom_anndata_read_zarr,
)
from ngio.tables.backends._utils import (
    convert_pandas_to_anndata,
    convert_polars_to_anndata,
    normalize_anndata,
)
from ngio.utils import NgioValueError


class AnnDataBackend(AbstractTableBackend):
    """A class to load and write tables from/to an AnnData object."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "anndata"

    @staticmethod
    def implements_anndata() -> bool:
        """Check if the backend implements the anndata protocol."""
        return True

    @staticmethod
    def implements_pandas() -> bool:
        """Whether the handler implements the dataframe protocol."""
        return True

    @staticmethod
    def implements_polars() -> bool:
        """Whether the handler implements the polars protocol."""
        return True

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        anndata = custom_anndata_read_zarr(self._group_handler._group)
        anndata = normalize_anndata(anndata, index_key=self.index_key)
        return anndata

    def load(self) -> AnnData:
        """Load the table as an AnnData object."""
        return self.load_as_anndata()

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        full_url = self._group_handler.full_url
        if full_url is None:
            raise NgioValueError(
                f"Ngio does not support writing file from a "
                f"store of type {type(self._group_handler)}. "
                "Please make sure to use a compatible "
                "store like a zarr.DirectoryStore."
            )
        table.write_zarr(full_url)  # type: ignore (AnnData writer requires a str path)

    def write_from_pandas(self, table: DataFrame) -> None:
        """Serialize the table from a pandas DataFrame."""
        anndata = convert_pandas_to_anndata(
            table,
            index_key=self.index_key,
        )
        self.write_from_anndata(anndata)

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Consolidate the metadata in the store."""
        anndata = convert_polars_to_anndata(
            table,
            index_key=self.index_key,
        )
        self.write_from_anndata(anndata)


class AnnDataBackendV1(AnnDataBackend):
    """A wrapper for the AnnData backend that for backwards compatibility."""

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "anndata_v1"
