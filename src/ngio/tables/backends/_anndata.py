import zarr
from anndata import AnnData
from anndata._settings import settings
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._anndata_utils import (
    custom_anndata_read_zarr,
)
from ngio.tables.backends._utils import (
    convert_pandas_to_anndata,
    convert_polars_to_anndata,
    normalize_anndata,
)
from ngio.utils import NgioValueError, copy_group


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
        settings.zarr_write_format = self._group_handler.zarr_format
        anndata = custom_anndata_read_zarr(self._group_handler._group)
        anndata = normalize_anndata(anndata, index_key=self.index_key)
        return anndata

    def load(self) -> AnnData:
        """Load the table as an AnnData object."""
        return self.load_as_anndata()

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        # Make sure to use the correct zarr format
        settings.zarr_write_format = self._group_handler.zarr_format

        if isinstance(self._group_handler.store, LocalStore):
            store = self._group_handler.full_url
            assert store is not None
            table.write_zarr(store)
        elif isinstance(self._group_handler.store, FsspecStore):
            full_url = self._group_handler.full_url
            assert full_url is not None
            # Remap to fsspec store to the new full URL
            fs = self._group_handler.store.fs
            store = fs.get_mapper(full_url)
            table.write_zarr(store)
        elif isinstance(self._group_handler.store, MemoryStore):
            store = MemoryStore()
            table.write_zarr(store)
            anndata_group = zarr.open_group(store, mode="r")
            copy_group(
                anndata_group,
                self._group_handler._group,
            )
        else:
            raise NgioValueError(
                f"Ngio does not support writing an AnnData table to a "
                f"store of type {type(self._group_handler.store)}. "
                "Please make sure to use a compatible "
                "store like a LocalStore, or FsspecStore."
            )

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
