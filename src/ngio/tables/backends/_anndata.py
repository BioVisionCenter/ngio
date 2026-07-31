import zarr
from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from zarr.storage import MemoryStore

from ngio.tables.backends._abstract_backend import AbstractTableBackend
from ngio.tables.backends._anndata_utils import (
    _update_anndata_global_settings,
    custom_anndata_read_zarr,
)
from ngio.tables.backends._utils import (
    convert_pandas_to_anndata,
    convert_polars_to_anndata,
    normalize_anndata,
)
from ngio.utils import NgioStore, NgioValueError, copy_group, retry_io


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
        _update_anndata_global_settings(self._group_handler.zarr_format)
        anndata = custom_anndata_read_zarr(self._group_handler._group)
        anndata = normalize_anndata(anndata, index_key=self.index_key)
        return anndata

    def load(self) -> AnnData:
        """Load the table as an AnnData object."""
        return self.load_as_anndata()

    @retry_io
    def _write_to_local_store(
        self, store: NgioStore, path: str, table: AnnData
    ) -> None:
        """Write the AnnData table to a LocalStore.

        AnnData writes its own zarr hierarchy directly to the path, bypassing
        the (retrying) `NgioStore`, so the `io_retry` policy is applied here.
        """
        url = store.full_url(path)
        if url is None:
            raise NgioValueError(
                f"Cannot resolve a local path for store {store} at {path}."
            )
        # ty resolves AnnData.write_zarr as an unbound function, not a method.
        table.write_zarr(url)  # type: ignore

    @retry_io
    def _write_to_fsspec_store(
        self, store: NgioStore, path: str, table: AnnData
    ) -> None:
        """Write the AnnData table to a FsspecStore.

        AnnData writes through a raw fsspec mapper, bypassing the (retrying)
        `NgioStore`, so the `io_retry` policy is applied here.
        """
        table.write_zarr(store.get_mapper(path))  # type: ignore

    def _write_to_memory_store(self, table: AnnData) -> None:
        """Write the AnnData table to a MemoryStore."""
        scratch_store = MemoryStore()
        table.write_zarr(scratch_store)  # type: ignore
        anndata_group = zarr.open_group(scratch_store, mode="r")
        copy_group(
            anndata_group,
            self._group_handler._group,
            suppress_warnings=True,
        )

    def _cleanup_after_write(self) -> None:
        """Clean up any temporary data after writing."""
        group = self._group_handler._group
        try:
            raw_group = group["raw"]
        except KeyError:
            return
        # Remove "raw" entry (encoding-type "null") that anndata <0.11 can't read
        if dict(raw_group.attrs).get("encoding-type") == "null":
            del group["raw"]

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        # Make sure to use the correct zarr format
        _update_anndata_global_settings(self._group_handler.zarr_format)
        store = NgioStore.ensure(self._group_handler.store)
        path = self._group_handler.group.path
        if store.store_type == "local":
            self._write_to_local_store(
                store,
                path,
                table,
            )
        elif store.store_type == "fsspec":
            self._write_to_fsspec_store(
                store,
                path,
                table,
            )
        elif store.store_type == "memory":
            self._write_to_memory_store(table)
        else:
            raise NgioValueError(
                f"Ngio does not support writing an AnnData table to a "
                f"store of type {type(store)}. "
                "Please make sure to use a compatible "
                "store like a LocalStore, or FsspecStore."
            )
        self._cleanup_after_write()

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
