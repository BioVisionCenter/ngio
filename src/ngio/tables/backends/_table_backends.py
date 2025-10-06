"""Protocol for table backends handlers."""

from typing import Literal, Protocol

from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._anndata import AnnDataBackend, AnnDataBackendV1
from ngio.tables.backends._csv import CsvTableBackend
from ngio.tables.backends._json import JsonTableBackend
from ngio.tables.backends._parquet import ParquetTableBackend
from ngio.tables.backends._utils import TabularData
from ngio.utils import NgioValueError, ZarrGroupHandler


class TableBackendProtocol(Protocol):
    def set_group_handler(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> None:
        """Attach a group handler to the backend.

        Index keys and index types are used to ensure that the
        serialization and deserialization of the table
        is consistent across different backends.

        Making sure that this is consistent is
        a duty of the backend implementations.
        """
        ...

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend.

        As a convention we set name as:
            {backend_name}_v{version}

        Where the version is a integer.
        """
        ...

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Return the group handler."""
        ...

    @staticmethod
    def implements_anndata() -> bool:
        """Check if the backend implements the anndata protocol.

        If this is True, the backend should implement the
        `write_from_anndata` method.

        AnnData objects are more complex than DataFrames,
        so if this is true the backend should implement the
        full serialization of the AnnData object.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        ...

    @staticmethod
    def implements_pandas() -> bool:
        """Check if the backend implements the pandas protocol.

        If this is True, the backend should implement the
        `write_from_dataframe` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        ...

    @staticmethod
    def implements_polars() -> bool:
        """Check if the backend implements the polars protocol.

        If this is True, the backend should implement the
        `write_from_polars` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        ...

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        ...

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame."""
        ...

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame."""
        ...

    def load(self) -> TabularData:
        """The default load method.

        This method will be default way to load the table
        from the backend. This method should wrap one of the
        `load_as_anndata`, `load_as_dataframe` or `load_as_polars`
        methods depending on the backend implementation.
        """
        ...

    def write_from_pandas(self, table: DataFrame) -> None:
        """Serialize the table from a pandas DataFrame."""
        ...

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        ...

    def write_from_polars(self, table: LazyFrame | PolarsDataFrame) -> None:
        """Serialize the table from a polars DataFrame or LazyFrame."""
        ...

    def write(
        self,
        table_data: DataFrame | AnnData | PolarsDataFrame | LazyFrame,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """This is a generic write method.

        Will call the appropriate write method
        depending on the type of the table.

        Moreover it will also write the metadata
        if provided, and the backend methadata

        the backend should write in the zarr group attributes
            - backend: the backend name (self.backend_name())
            - index_key: the index key
            - index_type: the index type

        """


class ImplementedTableBackends:
    """A class to manage the available table backends."""

    _instance = None
    _implemented_backends: dict[str, type[TableBackendProtocol]]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_backends = {}
        return cls._instance

    @property
    def available_backends(self) -> list[str]:
        """Return the available table backends."""
        return list(self._implemented_backends.keys())

    def get_backend(
        self,
        *,
        group_handler: ZarrGroupHandler,
        backend_name: str = "anndata",
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> TableBackendProtocol:
        """Try to get a handler for the given store based on the metadata version."""
        if backend_name not in self._implemented_backends:
            raise NgioValueError(f"Table backend {backend_name} not implemented.")
        backend = self._implemented_backends[backend_name]()
        backend.set_group_handler(
            group_handler=group_handler, index_key=index_key, index_type=index_type
        )
        return backend

    def _add_backend(
        self,
        table_backend: type[TableBackendProtocol],
        name: str,
        overwrite: bool = False,
    ) -> None:
        """Register a new handler."""
        if name in self._implemented_backends and not overwrite:
            raise NgioValueError(
                f"Table backend {name} already implemented. "
                "Use the `overwrite=True` parameter to overwrite it."
            )
        self._implemented_backends[name] = table_backend

    def add_backend(
        self,
        table_backend: type[TableBackendProtocol],
        overwrite: bool = False,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a new handler."""
        self._add_backend(
            table_backend=table_backend,
            name=table_backend.backend_name(),
            overwrite=overwrite,
        )
        if aliases is not None:
            for alias in aliases:
                self._add_backend(
                    table_backend=table_backend, name=alias, overwrite=overwrite
                )


ImplementedTableBackends().add_backend(AnnDataBackend)
ImplementedTableBackends().add_backend(AnnDataBackendV1)
ImplementedTableBackends().add_backend(
    JsonTableBackend, aliases=["experimental_json_v1"]
)
ImplementedTableBackends().add_backend(CsvTableBackend, aliases=["experimental_csv_v1"])
ImplementedTableBackends().add_backend(
    ParquetTableBackend, aliases=["experimental_parquet_v1"]
)

TableBackend = Literal["anndata", "json", "csv", "parquet"] | str | TableBackendProtocol
DefaultTableBackend = "anndata_v1"
