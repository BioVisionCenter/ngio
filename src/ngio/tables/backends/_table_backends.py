"""Protocol for table backends handlers."""

from typing import Literal, Protocol

from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame

from ngio.tables.backends._anndata_v1 import AnnDataBackend, AnnDataBackendFC
from ngio.tables.backends._csv_v1 import CsvTableBackend, CsvTableBackendFC
from ngio.tables.backends._json_v1 import JsonTableBackend, JsonTableBackendFC
from ngio.utils import NgioValueError, ZarrGroupHandler


class TableBackendProtocol(Protocol):
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ):
        """Backend constructor.

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

    @staticmethod
    def implements_anndata() -> bool:
        """Check if the backend implements the anndata protocol.

        If this is True, the backend should implement the
        `load_as_anndata` and `write_from_anndata` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        ...

    @staticmethod
    def implements_pandas() -> bool:
        """Check if the backend implements the pandas protocol.

        If this is True, the backend should implement the
        `load_as_dataframe` and `write_from_dataframe` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        ...

    @staticmethod
    def implements_polars() -> bool:
        """Check if the backend implements the polars protocol.

        If this is True, the backend should implement the
        `load_as_polars` and `write_from_polars` methods.

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
        table: DataFrame | AnnData | PolarsDataFrame | LazyFrame,
        metadata: dict[str, str] | None = None,
        mode: Literal["pandas", "anndata", "polars"] | None = None,
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
        backend_name: str | None,
        group_handler: ZarrGroupHandler,
        index_key: str | None = None,
        index_type: Literal["int", "str"] | None = None,
    ) -> TableBackendProtocol:
        """Try to get a handler for the given store based on the metadata version."""
        if backend_name is None:
            # Default to anndata since it is currently
            # the only backend in use.
            backend_name = "anndata_v1"

        if backend_name not in self._implemented_backends:
            raise NgioValueError(f"Table backend {backend_name} not implemented.")
        handler = self._implemented_backends[backend_name](
            group_handler=group_handler, index_key=index_key, index_type=index_type
        )
        return handler

    def add_backend(
        self,
        table_beckend: type[TableBackendProtocol],
        overwrite: bool = False,
    ):
        """Register a new handler."""
        backend_name = table_beckend.backend_name()
        if backend_name in self._implemented_backends and not overwrite:
            raise NgioValueError(
                f"Table backend {backend_name} already implemented. "
                "Use the `overwrite=True` parameter to overwrite it."
            )
        self._implemented_backends[backend_name] = table_beckend


ImplementedTableBackends().add_backend(AnnDataBackend)
ImplementedTableBackends().add_backend(JsonTableBackend)
ImplementedTableBackends().add_backend(CsvTableBackend)
ImplementedTableBackends().add_backend(AnnDataBackendFC)
ImplementedTableBackends().add_backend(JsonTableBackendFC)
ImplementedTableBackends().add_backend(CsvTableBackendFC)
