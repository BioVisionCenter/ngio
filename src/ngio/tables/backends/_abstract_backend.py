from abc import ABC, abstractmethod
from typing import Literal

from anndata import AnnData
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame
from pydantic import BaseModel, ConfigDict

from ngio.tables.backends._utils import (
    TabularData,
    convert_to_anndata,
    convert_to_pandas,
    convert_to_polars,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


class BackendMeta(BaseModel):
    """Metadata for the backend."""

    backend: str = "anndata_v1"
    index_key: str | None = None
    index_type: Literal["int", "str"] | None = None

    model_config = ConfigDict(extra="allow")


class AbstractTableBackend(ABC):
    """Abstract class for table backends."""

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
        self._group_handler = group_handler
        self._index_key = index_key
        self._index_type = index_type

    @staticmethod
    @abstractmethod
    def backend_name() -> str:
        """Return the name of the backend.

        As a convention we set name as:
            {backend_name}_v{version}

        Where the version is a integer.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_pandas() -> bool:
        """Check if the backend implements the pandas protocol.

        If this is True, the backend should implement the
        `write_from_dataframe` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def implements_polars() -> bool:
        """Check if the backend implements the polars protocol.

        If this is True, the backend should implement the
        `write_from_polars` methods.

        If this is False, these methods should raise a
        `NotImplementedError`.
        """
        raise NotImplementedError

    @property
    def group_handler(self) -> ZarrGroupHandler:
        """Get the group handler."""
        return self._group_handler

    @property
    def index_key(self) -> str | None:
        """Get the index key."""
        return self._index_key

    @property
    def index_type(self) -> Literal["int", "str"] | None:
        """Get the index type."""
        if self._index_type is None:
            return None

        if self._index_type not in ["int", "str"]:
            raise NgioValueError(
                f"Invalid index type {self._index_type}. Must be 'int' or 'str'."
            )
        return self._index_type  # type: ignore[return-value]

    @abstractmethod
    def load(self) -> TabularData:
        """Load the table from the store.

        This is a generic load method.
        Based on the explicit mode or the type of the table,
        it will call the appropriate load method.
        """
        ...

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object.

        Since the AnnData object is more complex than a DataFrame,
        selecting columns is not implemented, because it is not
        straightforward to do so for an arbitrary AnnData object.
        """
        table = self.load()
        return convert_to_anndata(
            table,
            index_key=self.index_key,
        )

    def load_as_pandas_df(self) -> DataFrame:
        """Load the table as a pandas DataFrame.

        If columns are provided, the table should be filtered
        """
        table = self.load()
        return convert_to_pandas(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )

    def load_as_polars_lf(self) -> LazyFrame:
        """Load the table as a polars LazyFrame.

        If columns are provided, the table should be filtered
        """
        table = self.load()
        return convert_to_polars(
            table,
            index_key=self.index_key,
            index_type=self.index_type,
        )

    def write_from_pandas(self, table: DataFrame) -> None:
        """Serialize the table from a pandas DataFrame."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of DataFrame objects."
        )

    def write_from_anndata(self, table: AnnData) -> None:
        """Serialize the table from an AnnData object."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of AnnData objects."
        )

    def write_from_polars(self, table: PolarsDataFrame | LazyFrame) -> None:
        """Serialize the table from a polars DataFrame or LazyFrame."""
        raise NotImplementedError(
            f"Backend {self.backend_name()} does not support "
            "serialization of Polars objects."
        )

    def write_metadata(self, metadata: dict | None = None) -> None:
        """Write the metadata to the store."""
        if metadata is None:
            metadata = {}

        backend_metadata = BackendMeta(
            backend=self.backend_name(),
            index_key=self.index_key,
            index_type=self.index_type,
        ).model_dump(exclude_none=True)
        metadata.update(backend_metadata)
        self._group_handler.write_attrs(metadata)

    def write(
        self,
        table_data: TabularData,
        metadata: dict | None = None,
    ) -> None:
        """Serialize the table to the store, and write the metadata.

        This is a generic write method.
        Based on the explicit mode or the type of the table,
        it will call the appropriate write method.
        """
        if isinstance(table_data, DataFrame):
            self.write_from_pandas(table_data)
        elif isinstance(table_data, AnnData):
            self.write_from_anndata(table_data)
        elif isinstance(table_data, PolarsDataFrame | LazyFrame):
            self.write_from_polars(table_data)
        else:
            raise NgioValueError(
                f"Unsupported table type {type(table_data)}. "
                "Please specify the mode explicitly. "
                "Supported serialization modes are: "
                "'pandas', 'anndata', 'polars'."
            )
        self.write_metadata(metadata)
