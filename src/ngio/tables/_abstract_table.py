"""Implementation of a generic table class."""

import builtins
from abc import ABC, abstractmethod
from typing import Literal, Self

import pandas as pd
import polars as pl
from anndata import AnnData

from ngio.tables.backends import (
    BackendMeta,
    ImplementedTableBackends,
    TableBackend,
    TableBackendProtocol,
    TabularData,
    convert_to_anndata,
    convert_to_pandas,
    convert_to_polars,
    normalize_table,
)
from ngio.utils import NgioValueError, ZarrGroupHandler


class AbstractBaseTable(ABC):
    """Abstract base class for a table.

    This is used to define common methods and properties
    for all tables.

    This class is not meant to be used directly.
    """

    def __init__(
        self,
        table_data: TabularData | None = None,
        *,
        meta: BackendMeta | None = None,
    ) -> None:
        """Initialize the table."""
        if meta is None:
            meta = BackendMeta()

        self._meta = meta
        if table_data is not None:
            table_data = normalize_table(
                table_data,
                index_key=meta.index_key,
                index_type=meta.index_type,
            )
        self._table_data = table_data
        self._table_backend = None

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        return f"{self.__class__.__name__}"

    @staticmethod
    @abstractmethod
    def table_type() -> str:
        """Return the type of the table."""
        ...

    @staticmethod
    @abstractmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        ...

    @property
    def backend_name(self) -> str:
        """Return the name of the backend.

        If no backend is attached yet, the backend name stored in the
        table metadata is returned.
        """
        if self._table_backend is None:
            return self._meta.backend
        return self._table_backend.backend_name()

    @property
    def meta(self) -> BackendMeta:
        """Return the metadata of the table."""
        return self._meta

    @property
    def index_key(self) -> str | None:
        """Get the index key."""
        return self._meta.index_key

    @property
    def index_type(self) -> Literal["int", "str"] | None:
        """Get the index type."""
        return self._meta.index_type

    def load_as_anndata(self) -> AnnData:
        """Load the table as an AnnData object."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_anndata()

    def load_as_pandas_df(self) -> pd.DataFrame:
        """Load the table as a pandas DataFrame."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_pandas_df()

    def load_as_polars_lf(self) -> pl.LazyFrame:
        """Load the table as a polars LazyFrame."""
        if self._table_backend is None:
            raise NgioValueError("No backend set for the table.")
        return self._table_backend.load_as_polars_lf()

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        if self._table_data is not None:
            return self._table_data

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        self._table_data = self._table_backend.load()
        return self._table_data

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        return convert_to_pandas(
            self.table_data, index_key=self.index_key, index_type=self.index_type
        )

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Return the table as a LazyFrame."""
        return convert_to_polars(
            self.table_data, index_key=self.index_key, index_type=self.index_type
        )

    @property
    def anndata(self) -> AnnData:
        """Return the table as an AnnData object."""
        return convert_to_anndata(self.table_data, index_key=self.index_key)

    @staticmethod
    def _load_backend(
        meta: BackendMeta,
        handler: ZarrGroupHandler,
        backend: TableBackend | None,
    ) -> TableBackendProtocol:
        """Create a new ROI table from a Zarr group handler."""
        if backend is None:
            backend = meta.backend
        if isinstance(backend, str):
            return ImplementedTableBackends().get_backend(
                backend_name=backend,
                group_handler=handler,
                index_key=meta.index_key,
                index_type=meta.index_type,
            )
        backend.set_group_handler(
            group_handler=handler,
            index_key=meta.index_key,
            index_type=meta.index_type,
        )
        return backend

    def set_table_data(
        self,
        table_data: TabularData | None = None,
        refresh: bool = False,
    ) -> None:
        """Set the table.

        If an object is passed, it will be used as the table.
        If None is passed, the table will be loaded from the backend.

        If refresh is True, the table will be reloaded from the backend.
            If table is not None, this will be ignored.
        """
        if table_data is not None:
            if not isinstance(table_data, TabularData):
                raise NgioValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            self._table_data = normalize_table(
                table_data,
                index_key=self.index_key,
                index_type=self.index_type,
            )
            return None

        if self._table_data is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )
        self._table_data = self._table_backend.load()

    def set_backend(
        self,
        handler: ZarrGroupHandler | None = None,
        backend: TableBackend | None = None,
    ) -> None:
        """Set the backend of the table.

        If `backend` is `None`, the backend stored in the table metadata
        is used.

        If no handler is provided and the table is not yet attached to a
        Zarr group, a string `backend` is only recorded as the table's
        preferred backend; the backend is instantiated when the table is
        written (e.g. by `add_table`).
        """
        if handler is None:
            if self._table_backend is None:
                if backend is None:
                    return None
                if isinstance(backend, str):
                    backends = ImplementedTableBackends()
                    self._meta.backend = backends.normalize_backend_name(backend)
                    return None
                raise NgioValueError(
                    "A ZarrGroupHandler must be provided to attach a "
                    "backend instance to the table."
                )
            handler = self._table_backend.group_handler

        meta = self._meta
        _backend = self._load_backend(
            meta=meta,
            handler=handler,
            backend=backend,
        )
        self._table_backend = _backend
        self._meta.backend = _backend.backend_name()

    @classmethod
    def _from_handler(
        cls,
        handler: ZarrGroupHandler,
        meta_model: builtins.type[BackendMeta],
        backend: TableBackend | None = None,
        attrs: dict | None = None,
    ) -> Self:
        """Create a new ROI table from a Zarr group handler.

        `attrs` lets a caller that has already read the group hand them over
        rather than making this read the same document a second time — which
        `TablesContainer.get` did, once to learn the table type and once here
        to build the concrete metadata model.
        """
        if attrs is None:
            attrs = handler.load_attrs()
        meta = meta_model(**attrs)
        table = cls(meta=meta)
        table.set_backend(handler=handler, backend=backend)
        return table

    @classmethod
    @abstractmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
        attrs: dict | None = None,
    ) -> Self:
        """Create a new ROI table from a Zarr group handler."""
        # Not `pass`: `abstractmethod` only guards instantiation, so a call on
        # a subclass that forgot to override this would silently return `None`
        # — `get_as(name, cls)` handed back nothing, with no error anywhere.
        raise NotImplementedError(f"{cls.__name__} does not implement `from_handler`.")

    @classmethod
    def from_table_data(cls, table_data: TabularData, meta: BackendMeta) -> Self:
        """Create a new ROI table from a Zarr group handler."""
        return cls(
            table_data=table_data,
            meta=meta,
        )

    def consolidate(self) -> None:
        """Write the current state of the table to the Zarr file."""
        if self._table_backend is None:
            raise NgioValueError(
                "No backend set for the table. "
                "Please add the table to a OME-Zarr Image before calling consolidate."
            )

        self._table_backend.write(
            self.table_data,
            metadata=self._meta.model_dump(exclude_none=True),
        )
