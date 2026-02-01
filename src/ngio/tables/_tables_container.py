"""Module for handling the /tables group in an OME-NGFF file."""

from typing import Literal, Protocol, TypeVar

import anndata as ad
import pandas as pd
import polars as pl

from ngio.tables.backends import (
    BackendMeta,
    DefaultTableBackend,
    TableBackend,
    TabularData,
)
from ngio.tables.v1 import (
    ConditionTableV1,
    FeatureTableV1,
    GenericTable,
    MaskingRoiTableV1,
    RoiTableV1,
)
from ngio.tables.v1._roi_table import GenericRoiTableV1
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)

GenericRoiTable = GenericRoiTableV1
RoiTable = RoiTableV1
MaskingRoiTable = MaskingRoiTableV1
FeatureTable = FeatureTableV1
ConditionTable = ConditionTableV1


class Table(Protocol):
    """Placeholder class for a table."""

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        ...

    @staticmethod
    def version() -> str:
        """Return the version of the table."""
        ...

    @property
    def backend_name(self) -> str | None:
        """The name of the backend."""
        ...

    @property
    def meta(self) -> BackendMeta:
        """Return the metadata for the table."""
        ...

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the table as a DataFrame."""
        ...

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Return the table as a LazyFrame."""
        ...

    @property
    def anndata(self) -> ad.AnnData:
        """Return the table as an AnnData object."""
        ...

    def set_table_data(
        self,
        table_data: TabularData | None = None,
        refresh: bool = False,
    ) -> None:
        """Make sure that the table data is set (exist in memory).

        If an object is passed, it will be used as the table.
        If None is passed, the table will be loaded from the backend.

        If refresh is True, the table will be reloaded from the backend.
            If table is not None, this will be ignored.
        """
        ...

    def set_backend(
        self,
        handler: ZarrGroupHandler | None = None,
        backend: TableBackend = DefaultTableBackend,
    ) -> None:
        """Set the backend store and path for the table.

        Either a handler or a backend must be provided.

        If the handler in none it will be inferred from the backend.
        If the backend is none, it will be inferred from the group attrs
        """
        ...

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
    ) -> "Table":
        """Create a new table from a Zarr group handler."""
        ...

    @classmethod
    def from_table_data(cls, table_data: TabularData, meta: BackendMeta) -> "Table":
        """Create a new table from a DataFrame."""
        ...

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        ...

    def consolidate(self) -> None:
        """Consolidate the table on disk."""
        ...


TypedTable = Literal[
    "generic_table",
    "roi_table",
    "masking_roi_table",
    "feature_table",
    "condition_table",
]

TypedRoiTable = Literal[
    "roi_table",
    "masking_roi_table",
]

TableType = TypeVar("TableType", bound=Table)


class TableMeta(BackendMeta):
    """Base class for table metadata."""

    table_version: str = "1"
    type: str = "generic_table"

    def unique_name(self) -> str:
        """Return the unique name for the table."""
        return f"{self.type}_v{self.table_version}"


def _get_meta(handler: ZarrGroupHandler) -> TableMeta:
    """Get the metadata from the handler."""
    attrs = handler.load_attrs()
    meta = TableMeta(**attrs)
    return meta


class ImplementedTables:
    """A singleton class to manage the available table handler plugins."""

    _instance = None
    _implemented_tables: dict[str, type[Table]]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._implemented_tables = {}
        return cls._instance

    def available_implementations(self) -> list[str]:
        """Get the available table handler versions."""
        return list(self._implemented_tables.keys())

    def get_table(
        self,
        meta: TableMeta,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
        strict: bool = True,
    ) -> Table:
        """Try to get a handler for the given store based on the metadata version."""
        if strict:
            default = None
        else:
            default = GenericTable

        table_cls = self._implemented_tables.get(meta.unique_name(), default)
        if table_cls is None:
            raise NgioValueError(
                f"Table handler for {meta.unique_name()} not implemented."
            )
        table = table_cls.from_handler(handler=handler, backend=backend)
        return table

    def _add_implementation(
        self, handler: type[Table], name: str, overwrite: bool = False
    ) -> None:
        """Register a new table handler."""
        if name in self._implemented_tables and not overwrite:
            raise NgioValueError(
                f"Table handler for {name} already implemented. "
                "Use overwrite=True to replace it."
            )
        self._implemented_tables[name] = handler

    def add_implementation(
        self,
        handler: type[Table],
        overwrite: bool = False,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a new table handler."""
        meta = TableMeta(
            type=handler.table_type(),
            table_version=handler.version(),
        )

        self._add_implementation(handler, meta.unique_name(), overwrite)

        if aliases is not None:
            for alias in aliases:
                self._add_implementation(handler, alias, overwrite)


class TablesContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler

        # Validate the group
        # Either contains a tables attribute or is empty
        attrs = self._group_handler.load_attrs()
        if len(attrs) == 0:
            # It's an empty group
            pass
        elif "tables" in attrs and isinstance(attrs["tables"], list):
            # It's a valid group
            pass
        else:
            raise NgioValidationError(
                f"Invalid /tables group. "
                f"Expected a single tables attribute with a list of table names. "
                f"Found: {attrs}"
            )

    def _get_tables_list(self) -> list[str]:
        """Create the /tables group if it doesn't exist."""
        attrs = self._group_handler.load_attrs()
        return attrs.get("tables", [])

    def _get_table_group_handler(self, name: str) -> ZarrGroupHandler:
        """Get the group handler for a table."""
        handler = self._group_handler.get_handler(path=name)
        return handler

    def list(self, filter_types: TypedTable | str | None = None) -> list[str]:
        """List all labels in the group."""
        tables = self._get_tables_list()
        if filter_types is None:
            return tables

        filtered_tables = []
        for table_name in tables:
            tb_handler = self._get_table_group_handler(table_name)
            table_type = _get_meta(tb_handler).type
            if table_type == filter_types:
                filtered_tables.append(table_name)
        return filtered_tables

    def get(
        self,
        name: str,
        backend: TableBackend | None = None,
        strict: bool = True,
    ) -> Table:
        """Get a label from the group."""
        if name not in self.list():
            raise NgioValueError(f"Table '{name}' not found in the group.")

        table_handler = self._get_table_group_handler(name)

        meta = _get_meta(table_handler)
        return ImplementedTables().get_table(
            meta=meta,
            handler=table_handler,
            backend=backend,
            strict=strict,
        )

    def get_as(
        self,
        name: str,
        table_cls: type[TableType],
        backend: TableBackend | None = None,
    ) -> TableType:
        """Get a table from the group as a specific type."""
        if name not in self.list():
            raise NgioValueError(f"Table '{name}' not found in the group.")

        table_handler = self._get_table_group_handler(name)
        return table_cls.from_handler(
            handler=table_handler,
            backend=backend,
        )  # type: ignore[return-value]

    def delete(self, name: str, missing_ok: bool = False) -> None:
        """Delete a table from the group.

        Args:
            name (str): The name of the table to delete.
            missing_ok (bool): If True, do not raise an error if
                the table does not exist.
        """
        existing_tables = self._get_tables_list()
        if name not in existing_tables:
            if missing_ok:
                return
            raise NgioValueError(
                f"Table '{name}' not found in the Tables group. "
                f"Available tables: {existing_tables}"
            )

        self._group_handler.delete_group(name)
        existing_tables.remove(name)
        self._group_handler.write_attrs({"tables": existing_tables})

    def add(
        self,
        name: str,
        table: Table,
        backend: TableBackend = DefaultTableBackend,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the group."""
        existing_tables = self._get_tables_list()
        if name in existing_tables and not overwrite:
            raise NgioValueError(
                f"Table '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        table_handler = self._group_handler.get_handler(path=name, overwrite=overwrite)

        if backend is None:
            backend = table.backend_name

        table.set_table_data()
        table.set_backend(
            handler=table_handler,
            backend=backend,
        )
        table.consolidate()
        if name not in existing_tables:
            existing_tables.append(name)
            self._group_handler.write_attrs({"tables": existing_tables})


ImplementedTables().add_implementation(RoiTableV1)
ImplementedTables().add_implementation(MaskingRoiTableV1)
ImplementedTables().add_implementation(FeatureTableV1)
ImplementedTables().add_implementation(ConditionTableV1)

###################################################################################
#
# Utility functions to open and write tables
#
###################################################################################


def open_tables_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> TablesContainer:
    """Open a table handler from a Zarr store."""
    handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    return TablesContainer(handler)


def open_table(
    store: StoreOrGroup,
    backend: TableBackend | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Table:
    """Open a table from a Zarr store."""
    handler = ZarrGroupHandler(
        store=store,
        cache=cache,
        mode=mode,
    )
    meta = _get_meta(handler)
    return ImplementedTables().get_table(
        meta=meta, handler=handler, backend=backend, strict=False
    )


def open_table_as(
    store: StoreOrGroup,
    table_cls: type[TableType],
    backend: TableBackend | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> TableType:
    """Open a table from a Zarr store as a specific type."""
    handler = ZarrGroupHandler(
        store=store,
        cache=cache,
        mode=mode,
    )
    return table_cls.from_handler(
        handler=handler,
        backend=backend,
    )  # type: ignore[return-value]


def write_table(
    store: StoreOrGroup,
    table: Table,
    backend: TableBackend = DefaultTableBackend,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
) -> None:
    """Write a table to a Zarr store.

    A table will be created at the given store location.

    Args:
        store (StoreOrGroup): The Zarr store or group to write the table to.
        table (Table): The table to write.
        backend (TableBackend): The backend to use for writing the table.
        cache (bool): Whether to use caching for the Zarr group handler.
        mode (AccessModeLiteral): The access mode to use for the Zarr group handler.

    """
    handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    table.set_backend(
        handler=handler,
        backend=backend,
    )
    table.consolidate()
