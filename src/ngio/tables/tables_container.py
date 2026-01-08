"""Module for handling the /tables group in an OME-NGFF file."""

from typing import Literal, Protocol

from ngio.tables.v1 import FeatureTableV1, MaskingRoiTableV1, RoiTableV1
from ngio.tables.v1._generic_table import GenericTable
from ngio.tables.v1._roi_table import _GenericRoiTableV1
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)

GenericRoiTable = _GenericRoiTableV1
RoiTable = RoiTableV1
MaskingRoiTable = MaskingRoiTableV1
FeatureTable = FeatureTableV1


class Table(Protocol):
    """Placeholder class for a table."""

    @staticmethod
    def type() -> str | None:
        """Return the type of the table."""
        ...

    @staticmethod
    def version() -> str | None:
        """Return the version of the table."""
        ...

    @property
    def backend_name(self) -> str | None:
        """The name of the backend."""
        ...

    @classmethod
    def _from_handler(
        cls, handler: ZarrGroupHandler, backend_name: str | None = None
    ) -> "Table":
        """Create a new table from a Zarr group handler."""
        ...

    def _set_backend(
        self,
        handler: ZarrGroupHandler,
        backend_name: str | None = None,
    ) -> None:
        """Set the backend store and path for the table."""
        ...

    def consolidate(self) -> None:
        """Consolidate the table on disk."""
        ...


TypedTable = Literal[
    "roi_table", "masking_roi_table", "feature_table", "generic_roi_table"
]


def _unique_table_name(type_name, version) -> str:
    """Return the unique name for a table."""
    return f"{type_name}_v{version}"


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
        type: str,
        version: str,
        handler: ZarrGroupHandler,
        backend_name: str | None = None,
        strict: bool = True,
    ) -> Table:
        """Try to get a handler for the given store based on the metadata version."""
        _errors = {}
        for name, table_cls in self._implemented_tables.items():
            if name != _unique_table_name(type, version):
                continue
            try:
                table = table_cls._from_handler(
                    handler=handler, backend_name=backend_name
                )
                return table
            except Exception as e:
                if strict:
                    raise NgioValidationError(
                        f"Could not load table {name} from handler. Error: {e}"
                    ) from e
                else:
                    _errors[name] = e
        # If no table was found, we can try to load the table from a generic table
        try:
            table = GenericTable._from_handler(
                handler=handler, backend_name=backend_name
            )
            return table
        except Exception as e:
            _errors["generic"] = e

        if len(_errors) == 0:
            raise NgioValidationError(
                f"Could not find a table implementation for {type} v{version}. "
                f"Available tables: {self.available_implementations()}"
            )

        raise NgioValidationError(
            f"Could not load table from any known version. Errors: {_errors}"
        )

    def add_implementation(self, handler: type[Table], overwrite: bool = False):
        """Register a new table handler."""
        table_type = handler.type()
        version = handler.version()
        if table_type is None:
            raise NgioValueError("Table handler must have a type.")

        if version is None:
            raise NgioValueError("Table handler must have a version.")

        table_unique_name = _unique_table_name(table_type, version)
        if table_unique_name in self._implemented_tables and not overwrite:
            raise NgioValueError(
                f"Table handler for {table_unique_name} already exists. "
                "Use overwrite=True to replace it."
            )
        self._implemented_tables[table_unique_name] = handler


def _get_table_type(handler: ZarrGroupHandler) -> str:
    """Get the type of the table from the handler."""
    attrs = handler.load_attrs()
    return attrs.get("type", "None")


def _get_table_version(handler: ZarrGroupHandler) -> str:
    """Get the version of the table from the handler."""
    attrs = handler.load_attrs()
    return attrs.get("fractal_table_version", "1")


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
        handler = self._group_handler.derive_handler(path=name)
        return handler

    def list_roi_tables(self) -> list[str]:
        """List all ROI tables in the group."""
        _tables = []
        for _type in ["roi_table", "masking_roi_table"]:
            _tables.extend(self.list(_type))
        return _tables

    def list(self, filter_types: str | None = None) -> list[str]:
        """List all labels in the group."""
        tables = self._get_tables_list()
        if filter_types is None:
            return tables

        filtered_tables = []
        for table_name in tables:
            tb_handler = self._get_table_group_handler(table_name)
            table_type = _get_table_type(tb_handler)
            if table_type == filter_types:
                filtered_tables.append(table_name)
        return filtered_tables

    def get(
        self, name: str, backend_name: str | None = None, strict: bool = True
    ) -> Table:
        """Get a label from the group."""
        if name not in self.list():
            raise KeyError(f"Table '{name}' not found in the group.")

        table_handler = self._get_table_group_handler(name)
        table_type = _get_table_type(table_handler)
        table_version = _get_table_version(table_handler)
        return ImplementedTables().get_table(
            type=table_type,
            version=table_version,
            handler=table_handler,
            backend_name=backend_name,
            strict=strict,
        )

    def add(
        self,
        name: str,
        table: Table,
        backend: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the group."""
        existing_tables = self._get_tables_list()
        if name in existing_tables and not overwrite:
            raise NgioValueError(
                f"Table '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        table_handler = self._group_handler.derive_handler(
            path=name, overwrite=overwrite
        )

        if backend is None:
            backend = table.backend_name

        table._set_backend(
            handler=table_handler,
            backend_name=backend,
        )
        table.consolidate()
        if name not in existing_tables:
            existing_tables.append(name)
            self._group_handler.write_attrs({"tables": existing_tables})


ImplementedTables().add_implementation(RoiTableV1)
ImplementedTables().add_implementation(MaskingRoiTableV1)
ImplementedTables().add_implementation(FeatureTableV1)

###################################################################################
#
# Utility functions to open and write tables
#
###################################################################################


def open_tables_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
    parallel_safe: bool = False,
) -> TablesContainer:
    """Open a table handler from a Zarr store."""
    handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return TablesContainer(handler)


def open_table(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
    parallel_safe: bool = False,
) -> Table:
    """Open a table from a Zarr store."""
    handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return ImplementedTables().get_table(
        _get_table_type(handler), _get_table_version(handler), handler
    )


def write_table(
    store: StoreOrGroup,
    table: Table,
    backend: str | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "a",
    parallel_safe: bool = False,
) -> None:
    """Write a table to a Zarr store."""
    handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    table._set_backend(
        handler=handler,
        backend_name=backend,
    )
    table.consolidate()
