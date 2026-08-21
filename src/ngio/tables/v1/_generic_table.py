"""Implementation of a generic table class."""

from ngio.tables._abstract_table import AbstractBaseTable
from ngio.tables.backends import BackendMeta, TableBackend
from ngio.utils import ZarrGroupHandler


class GenericTableMeta(BackendMeta):
    """Metadata for the generic table.

    This is used to store metadata for a generic table.
    It does not have a specific definition.
    """

    table_version: str | None = "1"
    type: str | None = "generic_table"


class GenericTable(AbstractBaseTable):
    """Class to a non-specific table.

    This can be used to load any table that does not have
    a specific definition.
    """

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_table"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
        attrs: dict | None = None,
    ) -> "GenericTable":
        return cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=BackendMeta,
            attrs=attrs,
        )
