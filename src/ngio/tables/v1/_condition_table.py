"""Implementation of a generic table class."""

from ngio.tables._abstract_table import AbstractBaseTable
from ngio.tables.backends import (
    BackendMeta,
    TableBackend,
    TabularData,
)
from ngio.utils import ZarrGroupHandler


class ConditionTableMeta(BackendMeta):
    """Metadata for the condition table."""

    table_version: str | None = "1"
    type: str | None = "condition_table"


class ConditionTableV1(AbstractBaseTable):
    """Condition table class.

    This class is used to load a condition table.
    The condition table is a generic table that does not
    have a specific definition.

    It is used to store information about the particular conditions
    used to generate the data.
    - How much drug was used in the experiment
    - What treatment was used
    - etc.
    """

    def __init__(
        self,
        table_data: TabularData | None = None,
        *,
        meta: ConditionTableMeta | None = None,
    ) -> None:
        """Initialize the ConditionTable."""
        if meta is None:
            meta = ConditionTableMeta()

        super().__init__(
            table_data=table_data,
            meta=meta,
        )

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "condition_table"

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
    ) -> "ConditionTableV1":
        return cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=ConditionTableMeta,
            attrs=attrs,
        )
