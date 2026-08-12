"""Implementation of the FeatureTableV1 class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from typing import Literal

from pydantic import BaseModel, Field

from ngio.tables._abstract_table import AbstractBaseTable
from ngio.tables.backends import BackendMeta, TableBackend, TabularData
from ngio.utils import NgioValueError
from ngio.utils._zarr_utils import ZarrGroupHandler


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class FeatureTableMeta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["feature_table"] = "feature_table"
    region: RegionMeta | None = None
    instance_key: str = "label"  # Legacy field, kept for compatibility
    # Backend metadata
    index_key: str | None = "label"
    index_type: Literal["int", "str"] | None = "int"
    # Columns optional types
    categorical_columns: list[str] = Field(default_factory=list)
    measurement_columns: list[str] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)


class FeatureTableV1(AbstractBaseTable):
    def __init__(
        self,
        table_data: TabularData | None = None,
        *,
        reference_label: str | None = None,
        meta: FeatureTableMeta | None = None,
    ) -> None:
        """Initialize the GenericTable."""
        if meta is None:
            meta = FeatureTableMeta()

        if reference_label is not None:
            path = f"../labels/{reference_label}"
            meta = FeatureTableMeta(region=RegionMeta(path=path))

        if table_data is not None and not isinstance(table_data, TabularData):
            raise NgioValueError(
                f"The table is not of type SupportedTables. Got {type(table_data)}"
            )

        if meta.index_key is None:
            meta.index_key = "label"

        if meta.index_type is None:
            meta.index_type = "int"

        meta.instance_key = meta.index_key

        super().__init__(
            table_data=table_data,
            meta=meta,
        )

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        num_rows = len(self.dataframe) if self.dataframe is not None else 0
        num_columns = len(self.dataframe.columns) if self.dataframe is not None else 0
        properties = f"num_rows={num_rows}, num_columns={num_columns}"
        if self.reference_label is not None:
            properties += f", reference_label={self.reference_label}"
        return f"FeatureTableV1({properties})"

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
        attrs: dict | None = None,
    ) -> "FeatureTableV1":
        return cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=FeatureTableMeta,
            attrs=attrs,
        )

    @property
    def meta(self) -> FeatureTableMeta:
        """Return the metadata of the table."""
        if not isinstance(self._meta, FeatureTableMeta):
            raise NgioValueError(
                "The metadata of the table is not of type FeatureTableMeta."
            )
        return self._meta

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "feature_table"

    @staticmethod
    def version() -> str:
        """The generic table does not have a version.

        Since does not follow a specific schema.
        """
        return "1"

    @property
    def reference_label(self) -> str | None:
        """Return the reference label."""
        path = self.meta.region
        if path is None:
            return None

        path = path.path
        path = path.split("/")[-1]
        return path
