"""Implementation of the ROI Table class.

This class follows the roi_table specification at:
https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
"""

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from ngio.common import Roi
from ngio.tables._abstract_table import (
    AbstractBaseTable,
    TabularData,
)
from ngio.tables.backends import (
    BackendMeta,
    TableBackend,
    convert_to_pandas,
    normalize_pandas_df,
)
from ngio.utils import (
    NgioTableValidationError,
    NgioValueError,
    ZarrGroupHandler,
    ngio_warn,
)

REQUIRED_COLUMNS = [
    "x_micrometer",
    "y_micrometer",
    "z_micrometer",
    "len_x_micrometer",
    "len_y_micrometer",
    "len_z_micrometer",
]

#####################
# Optional columns are not validated at the moment
# only a warning is raised if non optional columns are present
#####################

TIME_COLUMNS = [
    "t_second",
    "len_t_second",
]

ORIGIN_COLUMNS = [
    "x_micrometer_original",
    "y_micrometer_original",
    "z_micrometer_original",
]

TRANSLATION_COLUMNS = ["translation_x", "translation_y", "translation_z"]

PLATE_COLUMNS = [
    "plate_name",
    "row",
    "column",
    "path_in_well",
    "path_in_plate",
    "acquisition_id",
    "acquisition_name",
]

INDEX_COLUMNS = [
    "FieldIndex",
    "label",
]

OPTIONAL_COLUMNS = ORIGIN_COLUMNS + TRANSLATION_COLUMNS + PLATE_COLUMNS + INDEX_COLUMNS


def _check_optional_columns(col_name: str) -> None:
    """Check if the column name is in the optional columns."""
    if col_name not in OPTIONAL_COLUMNS + TIME_COLUMNS:
        ngio_warn(f"Column {col_name} is not in the optional columns.")


def _dataframe_to_rois(
    dataframe: pd.DataFrame,
    required_columns: list[str] = REQUIRED_COLUMNS,
) -> dict[str, Roi]:
    """Convert a DataFrame to a WorldCooROI object."""
    # Validate the columns of the DataFrame
    _required_columns = set(dataframe.columns).intersection(set(required_columns))
    if len(_required_columns) != len(required_columns):
        raise NgioTableValidationError(
            f"Could not find required columns: {_required_columns} in the table."
        )

    extra_columns = set(dataframe.columns).difference(
        set(required_columns + TIME_COLUMNS)
    )

    for col in extra_columns:
        _check_optional_columns(col)

    label_is_index = True if dataframe.index.name == "label" else False

    extras = {}

    rois = {}
    for row in dataframe.itertuples(index=True):
        # check if optional columns are present
        if len(extra_columns) > 0:
            extras = {col: getattr(row, col, None) for col in extra_columns}

        z_micrometer = getattr(row, "z_micrometer", None)
        z_length_micrometer = getattr(row, "len_z_micrometer", None)

        t_second = getattr(row, "t_second", None)
        t_length_second = getattr(row, "len_t_second", None)

        if label_is_index:
            label = int(row.Index)  # type: ignore (type can not be known here, but should be castable to int)
        else:
            label = getattr(row, "label", None)

        roi = Roi(
            name=str(row.Index),
            x=row.x_micrometer,  # type: ignore (type can not be known here)
            y=row.y_micrometer,  # type: ignore (type can not be known here)
            z=z_micrometer,
            t=t_second,
            x_length=row.len_x_micrometer,  # type: ignore (type can not be known here)
            y_length=row.len_y_micrometer,  # type: ignore (type can not be known here)
            z_length=z_length_micrometer,
            t_length=t_length_second,
            unit="micrometer",
            label=label,
            **extras,
        )
        rois[roi.name] = roi
    return rois


def _rois_to_dataframe(rois: dict[str, Roi], index_key: str | None) -> pd.DataFrame:
    """Convert a list of WorldCooROI objects to a DataFrame."""
    data = []
    for roi in rois.values():
        # This normalization is necessary for backward compatibility
        z_micrometer = roi.z if roi.z is not None else 0.0
        len_z_micrometer = roi.z_length if roi.z_length is not None else 1.0

        row = {
            index_key: roi.name,
            "x_micrometer": roi.x,
            "y_micrometer": roi.y,
            "z_micrometer": z_micrometer,
            "len_x_micrometer": roi.x_length,
            "len_y_micrometer": roi.y_length,
            "len_z_micrometer": len_z_micrometer,
        }

        if roi.t is not None:
            row["t_second"] = roi.t

        if roi.t_length is not None:
            row["len_t_second"] = roi.t_length

        if roi.label is not None and index_key != "label":
            row["label"] = roi.label

        extra = roi.model_extra or {}
        for col in extra:
            _check_optional_columns(col)
            row[col] = extra[col]
        data.append(row)

    dataframe = pd.DataFrame(data)
    dataframe = normalize_pandas_df(dataframe, index_key=index_key)
    return dataframe


class RoiDictWrapper:
    """A wrapper for a dictionary of ROIs to provide a consistent interface."""

    def __init__(self, rois: Iterable[Roi]) -> None:
        self._rois_by_name = {roi.name: roi for roi in rois}
        self._rois_by_label = {roi.label: roi for roi in rois if roi.label is not None}

    def get_by_name(self, name: str, default: Roi | None = None) -> Roi | None:
        """Get an ROI by its name."""
        return self._rois_by_name.get(name, default)

    def get_by_label(self, label: int, default: Roi | None = None) -> Roi | None:
        """Get an ROI by its label."""
        return self._rois_by_label.get(label, default)

    def _add_roi(self, roi: Roi, overwrite: bool = False) -> None:
        """Add an ROI to the wrapper."""
        if roi.name in self._rois_by_name and not overwrite:
            raise NgioValueError(f"ROI with name {roi.name} already exists.")

        self._rois_by_name[roi.name] = roi
        if roi.label is not None:
            self._rois_by_label[roi.label] = roi

    def add_rois(self, rois: Roi | Iterable[Roi], overwrite: bool = False) -> None:
        """Add ROIs to the wrapper."""
        if isinstance(rois, Roi):
            rois = [rois]

        for roi in rois:
            self._add_roi(roi, overwrite=overwrite)

    def to_list(self) -> list[Roi]:
        """Return the list of ROIs."""
        return list(self._rois_by_name.values())

    def to_dataframe(self, index_key: str | None = None) -> pd.DataFrame:
        """Convert the ROIs to a DataFrame."""
        return _rois_to_dataframe(self._rois_by_name, index_key=index_key)

    @classmethod
    def from_dataframe(
        cls, dataframe: pd.DataFrame, required_columns: list[str] = REQUIRED_COLUMNS
    ) -> "RoiDictWrapper":
        """Create a RoiDictWrapper from a DataFrame."""
        rois = _dataframe_to_rois(dataframe, required_columns=required_columns)
        return cls(rois.values())


def _table_to_rois(
    table: TabularData,
    index_key: str | None = None,
    index_type: Literal["int", "str"] | None = None,
    required_columns: list[str] = REQUIRED_COLUMNS,
) -> tuple[pd.DataFrame, RoiDictWrapper]:
    """Convert a table to a dictionary of ROIs.

    Args:
        table: The table to convert.
        index_key: The column name to use as the index of the DataFrame.
        index_type: The type of the index column in the DataFrame.
        required_columns: The required columns in the DataFrame.

    Returns:
        A tuple containing the DataFrame and a RoiDictWrapper with the ROIs.
    """
    dataframe = convert_to_pandas(
        table,
        index_key=index_key,
        index_type=index_type,
    )
    roi_dict_wrapper = RoiDictWrapper.from_dataframe(
        dataframe, required_columns=required_columns
    )
    return dataframe, roi_dict_wrapper


class GenericRoiTableV1(AbstractBaseTable):
    def __init__(
        self,
        *,
        rois: Iterable[Roi] | None = None,
        meta: BackendMeta,
        required_columns: list[str] = REQUIRED_COLUMNS,
    ) -> None:
        table = None

        self._rois: RoiDictWrapper | None = None
        if rois is not None:
            self._rois = RoiDictWrapper(rois)
            table = self._rois.to_dataframe(index_key=meta.index_key)

        self._required_columns = required_columns
        super().__init__(table_data=table, meta=meta)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        prop = f"num_rois={len(rois)}"
        class_name = self.__class__.__name__
        return f"{class_name}({prop})"

    @staticmethod
    def table_type() -> str:
        """Return the type of the table."""
        return "generic_roi_table"

    @staticmethod
    def version() -> Literal["1"]:
        """Return the version of the fractal table."""
        return "1"

    @property
    def table_data(self) -> TabularData:
        """Return the table."""
        if self._rois is None:
            return super().table_data

        if len(self.rois()) > 0:
            self._table_data = self._rois.to_dataframe(index_key=self.meta.index_key)
        return super().table_data

    def set_table_data(
        self, table_data: TabularData | None = None, refresh: bool = False
    ) -> None:
        if table_data is not None:
            if not isinstance(table_data, TabularData):
                raise NgioValueError(
                    "The table must be a pandas DataFrame, polars LazyFrame, "
                    " or AnnData object."
                )

            table_data, rois = _table_to_rois(
                table_data,
                index_key=self.index_key,
                index_type=self.index_type,
                required_columns=REQUIRED_COLUMNS,
            )
            self._table_data = table_data
            self._rois = rois
            return None

        if self._table_data is not None and not refresh:
            return None

        if self._table_backend is None:
            raise NgioValueError(
                "The table does not have a DataFrame in memory nor a backend."
            )

        table_data, rois = _table_to_rois(
            self._table_backend.load(),
            index_key=self.index_key,
            index_type=self.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        self._table_data = table_data
        self._rois = rois

    def _check_rois(self) -> None:
        """Load the ROIs from the table.

        If the ROIs are already loaded, do nothing.
        If the ROIs are not loaded, load them from the table.
        """
        if self._rois is None:
            self._rois = RoiDictWrapper.from_dataframe(
                self.dataframe, required_columns=self._required_columns
            )

    def rois(self) -> list[Roi]:
        """List all ROIs in the table."""
        self._check_rois()
        if self._rois is None:
            return []
        return self._rois.to_list()

    def add(self, roi: Roi | Iterable[Roi], overwrite: bool = False) -> None:
        """Append ROIs to the current table.

        Args:
            roi: A single ROI or a list of ROIs to add to the table.
            overwrite: If True, overwrite existing ROIs with the same name.
        """
        if isinstance(roi, Roi):
            roi = [roi]

        self._check_rois()
        if self._rois is None:
            self._rois = RoiDictWrapper([])

        self._rois.add_rois(roi, overwrite=overwrite)

    def get(self, roi_name: str) -> Roi:
        """Get an ROI from the table."""
        self._check_rois()
        if self._rois is None:
            self._rois = RoiDictWrapper([])

        roi = self._rois.get_by_name(roi_name)
        if roi is None:
            raise NgioValueError(f"ROI with name {roi_name} not found in the table.")
        return roi

    @classmethod
    def from_table_data(
        cls, table_data: TabularData, meta: BackendMeta
    ) -> "GenericRoiTableV1":
        """Create a new ROI table from a table data."""
        _, rois = _table_to_rois(
            table=table_data,
            index_key=meta.index_key,
            index_type=meta.index_type,
            required_columns=REQUIRED_COLUMNS,
        )
        return cls(rois=rois.to_list(), meta=meta)


class RoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["roi_table"] = "roi_table"
    index_key: str | None = "FieldIndex"
    index_type: Literal["str", "int"] | None = "str"


class RoiTableV1(GenericRoiTableV1):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self, rois: Iterable[Roi] | None = None, *, meta: RoiTableV1Meta | None = None
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = RoiTableV1Meta()

        if meta.index_key is None:
            meta.index_key = "FieldIndex"

        if meta.index_type is None:
            meta.index_type = "str"
        super().__init__(meta=meta, rois=rois)

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
    ) -> "RoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=RoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["roi_table"]:
        """Return the type of the table."""
        return "roi_table"


class RegionMeta(BaseModel):
    """Metadata for the region."""

    path: str


class MaskingRoiTableV1Meta(BackendMeta):
    """Metadata for the ROI table."""

    table_version: Literal["1"] = "1"
    type: Literal["masking_roi_table"] = "masking_roi_table"
    region: RegionMeta | None = None
    instance_key: str = "label"
    index_key: str | None = "label"
    index_type: Literal["int", "str"] | None = "int"


class MaskingRoiTableV1(GenericRoiTableV1):
    """Class to handle fractal ROI tables.

    To know more about the ROI table format, please refer to the
    specification at:
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
    """

    def __init__(
        self,
        rois: Iterable[Roi] | None = None,
        *,
        reference_label: str | None = None,
        meta: MaskingRoiTableV1Meta | None = None,
    ) -> None:
        """Create a new ROI table."""
        if meta is None:
            meta = MaskingRoiTableV1Meta()

        if reference_label is not None:
            path = f"../labels/{reference_label}"
            meta.region = RegionMeta(path=path)

        if meta.index_key is None:
            meta.index_key = "label"

        if meta.index_type is None:
            meta.index_type = "int"
        meta.instance_key = meta.index_key
        super().__init__(meta=meta, rois=rois)

    def __repr__(self) -> str:
        """Return a string representation of the table."""
        rois = self.rois()
        if self.reference_label is not None:
            prop = f"num_rois={len(rois)}, reference_label={self.reference_label}"
        else:
            prop = f"num_rois={len(rois)}"
        return f"MaskingRoiTableV1({prop})"

    @classmethod
    def from_handler(
        cls,
        handler: ZarrGroupHandler,
        backend: TableBackend | None = None,
    ) -> "MaskingRoiTableV1":
        table = cls._from_handler(
            handler=handler,
            backend=backend,
            meta_model=MaskingRoiTableV1Meta,
        )
        return table

    @staticmethod
    def table_type() -> Literal["masking_roi_table"]:
        """Return the type of the table."""
        return "masking_roi_table"

    @property
    def meta(self) -> MaskingRoiTableV1Meta:
        """Return the metadata of the table."""
        if not isinstance(self._meta, MaskingRoiTableV1Meta):
            raise NgioValueError(
                "The metadata of the table is not of type MaskingRoiTableV1Meta."
            )
        return self._meta

    @property
    def reference_label(self) -> str | None:
        """Return the reference label."""
        path = self.meta.region
        if path is None:
            return None

        path = path.path
        path = path.split("/")[-1]
        return path

    def get_label(self, label: int) -> Roi:
        """Get an ROI by label."""
        self._check_rois()
        if self._rois is None:
            self._rois = RoiDictWrapper([])
        roi = self._rois.get_by_label(label)

        if roi is None:
            raise NgioValueError(f"ROI with label {label} not found.")
        return roi
