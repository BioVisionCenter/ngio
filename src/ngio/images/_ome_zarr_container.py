"""Abstract class for handling OME-NGFF images."""

import warnings
from collections.abc import Sequence

import numpy as np

from ngio.images._create import create_empty_image_container
from ngio.images._image import Image, ImagesContainer
from ngio.images._label import Label, LabelsContainer
from ngio.images._masked_image import MaskedImage, MaskedLabel
from ngio.ome_zarr_meta import NgioImageMeta, PixelSize, find_label_meta_handler
from ngio.ome_zarr_meta.ngio_specs import (
    DefaultNgffVersion,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
)
from ngio.tables import (
    ConditionTable,
    DefaultTableBackend,
    FeatureTable,
    GenericRoiTable,
    MaskingRoiTable,
    RoiTable,
    Table,
    TableBackend,
    TablesContainer,
    TableType,
    TypedTable,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _default_table_container(handler: ZarrGroupHandler) -> TablesContainer | None:
    """Return a default table container."""
    success, table_handler = handler.safe_derive_handler("tables")
    if success and isinstance(table_handler, ZarrGroupHandler):
        return TablesContainer(table_handler)


def _default_label_container(handler: ZarrGroupHandler) -> LabelsContainer | None:
    """Return a default label container."""
    success, label_handler = handler.safe_derive_handler("labels")
    if success and isinstance(label_handler, ZarrGroupHandler):
        return LabelsContainer(label_handler)


class OmeZarrContainer:
    """This class is an object representation of an OME-Zarr image.

    It provides methods to access:
        - The multiscale image metadata
        - To open images at different levels of resolution
        - To access labels and tables associated with the image.
        - To derive new images, labels, and add tables to the image.
        - To modify the image metadata, such as axes units and channel metadata.

    Attributes:
        images_container (ImagesContainer): The container for the images.
        labels_container (LabelsContainer): The container for the labels.
        tables_container (TablesContainer): The container for the tables.

    """

    _images_container: ImagesContainer
    _labels_container: LabelsContainer | None
    _tables_container: TablesContainer | None

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        table_container: TablesContainer | None = None,
        label_container: LabelsContainer | None = None,
        validate_paths: bool = False,
    ) -> None:
        """Initialize the OmeZarrContainer.

        Args:
            group_handler (ZarrGroupHandler): The Zarr group handler.
            table_container (TablesContainer | None): The tables container.
            label_container (LabelsContainer | None): The labels container.
            validate_paths (bool): Whether to validate the paths of the image multiscale
        """
        self._group_handler = group_handler
        self._images_container = ImagesContainer(self._group_handler)

        self._labels_container = label_container
        self._tables_container = table_container

        if validate_paths:
            for level_path in self._images_container.levels_paths:
                self.get_image(path=level_path)

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        num_labels = len(self.list_labels())
        num_tables = len(self.list_tables())

        base_str = f"OmeZarrContainer(levels={self.levels}"
        if num_labels > 0 and num_labels < 3:
            base_str += f", labels={self.list_labels()}"
        elif num_labels >= 3:
            base_str += f", #labels={num_labels}"
        if num_tables > 0 and num_tables < 3:
            base_str += f", tables={self.list_tables()}"
        elif num_tables >= 3:
            base_str += f", #tables={num_tables}"
        base_str += ")"
        return base_str

    @property
    def images_container(self) -> ImagesContainer:
        """Return the images container.

        Returns:
            ImagesContainer: The images container.
        """
        return self._images_container

    def _get_labels_container(self) -> LabelsContainer | None:
        """Return the labels container."""
        if self._labels_container is None:
            _labels_container = _default_label_container(self._group_handler)
            if _labels_container is None:
                return None
            self._labels_container = _labels_container
        return self._labels_container

    @property
    def labels_container(self) -> LabelsContainer:
        """Return the labels container."""
        _labels_container = self._get_labels_container()
        if _labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return _labels_container

    def _get_tables_container(self) -> TablesContainer | None:
        """Return the tables container."""
        if self._tables_container is None:
            _tables_container = _default_table_container(self._group_handler)
            if _tables_container is None:
                return None
            self._tables_container = _tables_container
        return self._tables_container

    @property
    def tables_container(self) -> TablesContainer:
        """Return the tables container."""
        _tables_container = self._get_tables_container()
        if _tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        return _tables_container

    @property
    def image_meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self._images_container.meta

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._images_container.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._images_container.levels_paths

    @property
    def is_3d(self) -> bool:
        """Return True if the image is 3D."""
        return self.get_image().is_3d

    @property
    def is_2d(self) -> bool:
        """Return True if the image is 2D."""
        return self.get_image().is_2d

    @property
    def is_time_series(self) -> bool:
        """Return True if the image is a time series."""
        return self.get_image().is_time_series

    @property
    def is_2d_time_series(self) -> bool:
        """Return True if the image is a 2D time series."""
        return self.get_image().is_2d_time_series

    @property
    def is_3d_time_series(self) -> bool:
        """Return True if the image is a 3D time series."""
        return self.get_image().is_3d_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return True if the image is multichannel."""
        return self.get_image().is_multi_channels

    @property
    def space_unit(self) -> str | None:
        """Return the space unit of the image."""
        return self.image_meta.space_unit

    @property
    def time_unit(self) -> str | None:
        """Return the time unit of the image."""
        return self.image_meta.time_unit

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        image = self.get_image()
        return image.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the list of wavelength of the image."""
        image = self.get_image()
        return image.wavelength_ids

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return len(self.channel_labels)

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by its label or wavelength ID."""
        image = self.get_image()
        return image.channels_meta.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def set_channel_meta(
        self,
        labels: Sequence[str] | int | None = None,
        wavelength_id: Sequence[str] | None = None,
        percentiles: tuple[float, float] | None = None,
        colors: Sequence[str] | None = None,
        active: Sequence[bool] | None = None,
        **omero_kwargs: dict,
    ) -> None:
        """Create a ChannelsMeta object with the default unit."""
        self._images_container.set_channel_meta(
            labels=labels,
            wavelength_id=wavelength_id,
            start=None,
            end=None,
            percentiles=percentiles,
            colors=colors,
            active=active,
            **omero_kwargs,
        )

    def set_channel_percentiles(
        self,
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
    ) -> None:
        """Update the percentiles of the image."""
        self._images_container.set_channel_percentiles(
            start_percentile=start_percentile, end_percentile=end_percentile
        )

    def set_axes_units(
        self,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
        set_labels: bool = True,
    ) -> None:
        """Set the units of the image.

        Args:
            space_unit (SpaceUnits): The unit of space.
            time_unit (TimeUnits): The unit of time.
            set_labels (bool): Whether to set the units for the labels as well.
        """
        self._images_container.set_axes_unit(space_unit=space_unit, time_unit=time_unit)
        if not set_labels:
            return
        for label_name in self.list_labels():
            label = self.get_label(label_name)
            label.set_axes_unit(space_unit=space_unit, time_unit=time_unit)

    def get_image(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Image:
        """Get an image at a specific level.

        Args:
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size (PixelSize | None): The pixel size of the image.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.

        """
        return self._images_container.get(
            path=path, pixel_size=pixel_size, strict=strict
        )

    def _find_matching_masking_label(
        self,
        masking_label_name: str | None = None,
        masking_table_name: str | None = None,
        pixel_size: PixelSize | None = None,
    ) -> tuple[Label, MaskingRoiTable]:
        if masking_label_name is not None and masking_table_name is not None:
            # Both provided
            masking_label = self.get_label(
                name=masking_label_name, pixel_size=pixel_size, strict=False
            )
            masking_table = self.get_masking_roi_table(name=masking_table_name)

        elif masking_label_name is not None and masking_table_name is None:
            # Only the label provided
            masking_label = self.get_label(
                name=masking_label_name, pixel_size=pixel_size, strict=False
            )

            for table_name in self.list_roi_tables():
                table = self.get_generic_roi_table(name=table_name)
                if isinstance(table, MaskingRoiTable):
                    if table.reference_label == masking_label_name:
                        masking_table = table
                        break
            else:
                masking_table = masking_label.build_masking_roi_table()

        elif masking_table_name is not None and masking_label_name is None:
            # Only the table provided
            masking_table = self.get_masking_roi_table(name=masking_table_name)

            if masking_table.reference_label is None:
                raise NgioValueError(
                    f"Masking table {masking_table_name} does not have a reference "
                    "label. Please provide the masking_label_name explicitly."
                )
            masking_label = self.get_label(
                name=masking_table.reference_label,
                pixel_size=pixel_size,
                strict=False,
            )
        else:
            raise NgioValueError(
                "Neither masking_label_name nor masking_table_name were provided."
            )
        return masking_label, masking_table

    def get_masked_image(
        self,
        masking_label_name: str | None = None,
        masking_table_name: str | None = None,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> MaskedImage:
        """Get a masked image at a specific level.

        Args:
            masking_label_name (str | None): The name of the masking label to use.
                If None, the masking table must be provided.
            masking_table_name (str | None): The name of the masking table to use.
                If None, the masking label must be provided.
            path (str | None): The path to the image in the ome_zarr file.
                If None, the first level will be used.
            pixel_size (PixelSize | None): The pixel size of the image.
                This is only used if path is None.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        """
        image = self.get_image(path=path, pixel_size=pixel_size, strict=strict)
        masking_label, masking_table = self._find_matching_masking_label(
            masking_label_name=masking_label_name,
            masking_table_name=masking_table_name,
            pixel_size=pixel_size,
        )
        return MaskedImage(
            group_handler=image._group_handler,
            path=image.path,
            meta_handler=image.meta_handler,
            label=masking_label,
            masking_roi_table=masking_table,
        )

    def derive_image(
        self,
        store: StoreOrGroup,
        ref_path: str | None = None,
        shape: Sequence[int] | None = None,
        labels: Sequence[str] | None = None,
        pixel_size: PixelSize | None = None,
        axes_names: Sequence[str] | None = None,
        name: str | None = None,
        chunks: Sequence[int] | None = None,
        dtype: str | None = None,
        copy_labels: bool = False,
        copy_tables: bool = False,
        overwrite: bool = False,
    ) -> "OmeZarrContainer":
        """Create an empty OME-Zarr container from an existing image.

        Args:
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_path (str | None): The path to the reference image in
                the image container.
            shape (Sequence[int] | None): The shape of the new image.
            labels (Sequence[str] | None): The labels of the new image.
            pixel_size (PixelSize | None): The pixel size of the new image.
            axes_names (Sequence[str] | None): The axes names of the new image.
            chunks (Sequence[int] | None): The chunk shape of the new image.
            dtype (str | None): The data type of the new image.
            name (str | None): The name of the new image.
            copy_labels (bool): Whether to copy the labels from the reference image.
            copy_tables (bool): Whether to copy the tables from the reference image.
            overwrite (bool): Whether to overwrite an existing image.

        Returns:
            OmeZarrContainer: The new image container.

        """
        _ = self._images_container.derive(
            store=store,
            ref_path=ref_path,
            shape=shape,
            labels=labels,
            pixel_size=pixel_size,
            axes_names=axes_names,
            name=name,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
        )

        handler = ZarrGroupHandler(
            store, cache=self._group_handler.use_cache, mode=self._group_handler.mode
        )

        new_ome_zarr = OmeZarrContainer(
            group_handler=handler,
            validate_paths=False,
        )

        if copy_labels:
            self.labels_container._group_handler.copy_handler(
                new_ome_zarr.labels_container._group_handler
            )

        if copy_tables:
            self.tables_container._group_handler.copy_handler(
                new_ome_zarr.tables_container._group_handler
            )
        return new_ome_zarr

    def list_tables(self, filter_types: TypedTable | str | None = None) -> list[str]:
        """List all tables in the image."""
        table_container = self._get_tables_container()
        if table_container is None:
            return []

        return table_container.list(
            filter_types=filter_types,
        )

    def list_roi_tables(self) -> list[str]:
        """List all ROI tables in the image."""
        masking_roi = self.tables_container.list(
            filter_types="masking_roi_table",
        )
        roi = self.tables_container.list(
            filter_types="roi_table",
        )
        return masking_roi + roi

    def get_roi_table(self, name: str) -> RoiTable:
        """Get a ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, RoiTable):
            raise NgioValueError(f"Table {name} is not a ROI table. Got {type(table)}")
        return table

    def get_masking_roi_table(self, name: str) -> MaskingRoiTable:
        """Get a masking ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, MaskingRoiTable):
            raise NgioValueError(
                f"Table {name} is not a masking ROI table. Got {type(table)}"
            )
        return table

    def get_feature_table(self, name: str) -> FeatureTable:
        """Get a feature table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, FeatureTable):
            raise NgioValueError(
                f"Table {name} is not a feature table. Got {type(table)}"
            )
        return table

    def get_generic_roi_table(self, name: str) -> GenericRoiTable:
        """Get a generic ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, GenericRoiTable):
            raise NgioValueError(
                f"Table {name} is not a generic ROI table. Got {type(table)}"
            )
        return table

    def get_condition_table(self, name: str) -> ConditionTable:
        """Get a condition table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, ConditionTable):
            raise NgioValueError(
                f"Table {name} is not a condition table. Got {type(table)}"
            )
        return table

    def get_table(self, name: str, check_type: TypedTable | None = None) -> Table:
        """Get a table from the image.

        Args:
            name (str): The name of the table.
            check_type (TypedTable | None): Deprecated. Please use
                'get_table_as' instead, or one of the type specific
                get_*table() methods.

        """
        if check_type is not None:
            warnings.warn(
                "The 'check_type' argument is deprecated, and will be removed in "
                "ngio=0.3. Use 'get_table_as' instead or one of the "
                "type specific get_*table() methods.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.tables_container.get(name=name, strict=False)

    def get_table_as(
        self,
        name: str,
        table_cls: type[TableType],
        backend: TableBackend | None = None,
    ) -> TableType:
        """Get a table from the image as a specific type.

        Args:
            name (str): The name of the table.
            table_cls (type[TableType]): The type of the table.
            backend (TableBackend | None): The backend to use. If None,
                the default backend is used.
        """
        return self.tables_container.get_as(
            name=name,
            table_cls=table_cls,
            backend=backend,
        )

    def build_image_roi_table(self, name: str = "image") -> RoiTable:
        """Compute the ROI table for an image."""
        return self.get_image().build_image_roi_table(name=name)

    def build_masking_roi_table(self, label: str) -> MaskingRoiTable:
        """Compute the masking ROI table for a label."""
        return self.get_label(label).build_masking_roi_table()

    def add_table(
        self,
        name: str,
        table: Table,
        backend: TableBackend = DefaultTableBackend,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the image."""
        self.tables_container.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def list_labels(self) -> list[str]:
        """List all labels in the image."""
        label_container = self._get_labels_container()
        if label_container is None:
            return []
        return label_container.list()

    def get_label(
        self,
        name: str,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Label:
        """Get a label from the group.

        Args:
            name (str): The name of the label.
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size (PixelSize | None): The pixel size of the image.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        """
        return self.labels_container.get(
            name=name, path=path, pixel_size=pixel_size, strict=strict
        )

    def get_masked_label(
        self,
        label_name: str,
        masking_label_name: str | None = None,
        masking_table_name: str | None = None,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> MaskedLabel:
        """Get a masked image at a specific level.

        Args:
            label_name (str): The name of the label.
            masking_label_name (str | None): The name of the masking label.
            masking_table_name (str | None): The name of the masking table.
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size (PixelSize | None): The pixel size of the image.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        """
        label = self.get_label(
            name=label_name, path=path, pixel_size=pixel_size, strict=strict
        )
        masking_label, masking_table = self._find_matching_masking_label(
            masking_label_name=masking_label_name,
            masking_table_name=masking_table_name,
            pixel_size=pixel_size,
        )
        return MaskedLabel(
            group_handler=label._group_handler,
            path=label.path,
            meta_handler=label.meta_handler,
            label=masking_label,
            masking_roi_table=masking_table,
        )

    def derive_label(
        self,
        name: str,
        ref_image: Image | Label | None = None,
        shape: Sequence[int] | None = None,
        pixel_size: PixelSize | None = None,
        axes_names: Sequence[str] | None = None,
        chunks: Sequence[int] | None = None,
        dtype: str = "uint32",
        overwrite: bool = False,
    ) -> "Label":
        """Create an empty OME-Zarr label from a reference image.

        And add the label to the /labels group.

        Args:
            name (str): The name of the new image.
            ref_image (Image | Label | None): A reference image that will be used
                to create the new image.
            shape (Sequence[int] | None): The shape of the new image.
            pixel_size (PixelSize | None): The pixel size of the new image.
            axes_names (Sequence[str] | None): The axes names of the new image.
                For labels, the channel axis is not allowed.
            chunks (Sequence[int] | None): The chunk shape of the new image.
            dtype (str): The data type of the new label.
            overwrite (bool): Whether to overwrite an existing image.

        Returns:
            Label: The new label.

        """
        if ref_image is None:
            ref_image = self.get_image()
        return self.labels_container.derive(
            name=name,
            ref_image=ref_image,
            shape=shape,
            pixel_size=pixel_size,
            axes_names=axes_names,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
        )


def open_ome_zarr_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    validate_arrays: bool = True,
) -> OmeZarrContainer:
    """Open an OME-Zarr image."""
    handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    return OmeZarrContainer(
        group_handler=handler,
        validate_paths=validate_arrays,
    )


def open_image(
    store: StoreOrGroup,
    path: str | None = None,
    pixel_size: PixelSize | None = None,
    strict: bool = True,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Image:
    """Open a single level image from an OME-Zarr image.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        path (str | None): The path to the image in the ome_zarr file.
        pixel_size (PixelSize | None): The pixel size of the image.
        strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The
            access mode for the image. Defaults to "r+".
    """
    group_handler = ZarrGroupHandler(store, cache, mode)
    images_container = ImagesContainer(group_handler)
    return images_container.get(
        path=path,
        pixel_size=pixel_size,
        strict=strict,
    )


def open_label(
    store: StoreOrGroup,
    name: str | None = None,
    path: str | None = None,
    pixel_size: PixelSize | None = None,
    strict: bool = True,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Label:
    """Open a single level label from an OME-Zarr Label group.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        name (str | None): The name of the label. If None,
            we will try to open the store as a multiscale label.
        path (str | None): The path to the image in the ome_zarr file.
        pixel_size (PixelSize | None): The pixel size of the image.
        strict (bool): Only used if the pixel size is provided. If True, the
            pixel size must match the image pixel size exactly. If False, the
            closest pixel size level will be returned.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The access mode for the image. Defaults to "r+".

    """
    group_handler = ZarrGroupHandler(store, cache, mode)
    if name is None:
        label_meta_handler = find_label_meta_handler(group_handler)
        path = label_meta_handler.meta.get_dataset(
            path=path, pixel_size=pixel_size, strict=strict
        ).path
        return Label(group_handler, path, label_meta_handler)

    labels_container = LabelsContainer(group_handler)
    return labels_container.get(
        name=name,
        path=path,
        pixel_size=pixel_size,
        strict=strict,
    )


def create_empty_ome_zarr(
    store: StoreOrGroup,
    shape: Sequence[int],
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits = DefaultSpaceUnit,
    time_unit: TimeUnits = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    name: str | None = None,
    chunks: Sequence[int] | None = None,
    dtype: str = "uint16",
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_colors: Sequence[str] | None = None,
    channel_active: Sequence[bool] | None = None,
    overwrite: bool = False,
    version: NgffVersions = DefaultNgffVersion,
) -> OmeZarrContainer:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits, optional): The unit of space. Defaults to
            DefaultSpaceUnit.
        time_unit (TimeUnits, optional): The unit of time. Defaults to
            DefaultTimeUnit.
        axes_names (Sequence[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Sequence[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        dtype (str, optional): The data type of the image. Defaults to "uint16".
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_colors (Sequence[str] | None, optional): The colors of the channels.
            Defaults to None.
        channel_active (Sequence[bool] | None, optional): Whether the channels are
            active. Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (NgffVersion, optional): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
    """
    handler = create_empty_image_container(
        store=store,
        shape=shape,
        pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        yx_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        version=version,
    )

    ome_zarr = OmeZarrContainer(group_handler=handler)
    ome_zarr.set_channel_meta(
        labels=channel_labels,
        wavelength_id=channel_wavelengths,
        percentiles=None,
        colors=channel_colors,
        active=channel_active,
    )
    return ome_zarr


def create_ome_zarr_from_array(
    store: StoreOrGroup,
    array: np.ndarray,
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits = DefaultSpaceUnit,
    time_unit: TimeUnits = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    percentiles: tuple[float, float] | None = (0.1, 99.9),
    channel_colors: Sequence[str] | None = None,
    channel_active: Sequence[bool] | None = None,
    name: str | None = None,
    chunks: Sequence[int] | None = None,
    overwrite: bool = False,
    version: NgffVersions = DefaultNgffVersion,
) -> OmeZarrContainer:
    """Create an OME-Zarr image from a numpy array.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        array (np.ndarray): The image data.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits, optional): The unit of space. Defaults to
            DefaultSpaceUnit.
        time_unit (TimeUnits, optional): The unit of time. Defaults to
            DefaultTimeUnit.
        axes_names (Sequence[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Sequence[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        percentiles (tuple[float, float] | None, optional): The percentiles of the
            channels. Defaults to None.
        channel_colors (Sequence[str] | None, optional): The colors of the channels.
            Defaults to None.
        channel_active (Sequence[bool] | None, optional): Whether the channels are
            active. Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
    """
    handler = create_empty_image_container(
        store=store,
        shape=array.shape,
        pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        yx_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        chunks=chunks,
        dtype=str(array.dtype),
        overwrite=overwrite,
        version=version,
    )

    ome_zarr = OmeZarrContainer(group_handler=handler)
    image = ome_zarr.get_image()
    image.set_array(array)
    image.consolidate()
    ome_zarr.set_channel_meta(
        labels=channel_labels,
        wavelength_id=channel_wavelengths,
        percentiles=percentiles,
        colors=channel_colors,
        active=channel_active,
    )
    return ome_zarr
