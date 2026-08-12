"""Abstract class for handling OME-NGFF images."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from zarr.core.array import CompressorLike

from ngio.common._pyramid import ChunksLike, ShardsLike
from ngio.images._create_utils import init_image_like
from ngio.images._image import Image, ImagesContainer
from ngio.images._label import Label, LabelsContainer
from ngio.images._masked_image import MaskedImage, MaskedLabel
from ngio.ome_zarr_meta import (
    LabelMetaHandler,
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    Channel,
    DefaultNgffVersion,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
)
from ngio.ome_zarr_meta.ngio_specs._axes import AxesSetup
from ngio.ome_zarr_meta.ngio_specs._channels import ChannelsMeta
from ngio.tables import (
    ConditionTable,
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
    NgioError,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
    deprecated_alias,
)

logger = logging.getLogger(f"ngio:{__name__}")


def _try_get_table_container(
    handler: ZarrGroupHandler, create_mode: bool = True
) -> TablesContainer | None:
    """Return a default table container."""
    try:
        table_handler = handler.get_handler("tables", create_mode=create_mode)
        return TablesContainer(table_handler)
    except NgioError:
        return None


def _try_get_label_container(
    handler: ZarrGroupHandler,
    ngff_version: NgffVersions,
    axes_setup: AxesSetup | None = None,
    create_mode: bool = True,
) -> LabelsContainer | None:
    """Return a default label container."""
    try:
        label_handler = handler.get_handler("labels", create_mode=create_mode)
        return LabelsContainer(
            group_handler=label_handler,
            axes_setup=axes_setup,
            ngff_version=ngff_version,
        )
    except (NgioError, FileNotFoundError):
        return None


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

    @deprecated_alias(validate_paths="validate_arrays")
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        table_container: TablesContainer | None = None,
        label_container: LabelsContainer | None = None,
        axes_setup: AxesSetup | None = None,
        validate_arrays: bool = False,
    ) -> None:
        """Initialize the OmeZarrContainer.

        Args:
            group_handler: The Zarr group handler.
            table_container: The tables container.
            label_container: The labels container.
            axes_setup: Axes setup to load ome-zarr with non-standard axes
                configurations.
            validate_arrays: Whether to open every level listed in the multiscale
                metadata, so a missing or malformed array fails here rather than
                on first access.
        """
        self._group_handler = group_handler
        self._images_container = ImagesContainer(
            self._group_handler,
            axes_setup=axes_setup,
            validate_arrays=validate_arrays,
        )
        self._labels_container = label_container
        self._tables_container = table_container
        # Set when a read-only probe found no `/tables`; see
        # `_get_tables_container`.
        self._tables_absent = False

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

    def refresh(self) -> None:
        """Re-read every piece of metadata this container is holding.

        The answer to "someone else wrote to this container and I want to see
        it".

        Not a no-op under `cache=False`: the decoded metadata memo and each
        image's `dimensions` are held regardless of that flag, because both are
        derived from a `zarr.Array` handle that is itself fixed at construction.
        `cache=False` only means the raw attributes are re-read; this drops the
        derived values too.
        """
        self._group_handler.clean_cache()
        self._images_container._meta_handler.invalidate()
        # Rebuilt on next access against the refreshed handler.
        self._labels_container = None
        self._tables_container = None
        self._tables_absent = False

    @property
    def images_container(self) -> ImagesContainer:
        """Return the images container.

        Returns:
            ImagesContainer: The images container.
        """
        return self._images_container

    def _get_labels_container(self, create_mode: bool = True) -> LabelsContainer | None:
        """Return the labels container."""
        if self._labels_container is not None:
            return self._labels_container

        _labels_container = _try_get_label_container(
            self._group_handler,
            create_mode=create_mode,
            ngff_version=self.meta.version,
            axes_setup=self._images_container.axes_setup,
        )
        self._labels_container = _labels_container
        return self._labels_container

    @property
    def labels_container(self) -> LabelsContainer:
        """Return the labels container."""
        _labels_container = self._get_labels_container()
        if _labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return _labels_container

    def _get_tables_container(self, create_mode: bool = True) -> TablesContainer | None:
        """Return the tables container."""
        if self._tables_container is not None:
            return self._tables_container

        # "This image has no /tables" is worth remembering too. Only the
        # read-only probe may be remembered: a `create_mode=True` caller is
        # asking for the group to be made, so it has to try again even though
        # an earlier read-only probe found nothing.
        if self._tables_absent and not create_mode:
            return None

        _tables_container = _try_get_table_container(
            self._group_handler, create_mode=create_mode
        )
        self._tables_container = _tables_container
        self._tables_absent = _tables_container is None
        return self._tables_container

    @property
    def tables_container(self) -> TablesContainer:
        """Return the tables container."""
        _tables_container = self._get_tables_container()
        if _tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        return _tables_container

    @property
    def meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self.images_container.meta

    @property
    def axes_setup(self) -> AxesSetup:
        """Return the axes setup."""
        return self.images_container.axes_setup

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self.images_container.levels

    @property
    def level_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self.images_container.level_paths

    @property
    def is_3d(self) -> bool:
        """Return True if the image is 3D."""
        return self.images_container.is_3d

    @property
    def is_2d(self) -> bool:
        """Return True if the image is 2D."""
        return self.images_container.is_2d

    @property
    def is_time_series(self) -> bool:
        """Return True if the image is a time series."""
        return self.images_container.is_time_series

    @property
    def is_2d_time_series(self) -> bool:
        """Return True if the image is a 2D time series."""
        return self.images_container.is_2d_time_series

    @property
    def is_3d_time_series(self) -> bool:
        """Return True if the image is a 3D time series."""
        return self.images_container.is_3d_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return True if the image is multichannel."""
        return self.images_container.is_multi_channels

    @property
    def space_unit(self) -> str | None:
        """Return the space unit of the image."""
        return self.images_container.space_unit

    @property
    def time_unit(self) -> str | None:
        """Return the time unit of the image."""
        return self.images_container.time_unit

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        return self.images_container.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the list of wavelength of the image."""
        return self.images_container.wavelength_ids

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return self.images_container.num_channels

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by its label or wavelength ID."""
        return self.images_container.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def set_channel_meta(
        self,
        channel_meta: ChannelsMeta | None = None,
    ) -> None:
        """Set the channels metadata.

        Args:
            channel_meta: The channels metadata to set. If `None`, a default
                metadata is created from the number of channels in the image.
        """
        self._images_container.set_channel_meta(channel_meta=channel_meta)

    def set_channel_labels(
        self,
        labels: Sequence[str],
    ) -> None:
        """Update the labels of the channels.

        Args:
            labels (Sequence[str]): The new labels for the channels.
        """
        self._images_container.set_channel_labels(labels=labels)

    def set_channel_colors(
        self,
        colors: Sequence[str],
    ) -> None:
        """Update the colors of the channels.

        Args:
            colors (Sequence[str]): The new colors for the channels.
        """
        self._images_container.set_channel_colors(colors=colors)

    def set_channel_windows(
        self,
        starts_ends: Sequence[tuple[float, float]],
        min_max: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        """Update the channel windows.

        These values are used by viewers to set the display
        range of each channel.

        Args:
            starts_ends (Sequence[tuple[float, float]]): The start and end values
                for each channel.
            min_max (Sequence[tuple[float, float]] | None): The min and max values
                for each channel. If None, the min and max values will not be updated.
        """
        self._images_container.set_channel_windows(
            starts_ends=starts_ends,
            min_max=min_max,
        )

    def set_channel_windows_with_percentiles(
        self,
        percentiles: tuple[float, float] | list[tuple[float, float]] = (0.1, 99.9),
    ) -> None:
        """Update the channel windows using percentiles.

        Args:
            percentiles (tuple[float, float] | list[tuple[float, float]]):
                The start and end percentiles for each channel.
                If a single tuple is provided,
                the same percentiles will be used for all channels.
        """
        self._images_container.set_channel_windows_with_percentiles(
            percentiles=percentiles
        )

    def set_axes_units(
        self,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
        set_labels: bool = True,
    ) -> None:
        """Set the space and time units of the image axes.

        Args:
            space_unit: The unit of space.
            time_unit: The unit of time.
            set_labels: Whether to set the units for the labels as well.
        """
        if set_labels:
            for label_name in self.list_labels():
                label = self.get_label(label_name)
                label.set_axes_units(space_unit=space_unit, time_unit=time_unit)
        self._images_container.set_axes_units(
            space_unit=space_unit, time_unit=time_unit
        )

    def set_axes_names(
        self,
        axes_names: Sequence[str],
    ) -> None:
        """Set the axes names of the image.

        Args:
            axes_names (Sequence[str]): The axes names of the image.
        """
        self._images_container.set_axes_names(axes_names=axes_names)

    def set_name(
        self,
        name: str,
    ) -> None:
        """Set the name of the image in the metadata.

        This does not change the group name or any paths.

        Args:
            name (str): The name of the image.
        """
        self._images_container.set_name(name=name)

    def get_image(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Image:
        """Get an image at a specific level.

        Args:
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size: Select the pyramid level whose pixel size matches this one.
                A lookup key, not a value to write; to set a pixel size see
                `pixelsize` on the create/derive entry points.
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
            pixel_size: Select the pyramid level whose pixel size matches this one.
                A lookup key, not a value to write; to set a pixel size see
                `pixelsize` on the create/derive entry points.
                This is only used if path is None.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        """
        image = self.get_image(path=path, pixel_size=pixel_size, strict=strict)
        masking_label, masking_table = self._find_matching_masking_label(
            masking_label_name=masking_label_name,
            masking_table_name=masking_table_name,
            pixel_size=image.pixel_size,
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
        # Metadata parameters
        shape: Sequence[int] | None = None,
        pixelsize: float | tuple[float, float] | None = None,
        z_spacing: float | None = None,
        time_spacing: float | None = None,
        name: str | None = None,
        translation: Sequence[float] | None = None,
        channels_policy: Literal["squeeze", "same", "singleton"] | int = "same",
        channels_meta: Sequence[str | Channel] | None = None,
        ngff_version: NgffVersions | None = None,
        # Zarr Array parameters
        chunks: ChunksLike | None = None,
        shards: ShardsLike | None = None,
        dtype: str | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        compressors: CompressorLike | None = None,
        extra_array_kwargs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        # Copy from current image
        copy_labels: bool = False,
        copy_tables: bool = False,
    ) -> "OmeZarrContainer":
        """Derive a new OME-Zarr container from the current image.

        If a kwarg is not provided, the value from the reference image will be used.

        Args:
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_path (str | None): The path to the reference image in the image
                container.
            shape (Sequence[int] | None): The shape of the new image.
            pixelsize (float | tuple[float, float] | None): The pixel size of the new
                image.
                A value to write, not a lookup key; to select an existing
                level see `pixel_size` on the getters.
            z_spacing (float | None): The z spacing of the new image.
            time_spacing (float | None): The time spacing of the new image.
            name (str | None): The name of the new image.
            translation (Sequence[float] | None): The translation for each axis
                at the highest resolution level. Defaults to None.
            channels_policy (Literal["squeeze", "same", "singleton"] | int): Possible
                policies:
                - If "squeeze", the channels axis will be removed (no matter its size).
                - If "same", the channels axis will be kept as is (if it exists).
                - If "singleton", the channels axis will be set to size 1.
                - If an integer is provided, the channels axis will be changed to have
                    that size.
            channels_meta (Sequence[str | Channel] | None): The channels metadata
                of the new image.
            ngff_version (NgffVersions | None): The NGFF version to use.
            chunks (ChunksLike | None): The chunk shape of the new image.
            shards (ShardsLike | None): The shard shape of the new image.
            dtype (str | None): The data type of the new image.
            dimension_separator (Literal[".", "/"] | None): The separator to use for
                dimensions.
            compressors (CompressorLike | None): The compressors to use.
            extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
                the zarr array creation.
            overwrite (bool): Whether to overwrite an existing image. Defaults to False.
            copy_labels (bool): Whether to copy the labels from the current image.
                Defaults to False.
            copy_tables (bool): Whether to copy the tables from the current image.
                Defaults to False.

        Returns:
            OmeZarrContainer: The new derived OME-Zarr container.

        """
        new_container = self._images_container.derive(
            store=store,
            ref_path=ref_path,
            shape=shape,
            pixelsize=pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            name=name,
            translation=translation,
            channels_meta=channels_meta,
            channels_policy=channels_policy,
            ngff_version=ngff_version,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            dimension_separator=dimension_separator,
            compressors=compressors,
            extra_array_kwargs=extra_array_kwargs,
            overwrite=overwrite,
        )
        new_ome_zarr = OmeZarrContainer(
            group_handler=new_container._group_handler,
            validate_arrays=False,
            axes_setup=new_container.meta.axes_handler.axes_setup,
        )

        if copy_labels:
            self.labels_container._group_handler.copy_group(
                new_ome_zarr.labels_container._group_handler.group
            )

        if copy_tables:
            self.tables_container._group_handler.copy_group(
                new_ome_zarr.tables_container._group_handler.group
            )
        return new_ome_zarr

    def list_tables(self, filter_types: TypedTable | str | None = None) -> list[str]:
        """List all tables in the image."""
        table_container = self._get_tables_container(create_mode=False)
        if table_container is None:
            return []

        return table_container.list(
            filter_types=filter_types,
        )

    def list_roi_tables(self) -> list[str]:
        """List all ROI tables in the image.

        Returns `[]` when the image has no tables, matching `list_tables`.
        """
        table_container = self._get_tables_container(create_mode=False)
        if table_container is None:
            return []

        # One pass, not one per type: each `list(filter_types=...)` opens every
        # table to read its type, so asking twice read every document twice to
        # sort names the first pass had already sorted.
        types = table_container.table_types()
        return [
            name
            for name, table_type in types.items()
            if table_type in ("roi_table", "masking_roi_table")
        ]

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

    def get_table(self, name: str) -> Table:
        """Get a table from the image.

        Args:
            name (str): The name of the table.
        """
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

    def build_image_roi_table(self, name: str | None = "image") -> RoiTable:
        """Compute the ROI table for an image."""
        return self.get_image().build_image_roi_table(name=name)

    def build_masking_roi_table(self, label: str) -> MaskingRoiTable:
        """Compute the masking ROI table for a label."""
        return self.get_label(label).build_masking_roi_table()

    def add_table(
        self,
        name: str,
        table: Table,
        backend: TableBackend | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the image.

        If `backend` is `None` (default), the table's own backend is preserved.
        """
        self.tables_container.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def delete_table(self, name: str, missing_ok: bool = False) -> None:
        """Delete a table from the group.

        Args:
            name (str): The name of the table to delete.
            missing_ok (bool): If True, do not raise an error if the table does not
                exist.

        """
        table_container = self._get_tables_container(create_mode=False)
        if table_container is None and missing_ok:
            return
        if table_container is None:
            raise NgioValueError(
                f"No tables found in the image, cannot delete {name}. "
                "Set missing_ok=True to ignore this error."
            )
        table_container.delete(name=name, missing_ok=missing_ok)

    def list_labels(self) -> list[str]:
        """List all labels in the image."""
        label_container = self._get_labels_container(create_mode=False)
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
            pixel_size: Select the pyramid level whose pixel size matches this one.
                A lookup key, not a value to write; to set a pixel size see
                `pixelsize` on the create/derive entry points.
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
            pixel_size: Select the pyramid level whose pixel size matches this one.
                A lookup key, not a value to write; to set a pixel size see
                `pixelsize` on the create/derive entry points.
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
            pixel_size=label.pixel_size,
        )
        return MaskedLabel(
            group_handler=label._group_handler,
            path=label.path,
            meta_handler=label.meta_handler,
            label=masking_label,
            masking_roi_table=masking_table,
        )

    def delete_label(self, name: str, missing_ok: bool = False) -> None:
        """Delete a label from the group.

        Args:
            name (str): The name of the label to delete.
            missing_ok (bool): If True, do not raise an error if the label does not
                exist.

        """
        label_container = self._get_labels_container(create_mode=False)
        if label_container is None and missing_ok:
            return
        if label_container is None:
            raise NgioValueError(
                f"No labels found in the image, cannot delete {name}. "
                "Set missing_ok=True to ignore this error."
            )
        label_container.delete(name=name, missing_ok=missing_ok)

    def derive_label(
        self,
        name: str,
        ref_image: Image | Label | None = None,
        # Metadata parameters
        shape: Sequence[int] | None = None,
        pixelsize: float | tuple[float, float] | None = None,
        z_spacing: float | None = None,
        time_spacing: float | None = None,
        translation: Sequence[float] | None = None,
        channels_policy: Literal["same", "squeeze", "singleton"] | int = "squeeze",
        ngff_version: NgffVersions | None = None,
        # Zarr Array parameters
        chunks: ChunksLike | None = None,
        shards: ShardsLike | None = None,
        dtype: str | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        compressors: CompressorLike | None = None,
        extra_array_kwargs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "Label":
        """Derive a new label from an existing image or label.

        If a kwarg is not provided, the value from the reference image will be used.

        Args:
            name (str): The name of the new label.
            ref_image (Image | Label | None): The reference image to derive the new
                label from. If None, the first level image will be used.
            shape (Sequence[int] | None): The shape of the new label.
            pixelsize (float | tuple[float, float] | None): The pixel size of the new
                label.
                A value to write, not a lookup key; to select an existing
                level see `pixel_size` on the getters.
            z_spacing (float | None): The z spacing of the new label.
            time_spacing (float | None): The time spacing of the new label.
            translation (Sequence[float] | None): The translation for each axis
                at the highest resolution level. Defaults to None.
            channels_policy (Literal["same", "squeeze", "singleton"] | int): Possible
                policies:
                - If "squeeze", the channels axis will be removed (no matter its size).
                - If "same", the channels axis will be kept as is (if it exists).
                - If "singleton", the channels axis will be set to size 1.
                - If an integer is provided, the channels axis will be changed to have
                    that size.
                Defaults to "squeeze".
            ngff_version (NgffVersions | None): The NGFF version to use.
            chunks (ChunksLike | None): The chunk shape of the new label.
            shards (ShardsLike | None): The shard shape of the new label.
            dtype (str | None): The data type of the new label.
            dimension_separator (Literal[".", "/"] | None): The separator to use for
                dimensions.
            compressors (CompressorLike | None): The compressors to use.
            extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
                the zarr array creation.
            overwrite (bool): Whether to overwrite an existing label. Defaults to False.

        Returns:
            Label: The new derived label.

        """
        if ref_image is None:
            ref_image = self.get_image()
        return self.labels_container.derive(
            name=name,
            ref_image=ref_image,
            shape=shape,
            pixelsize=pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            translation=translation,
            channels_policy=channels_policy,
            ngff_version=ngff_version,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            dimension_separator=dimension_separator,
            compressors=compressors,
            extra_array_kwargs=extra_array_kwargs,
            overwrite=overwrite,
        )


def open_ome_zarr_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    axes_setup: AxesSetup | None = None,
    validate_arrays: bool = False,
) -> OmeZarrContainer:
    """Open an OME-Zarr image.

    Args:
        store: The Zarr store or group holding the image.
        cache: Whether to cache the zarr group metadata.
        mode: The access mode for the image.
        axes_setup: Axes setup to load ome-zarr with non-standard axes
            configurations.
        validate_arrays: Whether to open every level listed in the multiscale
            metadata up front, so a missing or malformed array fails here
            rather than on first access.
    """
    handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    return OmeZarrContainer(
        group_handler=handler,
        validate_arrays=validate_arrays,
        axes_setup=axes_setup,
    )


def open_image(
    store: StoreOrGroup,
    path: str | None = None,
    pixel_size: PixelSize | None = None,
    strict: bool = False,
    axes_setup: AxesSetup | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Image:
    """Open a single level image from an OME-Zarr image.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        path (str | None): The path to the image in the ome_zarr file.
        pixel_size: Select the pyramid level whose pixel size matches this one.
            A lookup key, not a value to write; to set a pixel size see
            `pixelsize` on the create/derive entry points.
        strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.
        axes_setup (AxesSetup | None): Axes setup to load ome-zarr with
            non-standard axes configurations.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The
            access mode for the image. Defaults to "r+".
    """
    group_handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    images_container = ImagesContainer(group_handler, axes_setup=axes_setup)
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
    strict: bool = False,
    axes_setup: AxesSetup | None = None,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Label:
    """Open a single level label from an OME-Zarr Label group.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        name (str | None): The name of the label. If None,
            we will try to open the store as a multiscale label.
        path (str | None): The path to the image in the ome_zarr file.
        pixel_size: Select the pyramid level whose pixel size matches this one.
            A lookup key, not a value to write; to set a pixel size see
            `pixelsize` on the create/derive entry points.
        strict (bool): Only used if the pixel size is provided. If True, the
            pixel size must match the image pixel size exactly. If False, the
            closest pixel size level will be returned.
        axes_setup (AxesSetup | None): Axes setup to load ome-zarr with
            non-standard axes configurations.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The access mode for the image. Defaults to "r+".

    """
    group_handler = ZarrGroupHandler(store=store, cache=cache, mode=mode)
    if name is None:
        label_meta_handler = LabelMetaHandler(group_handler, axes_setup=axes_setup)
        path = (
            label_meta_handler.get_meta()
            .get_dataset(path=path, pixel_size=pixel_size, strict=strict)
            .path
        )
        return Label(group_handler, path, label_meta_handler)

    labels_container = LabelsContainer(group_handler, axes_setup=axes_setup)
    return labels_container.get(
        name=name,
        path=path,
        pixel_size=pixel_size,
        strict=strict,
    )


def create_empty_ome_zarr(
    store: StoreOrGroup,
    shape: Sequence[int],
    pixelsize: float | tuple[float, float],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    scaling_factors: Sequence[float] | Literal["auto"] = "auto",
    levels: int | list[str] = 5,
    translation: Sequence[float] | None = None,
    space_unit: SpaceUnits = DefaultSpaceUnit,
    time_unit: TimeUnits = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    channels_meta: Sequence[str | Channel] | None = None,
    name: str | None = None,
    axes_setup: AxesSetup | None = None,
    ngff_version: NgffVersions = DefaultNgffVersion,
    chunks: ChunksLike = "auto",
    shards: ShardsLike | None = None,
    dtype: str = "uint16",
    dimension_separator: Literal[".", "/"] = "/",
    compressors: CompressorLike = "auto",
    extra_array_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> OmeZarrContainer:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        pixelsize (float | tuple[float, float] | None): The pixel size in x and y
            dimensions.
            A value to write, not a lookup key; to select an existing
            level see `pixel_size` on the getters.
        z_spacing (float): The spacing between z slices. Defaults to 1.0.
        time_spacing (float): The spacing between time points. Defaults to 1.0.
        scaling_factors (Sequence[float] | Literal["auto"]): The down-scaling factors
            for the pyramid levels. Defaults to "auto".
        levels (int | list[str]): The number of levels in the pyramid or a list of
            level names. Defaults to 5.
        translation (Sequence[float] | None): The translation for each axis.
            at the highest resolution level. Defaults to None.
        space_unit (SpaceUnits): The unit of space. Defaults to DefaultSpaceUnit.
        time_unit (TimeUnits): The unit of time. Defaults to DefaultTimeUnit.
        axes_names (Sequence[str] | None): The names of the axes. If None the
            canonical names are used. Defaults to None.
        channels_meta (Sequence[str | Channel] | None): The channels metadata.
            Defaults to None.
        name (str | None): The name of the image. Defaults to None.
        axes_setup (AxesSetup | None): Axes setup to create ome-zarr with
            non-standard axes configurations. Defaults to None.
        ngff_version (NgffVersions): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
        chunks (ChunksLike): The chunk shape. Defaults to "auto".
        shards (ShardsLike | None): The shard shape. Defaults to None.
        dtype (str): The data type of the image. Defaults to "uint16".
        dimension_separator (Literal[".", "/"]): The dimension separator to use.
            Defaults to "/".
        compressors (CompressorLike): The compressor to use. Defaults to "auto".
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation. Defaults to None.
        overwrite (bool): Whether to overwrite an existing image. Defaults to False.
    """
    handler, axes_setup = init_image_like(
        store=store,
        meta_type=NgioImageMeta,
        shape=shape,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        scaling_factors=scaling_factors,
        levels=levels,
        translation=translation,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        channels_meta=channels_meta,
        name=name,
        axes_setup=axes_setup,
        ngff_version=ngff_version,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        extra_array_kwargs=extra_array_kwargs,
        overwrite=overwrite,
    )

    return OmeZarrContainer(group_handler=handler, axes_setup=axes_setup)


def create_ome_zarr_from_array(
    store: StoreOrGroup,
    array: np.ndarray,
    pixelsize: float | tuple[float, float],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    scaling_factors: Sequence[float] | Literal["auto"] = "auto",
    levels: int | list[str] = 5,
    translation: Sequence[float] | None = None,
    space_unit: SpaceUnits = DefaultSpaceUnit,
    time_unit: TimeUnits = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    channels_meta: Sequence[str | Channel] | None = None,
    percentiles: tuple[float, float] = (0.1, 99.9),
    name: str | None = None,
    axes_setup: AxesSetup | None = None,
    ngff_version: NgffVersions = DefaultNgffVersion,
    chunks: ChunksLike = "auto",
    shards: ShardsLike | None = None,
    dimension_separator: Literal[".", "/"] = "/",
    compressors: CompressorLike = "auto",
    extra_array_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> OmeZarrContainer:
    """Create an OME-Zarr image from a numpy array.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        array (np.ndarray): The image data.
        pixelsize (float | tuple[float, float] | None): The pixel size in x and y
            dimensions.
            A value to write, not a lookup key; to select an existing
            level see `pixel_size` on the getters.
        z_spacing (float): The spacing between z slices. Defaults to 1.0.
        time_spacing (float): The spacing between time points. Defaults to 1.0.
        scaling_factors (Sequence[float] | Literal["auto"]): The down-scaling factors
            for the pyramid levels. Defaults to "auto".
        levels (int | list[str]): The number of levels in the pyramid or a list of
            level names. Defaults to 5.
        translation (Sequence[float] | None): The translation for each axis.
            at the highest resolution level. Defaults to None.
        space_unit (SpaceUnits): The unit of space. Defaults to DefaultSpaceUnit.
        time_unit (TimeUnits): The unit of time. Defaults to DefaultTimeUnit.
        axes_names (Sequence[str] | None): The names of the axes. If None the
            canonical names are used. Defaults to None.
        channels_meta (Sequence[str | Channel] | None): The channels metadata.
            Defaults to None.
        percentiles (tuple[float, float]): The percentiles of the channels for
            computing display ranges. Defaults to (0.1, 99.9).
        name (str | None): The name of the image. Defaults to None.
        axes_setup (AxesSetup | None): Axes setup to create ome-zarr with
            non-standard axes configurations. Defaults to None.
        ngff_version (NgffVersions): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
        chunks (ChunksLike): The chunk shape. Defaults to "auto".
        shards (ShardsLike | None): The shard shape. Defaults to None.
        dimension_separator (Literal[".", "/"]): The separator to use for
            dimensions. Defaults to "/".
        compressors (CompressorLike): The compressors to use. Defaults to "auto".
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation. Defaults to None.
        overwrite (bool): Whether to overwrite an existing image. Defaults to False.
    """
    if len(percentiles) != 2:
        raise NgioValueError(
            f"'percentiles' must be a tuple of two values. Got {percentiles}"
        )
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=array.shape,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        scaling_factors=scaling_factors,
        levels=levels,
        translation=translation,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        channels_meta=channels_meta,
        name=name,
        axes_setup=axes_setup,
        ngff_version=ngff_version,
        chunks=chunks,
        shards=shards,
        dtype=str(array.dtype),
        dimension_separator=dimension_separator,
        compressors=compressors,
        extra_array_kwargs=extra_array_kwargs,
        overwrite=overwrite,
    )
    # Populate through a cached view. Writing the array, building the pyramid
    # and computing the channel windows re-read the same few documents a dozen
    # times over, and this function is the only writer for the whole sequence —
    # the same reason `create_empty_plate` and `create_empty_well` build with
    # `cache=True`. The caller still gets an uncached container.
    working = open_ome_zarr_container(store=store, mode="r+", cache=True)
    image = working.get_image()
    image.set_array(array)
    image.consolidate()
    working.set_channel_windows_with_percentiles(percentiles=percentiles)

    # `ome_zarr` was opened before any of the writes above. It is uncached, so
    # it re-reads and would see them anyway; this is a guard against that
    # ceasing to be true, and it costs no store reads.
    ome_zarr.refresh()
    return ome_zarr
