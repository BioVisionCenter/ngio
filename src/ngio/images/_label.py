"""A module for handling label images in OME-NGFF files."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from zarr.core.array import CompressorLike

from ngio.common import compute_masking_roi
from ngio.common._pyramid import ChunksLike, ShardsLike
from ngio.images._abstract_image import AbstractImage, abstract_derive
from ngio.images._image import Image
from ngio.ome_zarr_meta import (
    LabelMetaHandler,
    LabelsGroupMetaHandler,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    PixelSize,
    update_ngio_labels_group_meta,
)
from ngio.ome_zarr_meta.ngio_specs import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
)
from ngio.tables import MaskingRoiTable
from ngio.utils import (
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class Label(AbstractImage):
    """Placeholder class for a label."""

    get_as_numpy = AbstractImage._get_as_numpy
    get_as_dask = AbstractImage._get_as_dask
    get_array = AbstractImage._get_array
    get_roi_as_numpy = AbstractImage._get_roi_as_numpy
    get_roi_as_dask = AbstractImage._get_roi_as_dask
    get_roi = AbstractImage._get_roi
    set_array = AbstractImage._set_array
    set_roi = AbstractImage._set_roi

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = LabelMetaHandler(group_handler)
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    def __repr__(self) -> str:
        """Return the string representation of the label."""
        return f"Label(path={self.path}, {self.dimensions})"

    @property
    def meta_handler(self) -> LabelMetaHandler:
        """Return the metadata handler."""
        assert isinstance(self._meta_handler, LabelMetaHandler)
        return self._meta_handler

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        meta = self.meta_handler.get_meta()
        assert isinstance(meta, NgioLabelMeta)
        return meta

    def set_axes_unit(
        self,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
    ) -> None:
        """Set the axes unit of the image.

        Args:
            space_unit (SpaceUnits): The space unit of the image.
            time_unit (TimeUnits): The time unit of the image.
        """
        meta = self.meta
        meta = meta.to_units(space_unit=space_unit, time_unit=time_unit)
        self.meta_handler.update_meta(meta)

    def build_masking_roi_table(
        self, axes_order: Sequence[str] | None = None
    ) -> MaskingRoiTable:
        """Compute the masking ROI table."""
        return build_masking_roi_table(self, axes_order=axes_order)

    def consolidate(
        self,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        self._consolidate(
            order="nearest",
            mode=mode,
        )


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        ngff_version: NgffVersions | None = None,
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        # If the group is empty, initialize the metadata
        try:
            self._meta_handler = LabelsGroupMetaHandler(group_handler)
        except NgioValidationError:
            if ngff_version is None:
                raise NgioValueError(
                    "The /labels group is missing metadata. "
                    "Please provide the ngff_version to initialize it."
                ) from None
            meta = NgioLabelsGroupMeta(labels=[], version=ngff_version)
            update_ngio_labels_group_meta(
                group_handler=group_handler,
                ngio_meta=meta,
            )
            self._group_handler = self._group_handler.reopen_handler()
            self._meta_handler = LabelsGroupMetaHandler(group_handler)

    @property
    def meta(self) -> NgioLabelsGroupMeta:
        """Return the metadata."""
        meta = self._meta_handler.get_meta()
        return meta

    def list(self) -> list[str]:
        """Create the /labels group if it doesn't exist."""
        return self.meta.labels

    def get(
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
        if name not in self.list():
            raise NgioValueError(
                f"Label '{name}' not found in the Labels group. "
                f"Available labels: {self.list()}"
            )

        group_handler = self._group_handler.get_handler(name)
        label_meta_handler = LabelMetaHandler(group_handler)
        path = (
            label_meta_handler.get_meta()
            .get_dataset(path=path, pixel_size=pixel_size, strict=strict)
            .path
        )
        return Label(group_handler, path, label_meta_handler)

    def delete(self, name: str, missing_ok: bool = False) -> None:
        """Delete a label from the group.

        Args:
            name (str): The name of the label to delete.
            missing_ok (bool): If True, do not raise an error if the label does not
                exist.

        """
        existing_labels = self.list()
        if name not in existing_labels:
            if missing_ok:
                return
            raise NgioValueError(
                f"Label '{name}' not found in the Labels group. "
                f"Available labels: {existing_labels}"
            )

        self._group_handler.delete_group(name)
        existing_labels.remove(name)
        update_meta = NgioLabelsGroupMeta(
            labels=existing_labels, version=self.meta.version
        )
        self._meta_handler.update_meta(update_meta)

    def derive(
        self,
        name: str,
        ref_image: Image | Label,
        # Metadata parameters
        shape: Sequence[int] | None = None,
        pixelsize: float | tuple[float, float] | None = None,
        z_spacing: float | None = None,
        time_spacing: float | None = None,
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
        # Deprecated arguments
        labels: Sequence[str] | None = None,
        pixel_size: PixelSize | None = None,
    ) -> "Label":
        """Create an empty OME-Zarr image from an existing image.

        If a kwarg is not provided, the value from the reference image will be used.

        Args:
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_image (Image | Label): The reference image to derive the new image from.
            shape (Sequence[int] | None): The shape of the new image.
            pixelsize (float | tuple[float, float] | None): The pixel size of the new
                image.
            z_spacing (float | None): The z spacing of the new image.
            time_spacing (float | None): The time spacing of the new image.
            scaling_factors (Sequence[float] | Literal["auto"] | None): The scaling
                factors of the new image.
            axes_names (Sequence[str] | None): The axes names of the new image.
            name (str | None): The name of the new image.
            channels_meta (Sequence[str | Channel] | None): The channels metadata
                of the new image.
            channels_policy (Literal["squeeze", "same", "singleton"] | int):
                Possible policies:
                - If "squeeze", the channels axis will be removed (no matter its size).
                - If "same", the channels axis will be kept as is (if it exists).
                - If "singleton", the channels axis will be set to size 1.
                - If an integer is provided, the channels axis will be changed to have
                    that size.
            ngff_version (NgffVersions | None): The NGFF version to use.
            chunks (ChunksLike | None): The chunk shape of the new image.
            shards (ShardsLike | None): The shard shape of the new image.
            dtype (str | None): The data type of the new image.
            dimension_separator (DIMENSION_SEPARATOR | None): The separator to use for
                dimensions.
            compressors (CompressorLike | None): The compressors to use.
            extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
                the zarr array creation.
            overwrite (bool): Whether to overwrite an existing image.
            labels (Sequence[str] | None): The labels of the new image.
                This argument is deprecated please use channels_meta instead.
            pixel_size (PixelSize | None): The pixel size of the new image.
                This argument is deprecated please use pixelsize, z_spacing,
                and time_spacing instead.

        Returns:
            Label: The new derived label.

        """
        existing_labels = self.list()
        if name in existing_labels and not overwrite:
            raise NgioValueError(
                f"Label '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        label_group = self._group_handler.get_group(name, create_mode=True)

        derive_label(
            ref_image=ref_image,
            store=label_group,
            shape=shape,
            pixelsize=pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            name=name,
            channels_policy=channels_policy,
            ngff_version=ngff_version,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            dimension_separator=dimension_separator,
            compressors=compressors,
            extra_array_kwargs=extra_array_kwargs,
            overwrite=overwrite,
            labels=labels,
            pixel_size=pixel_size,
        )

        if name not in existing_labels:
            existing_labels.append(name)

        update_meta = NgioLabelsGroupMeta(
            labels=existing_labels, version=self.meta.version
        )
        self._meta_handler.update_meta(update_meta)
        return self.get(name)


def derive_label(
    *,
    store: StoreOrGroup,
    ref_image: Image | Label,
    # Metadata parameters
    shape: Sequence[int] | None = None,
    pixelsize: float | tuple[float, float] | None = None,
    z_spacing: float | None = None,
    time_spacing: float | None = None,
    name: str | None = None,
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
    # Deprecated arguments
    labels: Sequence[str] | None = None,
    pixel_size: PixelSize | None = None,
) -> ZarrGroupHandler:
    """Derive a new OME-Zarr label from an existing image or label.

    If a kwarg is not provided, the value from the reference image will be used.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the label in.
        ref_image (Image | Label): The reference image to derive the new label from.
        shape (Sequence[int] | None): The shape of the new label.
        pixelsize (float | tuple[float, float] | None): The pixel size of the new label.
        z_spacing (float | None): The z spacing of the new label.
        time_spacing (float | None): The time spacing of the new label.
        name (str | None): The name of the new label.
        channels_policy (Literal["squeeze", "same", "singleton"] | int): Possible
            policies:
            - If "squeeze", the channels axis will be removed (no matter its size).
            - If "same", the channels axis will be kept as is (if it exists).
            - If "singleton", the channels axis will be set to size 1.
            - If an integer is provided, the channels axis will be changed to have that
                size.
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
        labels (Sequence[str] | None): Deprecated. This argument is deprecated,
            please use channels_meta instead.
        pixel_size (PixelSize | None): Deprecated. The pixel size of the new label.
            This argument is deprecated, please use pixelsize, z_spacing,
            and time_spacing instead.

    Returns:
        ZarrGroupHandler: The group handler of the new label.

    """
    if dtype is None and isinstance(ref_image, Image):
        dtype = "uint32"
    group_handler = abstract_derive(
        ref_image=ref_image,
        meta_type=NgioLabelMeta,
        store=store,
        shape=shape,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        name=name,
        channels_meta=None,
        channels_policy=channels_policy,
        ngff_version=ngff_version,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        extra_array_kwargs=extra_array_kwargs,
        overwrite=overwrite,
        labels=labels,
        pixel_size=pixel_size,
    )
    return group_handler


def build_masking_roi_table(
    label: Label, axes_order: Sequence[str] | None = None
) -> MaskingRoiTable:
    """Compute the masking ROI table for a label."""
    axes_order = axes_order or label.axes
    array = label.get_as_dask(axes_order=axes_order)
    rois = compute_masking_roi(array, label.pixel_size, axes_order=axes_order)
    return MaskingRoiTable(rois, reference_label=label.meta.name)
