"""A module for handling label images in OME-NGFF files."""

from collections.abc import Sequence
from typing import Literal

from ngio.common import compute_masking_roi
from ngio.images._abstract_image import AbstractImage
from ngio.images._create import create_empty_label_container
from ngio.images._image import Image
from ngio.ome_zarr_meta import (
    LabelMetaHandler,
    NgioLabelMeta,
    PixelSize,
    find_label_meta_handler,
)
from ngio.ome_zarr_meta.ngio_specs import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
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


class Label(AbstractImage[LabelMetaHandler]):
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
            meta_handler = find_label_meta_handler(group_handler)
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    def __repr__(self) -> str:
        """Return the string representation of the label."""
        return f"Label(path={self.path}, {self.dimensions})"

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        return self._meta_handler.meta

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
        self._meta_handler.write_meta(meta)

    def build_masking_roi_table(self) -> MaskingRoiTable:
        """Compute the masking ROI table."""
        return build_masking_roi_table(self)

    def consolidate(
        self,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        self._consolidate(
            order=0,
            mode=mode,
        )


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler

        # Validate the group
        # Either contains a labels attribute or is empty
        attrs = self._group_handler.load_attrs()
        if len(attrs) == 0:
            # It's an empty group
            pass
        elif "labels" in attrs and isinstance(attrs["labels"], list):
            # It's a valid group
            pass
        else:
            raise NgioValidationError(
                f"Invalid /labels group. "
                f"Expected a single labels attribute with a list of label names. "
                f"Found: {attrs}"
            )

    def list(self) -> list[str]:
        """Create the /labels group if it doesn't exist."""
        attrs = self._group_handler.load_attrs()
        return attrs.get("labels", [])

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

        group_handler = self._group_handler.derive_handler(name)
        label_meta_handler = find_label_meta_handler(group_handler)
        path = label_meta_handler.meta.get_dataset(
            path=path, pixel_size=pixel_size, strict=strict
        ).path
        return Label(group_handler, path, label_meta_handler)

    def derive(
        self,
        name: str,
        ref_image: Image | Label,
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
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_image (Image | Label): A reference image that will be used to create
                the new image.
            name (str): The name of the new image.
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
        existing_labels = self.list()
        if name in existing_labels and not overwrite:
            raise NgioValueError(
                f"Label '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        label_group = self._group_handler.get_group(name, create_mode=True)

        derive_label(
            store=label_group,
            ref_image=ref_image,
            name=name,
            shape=shape,
            pixel_size=pixel_size,
            axes_names=axes_names,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
        )

        if name not in existing_labels:
            existing_labels.append(name)
            self._group_handler.write_attrs({"labels": existing_labels})

        return self.get(name)


def derive_label(
    store: StoreOrGroup,
    ref_image: Image | Label,
    name: str,
    shape: Sequence[int] | None = None,
    pixel_size: PixelSize | None = None,
    axes_names: Sequence[str] | None = None,
    chunks: Sequence[int] | None = None,
    dtype: str = "uint32",
    overwrite: bool = False,
) -> None:
    """Create an empty OME-Zarr label from a reference image.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        ref_image (Image | Label): A reference image that will be used to
            create the new image.
        name (str): The name of the new image.
        shape (Sequence[int] | None): The shape of the new image.
        pixel_size (PixelSize | None): The pixel size of the new image.
        axes_names (Sequence[str] | None): The axes names of the new image.
            For labels, the channel axis is not allowed.
        chunks (Sequence[int] | None): The chunk shape of the new image.
        dtype (str): The data type of the new label.
        overwrite (bool): Whether to overwrite an existing image.

    Returns:
        None

    """
    ref_meta = ref_image.meta

    if shape is None:
        shape = ref_image.shape

    if pixel_size is None:
        pixel_size = ref_image.pixel_size

    if axes_names is None:
        axes_names = ref_meta.axes_mapper.axes_names
        c_axis = ref_meta.axes_mapper.get_index("c")
    else:
        if "c" in axes_names:
            raise NgioValidationError(
                "Labels cannot have a channel axis. "
                "Please remove the channel axis from the axes names."
            )
        c_axis = None

    if len(axes_names) != len(shape):
        raise NgioValidationError(
            "The axes names of the new image does not match the reference image."
            f"Got {axes_names} for shape {shape}."
        )

    if chunks is None:
        chunks = ref_image.chunks

    if len(chunks) != len(shape):
        raise NgioValidationError(
            "The chunks of the new image does not match the reference image."
            f"Got {chunks} for shape {shape}."
        )

    if c_axis is not None:
        # remove channel if present
        shape = list(shape)
        shape = shape[:c_axis] + shape[c_axis + 1 :]
        chunks = list(chunks)
        chunks = chunks[:c_axis] + chunks[c_axis + 1 :]
        axes_names = list(axes_names)
        axes_names = axes_names[:c_axis] + axes_names[c_axis + 1 :]

    _ = create_empty_label_container(
        store=store,
        shape=shape,
        pixelsize=ref_image.pixel_size.x,
        z_spacing=ref_image.pixel_size.z,
        time_spacing=ref_image.pixel_size.t,
        levels=ref_meta.levels,
        yx_scaling_factor=ref_meta.yx_scaling(),
        z_scaling_factor=ref_meta.z_scaling(),
        time_unit=ref_image.pixel_size.time_unit,
        space_unit=ref_image.pixel_size.space_unit,
        axes_names=axes_names,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        version=ref_meta.version,
        name=name,
    )
    return None


def build_masking_roi_table(label: Label) -> MaskingRoiTable:
    """Compute the masking ROI table for a label."""
    array = label.get_as_dask(axes_order=["t", "z", "y", "x"])
    rois = compute_masking_roi(array, label.pixel_size)
    return MaskingRoiTable(rois, reference_label=label.meta.name)
