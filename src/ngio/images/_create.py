"""Utility functions for working with OME-Zarr images."""

from collections.abc import Sequence
from typing import TypeVar

from ngio.common._pyramid import init_empty_pyramid
from ngio.ome_zarr_meta import (
    NgioImageMeta,
    NgioLabelMeta,
    PixelSize,
    get_image_meta_handler,
    get_label_meta_handler,
)
from ngio.ome_zarr_meta.ngio_specs import (
    DefaultNgffVersion,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
    canonical_axes_order,
    canonical_label_axes_order,
)
from ngio.utils import NgioValueError, StoreOrGroup, ZarrGroupHandler

_image_or_label_meta = TypeVar("_image_or_label_meta", NgioImageMeta, NgioLabelMeta)


def _init_generic_meta(
    meta_type: type[_image_or_label_meta],
    pixelsize: float,
    axes_names: Sequence[str],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    yx_scaling_factor: float | tuple[float, float] = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit,
    time_unit: TimeUnits | str | None = DefaultTimeUnit,
    name: str | None = None,
    version: NgffVersions = DefaultNgffVersion,
) -> tuple[_image_or_label_meta, list[float]]:
    """Initialize the metadata for an image or label."""
    scaling_factors = []
    for ax in axes_names:
        if ax == "z":
            scaling_factors.append(z_scaling_factor)
        elif ax in ["x"]:
            if isinstance(yx_scaling_factor, tuple):
                scaling_factors.append(yx_scaling_factor[1])
            else:
                scaling_factors.append(yx_scaling_factor)
        elif ax in ["y"]:
            if isinstance(yx_scaling_factor, tuple):
                scaling_factors.append(yx_scaling_factor[0])
            else:
                scaling_factors.append(yx_scaling_factor)
        else:
            scaling_factors.append(1.0)

    pixel_sizes = PixelSize(
        x=pixelsize,
        y=pixelsize,
        z=z_spacing,
        t=time_spacing,
        space_unit=space_unit,
        time_unit=time_unit,
    )

    meta = meta_type.default_init(
        name=name,
        levels=levels,
        axes_names=axes_names,
        pixel_size=pixel_sizes,
        scaling_factors=scaling_factors,
        version=version,
    )
    return meta, scaling_factors


def create_empty_label_container(
    store: StoreOrGroup,
    shape: Sequence[int],
    pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    yx_scaling_factor: float | tuple[float, float] = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit,
    time_unit: TimeUnits | str | None = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    name: str | None = None,
    chunks: Sequence[int] | None = None,
    dtype: str = "uint32",
    overwrite: bool = False,
    version: NgffVersions = DefaultNgffVersion,
) -> ZarrGroupHandler:
    """Create an empty label with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        yx_scaling_factor (float, optional): The down-scaling factor in x and y
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
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to DefaultVersion.

    """
    if axes_names is None:
        axes_names = canonical_label_axes_order()[-len(shape) :]

    if len(axes_names) != len(shape):
        raise NgioValueError(
            f"Number of axes names {axes_names} does not match the number of "
            f"dimensions {shape}."
        )

    meta, scaling_factors = _init_generic_meta(
        meta_type=NgioLabelMeta,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        yx_scaling_factor=yx_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        version=version,
    )

    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(store=store, mode=mode, cache=False)
    image_handler = get_label_meta_handler(version=version, group_handler=group_handler)
    image_handler.write_meta(meta)

    init_empty_pyramid(
        store=store,
        paths=meta.paths,
        scaling_factors=scaling_factors,
        ref_shape=shape,
        chunks=chunks,
        dtype=dtype,
        mode="a",
    )
    group_handler._mode = "r+"
    return group_handler


def create_empty_image_container(
    store: StoreOrGroup,
    shape: Sequence[int],
    pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    yx_scaling_factor: float | tuple[float, float] = 2,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit,
    time_unit: TimeUnits | str | None = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    name: str | None = None,
    chunks: Sequence[int] | None = None,
    dtype: str = "uint16",
    overwrite: bool = False,
    version: NgffVersions = DefaultNgffVersion,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        yx_scaling_factor (float, optional): The down-scaling factor in x and y
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
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to DefaultVersion.

    """
    if axes_names is None:
        axes_names = canonical_axes_order()[-len(shape) :]

    if len(axes_names) != len(shape):
        raise NgioValueError(
            f"Number of axes names {axes_names} does not match the number of "
            f"dimensions {shape}."
        )

    meta, scaling_factors = _init_generic_meta(
        meta_type=NgioImageMeta,
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        yx_scaling_factor=yx_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        version=version,
    )
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(store=store, mode=mode, cache=False)
    image_handler = get_image_meta_handler(version=version, group_handler=group_handler)
    image_handler.write_meta(meta)

    init_empty_pyramid(
        store=store,
        paths=meta.paths,
        scaling_factors=scaling_factors,
        ref_shape=shape,
        chunks=chunks,
        dtype=dtype,
        mode="a",
    )

    group_handler._mode = "r+"
    return group_handler
