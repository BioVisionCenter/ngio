"""Utility functions for working with OME-Zarr images."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar

from zarr.core.array import CompressorLike

from ngio.common._pyramid import ChunksLike, ImagePyramidBuilder, ShardsLike
from ngio.ome_zarr_meta import (
    NgioImageMeta,
    NgioLabelMeta,
    update_ngio_meta,
)
from ngio.ome_zarr_meta.ngio_specs import (
    AxesHandler,
    Channel,
    ChannelsMeta,
    DefaultNgffVersion,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    NgffVersions,
    SpaceUnits,
    TimeUnits,
    build_canonical_axes_handler,
    canonical_axes_order,
    canonical_label_axes_order,
)
from ngio.ome_zarr_meta.ngio_specs._axes import AxesSetup
from ngio.utils import NgioValueError, StoreOrGroup, ZarrGroupHandler

_image_or_label_meta = TypeVar("_image_or_label_meta", NgioImageMeta, NgioLabelMeta)


def _build_axes_handler(
    *,
    shape: tuple[int, ...],
    axes_names: Sequence[str] | None,
    default_channel_order: tuple[str, ...],
    space_units: SpaceUnits | str | None = DefaultSpaceUnit,
    time_units: TimeUnits | str | None = DefaultTimeUnit,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = False,
) -> AxesHandler:
    """Compute axes names for given shape."""
    if axes_names is None:
        axes_names = default_channel_order[-len(shape) :]
    # Validate length
    if len(axes_names) != len(shape):
        raise NgioValueError(
            f"Number of axes names {axes_names} does not match the number of "
            f"dimensions {shape}."
        )
    return build_canonical_axes_handler(
        axes_names=axes_names,
        space_units=space_units,
        time_units=time_units,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def _align_to_axes(
    *,
    values: dict[str, float],
    axes_handler: AxesHandler,
    default_value: float = 1.0,
) -> tuple[float, ...]:
    """Align given values to axes names."""
    aligned_values = [default_value] * len(axes_handler.axes_names)
    for ax, value in values.items():
        index = axes_handler.get_index(ax)
        if index is not None:
            aligned_values[index] = value
    return tuple(aligned_values)


def _check_deprecated_scaling_factors(
    *,
    yx_scaling_factor: float | tuple[float, float] | None = None,
    z_scaling_factor: float | None = None,
    scaling_factors: Sequence[float] | Literal["auto"] = "auto",
    shape: tuple[int, ...],
) -> Sequence[float] | Literal["auto"]:
    if yx_scaling_factor is not None or z_scaling_factor is not None:
        warnings.warn(
            "The 'yx_scaling_factor' and 'z_scaling_factor' arguments are deprecated "
            "and will be removed in future versions. Please use the 'scaling_factors' "
            "argument instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if scaling_factors != "auto":
            raise NgioValueError(
                "Cannot use both 'scaling_factors' and deprecated "
                "'yx_scaling_factor'/'z_scaling_factor' arguments."
            )
        if isinstance(yx_scaling_factor, tuple):
            if len(yx_scaling_factor) != 2:
                raise NgioValueError(
                    "yx_scaling_factor tuple must have length 2 for y and x scaling."
                )
            y_scale = yx_scaling_factor[0]
            x_scale = yx_scaling_factor[1]
        else:
            y_scale = yx_scaling_factor if yx_scaling_factor is not None else 2.0
            x_scale = yx_scaling_factor if yx_scaling_factor is not None else 2.0
        z_scale = z_scaling_factor if z_scaling_factor is not None else 1.0
        scaling_factors = (z_scale, x_scale, y_scale)
        if len(scaling_factors) < len(shape):
            padding = (1.0,) * (len(shape) - len(scaling_factors))
            scaling_factors = padding + scaling_factors

        return scaling_factors
    return scaling_factors


def _compute_scaling_factors(
    *,
    scaling_factors: Sequence[float] | Literal["auto"],
    shape: tuple[int, ...],
    axes_handler: AxesHandler,
    xy_scaling_factor: float | tuple[float, float] | None = None,
    z_scaling_factor: float | None = None,
) -> tuple[float, ...]:
    """Compute scaling factors for given axes names."""
    # TODO remove with ngio 0.6
    scaling_factors = _check_deprecated_scaling_factors(
        yx_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        scaling_factors=scaling_factors,
        shape=shape,
    )
    if scaling_factors == "auto":
        return _align_to_axes(
            values={
                "x": 2.0,
                "y": 2.0,
                "z": 1.0,
            },
            axes_handler=axes_handler,
        )
    if len(scaling_factors) != len(shape):
        raise NgioValueError(
            "Length of scaling_factors does not match the number of dimensions."
        )
    return tuple(scaling_factors)


def _compute_base_scale(
    *,
    pixelsize: float | tuple[float, float],
    z_spacing: float,
    time_spacing: float,
    axes_handler: AxesHandler,
) -> tuple[float, ...]:
    """Compute base scale for given axes names."""
    if isinstance(pixelsize, tuple):
        if len(pixelsize) != 2:
            raise NgioValueError(
                "pixelsize tuple must have length 2 for y and x pixel sizes."
            )
        x_size = pixelsize[1]
        y_size = pixelsize[0]
    else:
        x_size = pixelsize
        y_size = pixelsize
    return _align_to_axes(
        values={
            "x": x_size,
            "y": y_size,
            "z": z_spacing,
            "t": time_spacing,
        },
        axes_handler=axes_handler,
    )


def _create_image_like_group(
    *,
    store: StoreOrGroup,
    pyramid_builder: ImagePyramidBuilder,
    meta: _image_or_label_meta,
    overwrite: bool = False,
) -> ZarrGroupHandler:
    """Advanced create empty image container function placeholder."""
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(
        store=store, mode=mode, cache=False, zarr_format=meta.zarr_format
    )
    update_ngio_meta(group_handler, meta)
    # Reopen in r+ mode
    group_handler = group_handler.reopen_handler()
    # Write the pyramid
    pyramid_builder.to_zarr(group=group_handler.group)
    return group_handler


def _add_channels_meta(
    *,
    meta: _image_or_label_meta,
    channels_meta: Sequence[str | Channel] | None = None,
) -> _image_or_label_meta:
    """Create ChannelsMeta from given channels_meta input."""
    if isinstance(meta, NgioLabelMeta):
        if channels_meta is not None:
            raise NgioValueError(
                "Cannot add channels_meta to NgioLabelMeta. "
                "Labels do not have channels."
            )
        else:
            return meta
    if channels_meta is None:
        return meta
    list_of_channels = []
    for c in channels_meta:
        if isinstance(c, str):
            channel = Channel.default_init(label=c)
        elif isinstance(c, Channel):
            channel = c
        else:
            raise NgioValueError(
                "channels_meta must be a list of strings or Channel objects."
            )
        list_of_channels.append(channel)

    channels_meta_ = ChannelsMeta(channels=list_of_channels)
    meta.set_channels_meta(channels_meta=channels_meta_)
    return meta


def init_image_like(
    *,
    # Where to create the image
    store: StoreOrGroup,
    # Ngff image parameters
    meta_type: type[_image_or_label_meta],
    shape: Sequence[int],
    pixelsize: float | tuple[float, float],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    scaling_factors: Sequence[float] | Literal["auto"] = "auto",
    levels: int | list[str] = 5,
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit,
    time_unit: TimeUnits | str | None = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    name: str | None = None,
    channels_meta: Sequence[str | Channel] | None = None,
    ngff_version: NgffVersions = DefaultNgffVersion,
    # Zarr Array parameters
    chunks: ChunksLike = "auto",
    shards: ShardsLike | None = None,
    dtype: str = "uint16",
    dimension_separator: Literal[".", "/"] = "/",
    compressors: CompressorLike = "auto",
    extra_array_kwargs: Mapping[str, Any] | None = None,
    # internal axes configuration for advanced use cases
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = False,
    # Whether to overwrite existing image
    overwrite: bool = False,
    # Deprecated arguments
    yx_scaling_factor: float | tuple[float, float] | None = None,
    z_scaling_factor: float | None = None,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr image with the given shape and metadata."""
    shape = tuple(shape)
    if meta_type is NgioImageMeta:
        default_axes_order = canonical_axes_order()
    else:
        default_axes_order = canonical_label_axes_order()

    axes_handler = _build_axes_handler(
        shape=shape,
        axes_names=axes_names,
        default_channel_order=default_axes_order,
        space_units=space_unit,
        time_units=time_unit,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )
    base_scale = _compute_base_scale(
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        axes_handler=axes_handler,
    )
    scaling_factors = _compute_scaling_factors(
        scaling_factors=scaling_factors,
        shape=shape,
        axes_handler=axes_handler,
        xy_scaling_factor=yx_scaling_factor,
        z_scaling_factor=z_scaling_factor,
    )
    if isinstance(levels, int):
        levels_paths = tuple(str(i) for i in range(levels))
    else:
        levels_paths = tuple(levels)

    pyramid_builder = ImagePyramidBuilder.from_scaling_factors(
        levels_paths=levels_paths,
        scaling_factors=scaling_factors,
        base_shape=shape,
        base_scale=base_scale,
        axes=axes_handler.axes_names,
        chunks=chunks,
        data_type=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        shards=shards,
        zarr_format=2 if ngff_version == "0.4" else 3,
        other_array_kwargs=extra_array_kwargs,
    )
    meta = meta_type.default_init(
        levels=[p.path for p in pyramid_builder.levels],
        axes_handler=axes_handler,
        scales=[p.scale for p in pyramid_builder.levels],
        translations=[None for _ in pyramid_builder.levels],
        name=name,
        version=ngff_version,
    )
    meta = _add_channels_meta(meta=meta, channels_meta=channels_meta)
    # Keep this creation at the end to avoid partial creations on errors
    return _create_image_like_group(
        store=store,
        pyramid_builder=pyramid_builder,
        meta=meta,
        overwrite=overwrite,
    )


def init_image_like_from_shapes(
    *,
    # Where to create the image
    store: StoreOrGroup,
    # Ngff image parameters
    meta_type: type[_image_or_label_meta],
    shapes: Sequence[tuple[int, ...]],
    pixelsize: float | tuple[float, float],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: list[str] | None = None,
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit,
    time_unit: TimeUnits | str | None = DefaultTimeUnit,
    axes_names: Sequence[str] | None = None,
    name: str | None = None,
    channels_meta: Sequence[str | Channel] | None = None,
    ngff_version: NgffVersions = DefaultNgffVersion,
    # Zarr Array parameters
    chunks: ChunksLike = "auto",
    shards: ShardsLike | None = None,
    dtype: str = "uint16",
    dimension_separator: Literal[".", "/"] = "/",
    compressors: CompressorLike = "auto",
    extra_array_kwargs: Mapping[str, Any] | None = None,
    # internal axes configuration for advanced use cases
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = False,
    # Whether to overwrite existing image
    overwrite: bool = False,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr image with the given shape and metadata."""
    base_shape = shapes[0]
    if meta_type is NgioImageMeta:
        default_axes_order = canonical_axes_order()
    else:
        default_axes_order = canonical_label_axes_order()

    axes_handler = _build_axes_handler(
        shape=base_shape,
        axes_names=axes_names,
        default_channel_order=default_axes_order,
        space_units=space_unit,
        time_units=time_unit,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )
    base_scale = _compute_base_scale(
        pixelsize=pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        axes_handler=axes_handler,
    )
    if levels is None:
        levels_paths = tuple(str(i) for i in range(len(shapes)))
    else:
        levels_paths = tuple(levels)

    pyramid_builder = ImagePyramidBuilder.from_shapes(
        shapes=shapes,
        base_scale=base_scale,
        levels_paths=levels_paths,
        axes=axes_handler.axes_names,
        chunks=chunks,
        data_type=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        shards=shards,
        zarr_format=2 if ngff_version == "0.4" else 3,
        other_array_kwargs=extra_array_kwargs,
    )
    meta = meta_type.default_init(
        levels=[p.path for p in pyramid_builder.levels],
        axes_handler=axes_handler,
        scales=[p.scale for p in pyramid_builder.levels],
        translations=[None for _ in pyramid_builder.levels],
        name=name,
        version=ngff_version,
    )
    meta = _add_channels_meta(meta=meta, channels_meta=channels_meta)
    # Keep this creation at the end to avoid partial creations on errors
    return _create_image_like_group(
        store=store,
        pyramid_builder=pyramid_builder,
        meta=meta,
        overwrite=overwrite,
    )
