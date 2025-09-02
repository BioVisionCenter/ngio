"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar
from warnings import warn

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict

from ngio.common._array_io_pipes import (
    build_dask_getter,
    build_dask_setter,
    build_masked_dask_getter,
    build_masked_dask_setter,
    build_masked_numpy_getter,
    build_masked_numpy_setter,
    build_numpy_getter,
    build_numpy_setter,
)
from ngio.common._array_io_utils import SlicingInputType, TransformProtocol
from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import DefaultSpaceUnit, PixelSize, SpaceUnits
from ngio.utils import NgioValueError


def _to_raster(value: float, pixel_size: float, max_shape: int | None) -> int:
    """Convert to raster coordinates."""
    round_value = int(np.round(value / pixel_size))
    # Ensure the value is within the image shape boundaries
    if max_shape is not None:
        return max(0, min(round_value, max_shape))
    return round_value


def _to_world(value: int, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


T = TypeVar("T", int, float)


class GenericRoi(BaseModel, Generic[T]):
    """A generic Region of Interest (ROI) model."""

    name: str
    x: T
    y: T
    z: T | None = None
    t: T | None = None
    x_length: T
    y_length: T
    z_length: T | None = None
    t_length: T | None = None
    label: int | None = None
    unit: SpaceUnits | str | None = None

    model_config = ConfigDict(extra="allow")

    def intersection(self, other: "GenericRoi[T]") -> "GenericRoi[T] | None":
        """Calculate the intersection of this ROI with another ROI."""
        return roi_intersection(self, other)


def _1d_intersection(
    a: T | None, a_length: T | None, b: T | None, b_length: T | None
) -> tuple[T | None, T | None]:
    """Calculate the intersection of two 1D intervals."""
    if a is None:
        if b is not None and b_length is not None:
            return b, b_length
        return None, None
    if b is None:
        if a is not None and a_length is not None:
            return a, a_length
        return None, None

    assert (
        a is not None
        and a_length is not None
        and b is not None
        and b_length is not None
    )
    start = max(a, b)
    end = min(a + a_length, b + b_length)
    length = end - start

    if length <= 0:
        return None, None

    return start, length


def roi_intersection(
    ref_roi: GenericRoi[T], other_roi: GenericRoi[T]
) -> GenericRoi[T] | None:
    """Calculate the intersection of two ROIs."""
    if (
        ref_roi.unit is not None
        and other_roi.unit is not None
        and ref_roi.unit != other_roi.unit
    ):
        raise NgioValueError(
            "Cannot calculate intersection of ROIs with different units."
        )

    x, x_length = _1d_intersection(
        ref_roi.x, ref_roi.x_length, other_roi.x, other_roi.x_length
    )
    assert x is not None and x_length is not None

    y, y_length = _1d_intersection(
        ref_roi.y, ref_roi.y_length, other_roi.y, other_roi.y_length
    )
    assert y is not None and y_length is not None

    z, z_length = _1d_intersection(
        ref_roi.z, ref_roi.z_length, other_roi.z, other_roi.z_length
    )
    t, t_length = _1d_intersection(
        ref_roi.t, ref_roi.t_length, other_roi.t, other_roi.t_length
    )

    if (
        x_length <= 0
        or y_length <= 0
        or (z_length is not None and z_length <= 0)
        or (t_length is not None and t_length <= 0)
    ):
        # No intersection
        return None

    # Find label
    if ref_roi.label is not None and other_roi.label is not None:
        if ref_roi.label != other_roi.label:
            raise NgioValueError(
                "Cannot calculate intersection of ROIs with different labels."
            )
    label = ref_roi.label or other_roi.label

    cls_ref = ref_roi.__class__
    return cls_ref(
        name=f"[{ref_roi.name}_x_{other_roi.name}]",
        x=x,
        y=y,
        z=z,
        t=t,
        x_length=x_length,
        y_length=y_length,
        z_length=z_length,
        t_length=t_length,
        unit=ref_roi.unit,
        label=label,
    )


class Roi(GenericRoi[float]):
    x: float = 0.0
    y: float = 0.0
    unit: SpaceUnits | str | None = DefaultSpaceUnit

    def to_roi_pixels(
        self, pixel_size: PixelSize, dimensions: Dimensions | None = None
    ) -> "RoiPixels":
        """Convert to raster coordinates."""
        if dimensions is None:
            # No check for dimensions
            dim_x, dim_y, dim_z, dim_t = None, None, None, None
        else:
            dim_x, dim_y, dim_z, dim_t = (
                dimensions.get("x"),
                dimensions.get("y"),
                dimensions.get("z"),
                dimensions.get("t"),
            )

        x = _to_raster(self.x, pixel_size.x, dim_x)
        x_length = _to_raster(self.x_length, pixel_size.x, dim_x)
        y = _to_raster(self.y, pixel_size.y, dim_y)
        y_length = _to_raster(self.y_length, pixel_size.y, dim_y)

        if self.z is None:
            z = None
        else:
            z = _to_raster(self.z, pixel_size.z, dim_z)

        if self.z_length is None:
            z_length = None
        else:
            z_length = _to_raster(self.z_length, pixel_size.z, dim_z)

        if self.t is None:
            t = None
        else:
            t = _to_raster(self.t, pixel_size.t, dim_t)

        if self.t_length is None:
            t_length = None
        else:
            t_length = _to_raster(self.t_length, pixel_size.t, dim_t)

        extra_dict = self.model_extra if self.model_extra else {}
        return RoiPixels(
            name=self.name,
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            label=self.label,
            unit=self.unit,
            **extra_dict,
        )

    def to_pixel_roi(
        self, pixel_size: PixelSize, dimensions: Dimensions | None = None
    ) -> "RoiPixels":
        """Convert to raster coordinates."""
        warn(
            "to_pixel_roi is deprecated and will be removed in a future release. "
            "Use to_roi_pixels instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.to_roi_pixels(pixel_size=pixel_size, dimensions=dimensions)

    def zoom(self, zoom_factor: float = 1) -> "Roi":
        """Zoom the ROI by a factor.

        Args:
            zoom_factor: The zoom factor. If the zoom factor
                is less than 1 the ROI will be zoomed in.
                If the zoom factor is greater than 1 the ROI will be zoomed out.
                If the zoom factor is 1 the ROI will not be changed.
        """
        return zoom_roi(self, zoom_factor)


class RoiPixels(GenericRoi[int]):
    """Region of interest (ROI) in pixel coordinates."""

    x: int = 0
    y: int = 0
    unit: SpaceUnits | str | None = None

    def to_roi(self, pixel_size: PixelSize) -> "Roi":
        """Convert to raster coordinates."""
        x = _to_world(self.x, pixel_size.x)
        x_length = _to_world(self.x_length, pixel_size.x)
        y = _to_world(self.y, pixel_size.y)
        y_length = _to_world(self.y_length, pixel_size.y)

        if self.z is None:
            z = None
        else:
            z = _to_world(self.z, pixel_size.z)

        if self.z_length is None:
            z_length = None
        else:
            z_length = _to_world(self.z_length, pixel_size.z)

        if self.t is None:
            t = None
        else:
            t = _to_world(self.t, pixel_size.t)

        if self.t_length is None:
            t_length = None
        else:
            t_length = _to_world(self.t_length, pixel_size.t)

        extra_dict = self.model_extra if self.model_extra else {}
        return Roi(
            name=self.name,
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            label=self.label,
            unit=self.unit,
            **extra_dict,
        )

    def to_slicing_dict(self) -> dict[str, SlicingInputType]:
        x_slice = slice(self.x, self.x + self.x_length)
        y_slice = slice(self.y, self.y + self.y_length)
        if self.z is not None and self.z_length is not None:
            z_slice = slice(self.z, self.z + self.z_length)
        else:
            z_slice = slice(None)
        if self.t is not None and self.t_length is not None:
            t_slice = slice(self.t, self.t + self.t_length)
        else:
            t_slice = slice(None)
        return {
            "x": x_slice,
            "y": y_slice,
            "z": z_slice,
            "t": t_slice,
        }


def zoom_roi(roi: Roi, zoom_factor: float = 1) -> Roi:
    """Zoom the ROI by a factor.

    Args:
        roi: The ROI to zoom.
        zoom_factor: The zoom factor. If the zoom factor
            is less than 1 the ROI will be zoomed in.
            If the zoom factor is greater than 1 the ROI will be zoomed out.
            If the zoom factor is 1 the ROI will not be changed.
    """
    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    # the zoom factor needs to be rescaled
    # from the range [-1, inf) to [0, inf)
    zoom_factor -= 1
    diff_x = roi.x_length * zoom_factor
    diff_y = roi.y_length * zoom_factor

    new_x = max(roi.x - diff_x / 2, 0)
    new_y = max(roi.y - diff_y / 2, 0)

    new_roi = Roi(
        name=roi.name,
        x=new_x,
        y=new_y,
        z=roi.z,
        t=roi.t,
        x_length=roi.x_length + diff_x,
        y_length=roi.y_length + diff_y,
        z_length=roi.z_length,
        t_length=roi.t_length,
        label=roi.label,
        unit=roi.unit,
    )
    return new_roi


def roi_to_slicing_dict(
    roi: Roi | RoiPixels,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> dict[str, SlicingInputType]:
    """Convert a ROI to a slicing dictionary."""
    if isinstance(roi, Roi):
        if pixel_size is None:
            raise NgioValueError(
                "pixel_size must be provided when converting a Roi to slice_kwargs."
            )
        roi = roi.to_roi_pixels(pixel_size=pixel_size, dimensions=dimensions)

    roi_slicing_dict = roi.to_slicing_dict()
    if slicing_dict is None:
        return roi_slicing_dict

    # Additional slice kwargs can be provided
    # and will override the ones from the ROI
    roi_slicing_dict.update(slicing_dict)
    return roi_slicing_dict


def build_roi_numpy_getter(
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    roi: Roi | RoiPixels,
    pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]:
    """Prepare slice kwargs for setting an array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    return build_numpy_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=input_slice_kwargs,
        remove_channel_selection=remove_channel_selection,
    )


def build_roi_numpy_setter(
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    roi: Roi | RoiPixels,
    pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> Callable[[np.ndarray], None]:
    """Prepare slice kwargs for setting an array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    return build_numpy_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=input_slice_kwargs,
    )


def build_roi_dask_getter(
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    roi: Roi | RoiPixels,
    pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], da.Array]:
    """Prepare slice kwargs for getting an array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    return build_dask_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=input_slice_kwargs,
        remove_channel_selection=remove_channel_selection,
    )


def build_roi_dask_setter(
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    roi: Roi | RoiPixels,
    pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> Callable[[da.Array], None]:
    """Prepare slice kwargs for setting an array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    return build_dask_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=input_slice_kwargs,
    )


################################################################
#
# Masked ROIs array pipes
#
################################################################


def build_roi_masked_numpy_getter(
    *,
    roi: Roi | RoiPixels,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    fill_value: int | float = 0,
    allow_scaling: bool = True,
) -> Callable[[], np.ndarray]:
    """Prepare slice kwargs for getting a masked array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    label_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=label_dimensions,
        pixel_size=label_pixel_size,
        slicing_dict=label_slicing_dict,
    )
    return build_masked_numpy_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        label_zarr_array=label_zarr_array,
        label_dimensions=label_dimensions,
        label_id=roi.label,
        axes_order=axes_order,
        transforms=transforms,
        label_transforms=label_transforms,
        slicing_dict=input_slice_kwargs,
        label_slicing_dict=label_slice_kwargs,
        fill_value=fill_value,
        allow_scaling=allow_scaling,
    )


def build_roi_masked_numpy_setter(
    *,
    roi: Roi | RoiPixels,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    allow_scaling: bool = True,
) -> Callable[[np.ndarray], None]:
    """Prepare slice kwargs for setting a masked array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    label_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=label_dimensions,
        pixel_size=label_pixel_size,
        slicing_dict=label_slicing_dict,
    )
    return build_masked_numpy_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        label_zarr_array=label_zarr_array,
        label_dimensions=label_dimensions,
        label_id=roi.label,
        axes_order=axes_order,
        transforms=transforms,
        label_transforms=label_transforms,
        slicing_dict=input_slice_kwargs,
        label_slicing_dict=label_slice_kwargs,
        allow_scaling=allow_scaling,
    )


def build_roi_masked_dask_getter(
    *,
    roi: Roi | RoiPixels,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    allow_scaling: bool = True,
) -> Callable[[], da.Array]:
    """Prepare slice kwargs for getting a masked array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    label_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=label_dimensions,
        pixel_size=label_pixel_size,
        slicing_dict=label_slicing_dict,
    )
    return build_masked_dask_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        label_zarr_array=label_zarr_array,
        label_dimensions=label_dimensions,
        label_id=roi.label,
        axes_order=axes_order,
        transforms=transforms,
        label_transforms=label_transforms,
        slicing_dict=input_slice_kwargs,
        label_slicing_dict=label_slice_kwargs,
        allow_scaling=allow_scaling,
    )


def build_roi_masked_dask_setter(
    *,
    roi: Roi | RoiPixels,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_pixel_size: PixelSize | None = None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    allow_scaling: bool = True,
) -> Callable[[da.Array], None]:
    """Prepare slice kwargs for setting a masked array."""
    input_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=dimensions,
        pixel_size=pixel_size,
        slicing_dict=slicing_dict,
    )
    label_slice_kwargs = roi_to_slicing_dict(
        roi=roi,
        dimensions=label_dimensions,
        pixel_size=label_pixel_size,
        slicing_dict=label_slicing_dict,
    )
    return build_masked_dask_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        label_zarr_array=label_zarr_array,
        label_dimensions=label_dimensions,
        label_id=roi.label,
        axes_order=axes_order,
        transforms=transforms,
        label_transforms=label_transforms,
        slicing_dict=input_slice_kwargs,
        label_slicing_dict=label_slice_kwargs,
        allow_scaling=allow_scaling,
    )
