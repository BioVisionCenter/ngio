"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, Field

from ngio.common._array_io_pipe import (
    SlicingInputType,
    build_dask_getter,
    build_dask_setter,
    build_masked_dask_getter,
    build_masked_dask_setter,
    build_masked_numpy_getter,
    build_masked_numpy_setter,
    build_numpy_getter,
    build_numpy_setter,
)
from ngio.common._dimensions import Dimensions
from ngio.common._io_transforms import TransformProtocol
from ngio.ome_zarr_meta.ngio_specs import DefaultSpaceUnit, PixelSize, SpaceUnits
from ngio.utils import NgioValueError


def _to_raster(value: float, pixel_size: float, max_shape: int) -> int:
    """Convert to raster coordinates."""
    round_value = int(np.round(value / pixel_size))
    # Ensure the value is within the image shape boundaries
    return max(0, min(round_value, max_shape))


def _to_world(value: int, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


class Roi(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x_length: float
    y_length: float
    z_length: float = 1.0
    t_length: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    unit: SpaceUnits | str | None = Field(DefaultSpaceUnit, repr=False)
    label: int | None = None

    model_config = ConfigDict(extra="allow")

    def to_pixel_roi(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RoiPixels":
        """Convert to raster coordinates."""
        dim_x = dimensions.get("x")
        dim_y = dimensions.get("y")
        # Will default to 1 if z does not exist
        dim_z = dimensions.get("z", default=1)
        dim_t = dimensions.get("t", default=1)
        extra_dict = self.model_extra if self.model_extra else {}

        return RoiPixels(
            name=self.name,
            x=_to_raster(self.x, pixel_size.x, dim_x),
            y=_to_raster(self.y, pixel_size.y, dim_y),
            z=_to_raster(self.z, pixel_size.z, dim_z),
            t=_to_raster(self.t, pixel_size.t, dim_t),
            x_length=_to_raster(self.x_length, pixel_size.x, dim_x),
            y_length=_to_raster(self.y_length, pixel_size.y, dim_y),
            z_length=_to_raster(self.z_length, pixel_size.z, dim_z),
            t_length=_to_raster(self.t_length, pixel_size.t, dim_t),
            label=self.label,
            **extra_dict,
        )

    def zoom(self, zoom_factor: float = 1) -> "Roi":
        """Zoom the ROI by a factor.

        Args:
            zoom_factor: The zoom factor. If the zoom factor
                is less than 1 the ROI will be zoomed in.
                If the zoom factor is greater than 1 the ROI will be zoomed out.
                If the zoom factor is 1 the ROI will not be changed.
        """
        return zoom_roi(self, zoom_factor)

    def intersection(self, other: "Roi") -> "Roi | None":
        """Calculate the intersection of two ROIs."""
        if self.unit != other.unit:
            raise NgioValueError(
                "Cannot calculate intersection of ROIs with different units."
            )

        x = max(self.x, other.x)
        y = max(self.y, other.y)
        z = max(self.z, other.z)
        t = max(self.t, other.t)

        x_length = min(self.x + self.x_length, other.x + other.x_length) - x
        y_length = min(self.y + self.y_length, other.y + other.y_length) - y
        z_length = min(self.z + self.z_length, other.z + other.z_length) - z
        t_length = min(self.t + self.t_length, other.t + other.t_length) - t

        if x_length <= 0 or y_length <= 0 or z_length <= 0 or t_length <= 0:
            # No intersection
            return None

        # Find label
        if self.label is not None and other.label is not None:
            if self.label != other.label:
                raise NgioValueError(
                    "Cannot calculate intersection of ROIs with different labels."
                )
        label = self.label or other.label

        return Roi(
            name=f"[{self.name}_x_{other.name}]",
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            unit=self.unit,
            label=label,
        )


class RoiPixels(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x_length: int
    y_length: int
    z_length: int = 1
    t_length: int = 1
    x: int = 0
    y: int = 0
    z: int = 0
    t: int = 0
    label: int | None = None

    model_config = ConfigDict(extra="allow")

    def to_roi(self, pixel_size: PixelSize) -> Roi:
        """Convert to world coordinates."""
        extra_dict = self.model_extra if self.model_extra else {}
        return Roi(
            name=self.name,
            x=_to_world(self.x, pixel_size.x),
            y=_to_world(self.y, pixel_size.y),
            z=_to_world(self.z, pixel_size.z),
            t=_to_world(self.t, pixel_size.t),
            x_length=_to_world(self.x_length, pixel_size.x),
            y_length=_to_world(self.y_length, pixel_size.y),
            z_length=_to_world(self.z_length, pixel_size.z),
            t_length=_to_world(self.t_length, pixel_size.t),
            unit=pixel_size.space_unit,
            label=self.label,
            **extra_dict,
        )

    def to_slicing_dict(self) -> dict[str, SlicingInputType]:
        """Return the slices for the ROI."""
        return {
            "x": slice(self.x, self.x + self.x_length),
            "y": slice(self.y, self.y + self.y_length),
            "z": slice(self.z, self.z + self.z_length),
            "t": slice(self.t, self.t + self.t_length),
        }

    def intersection(self, other: "RoiPixels") -> "RoiPixels | None":
        """Calculate the intersection of two ROIs."""
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        z = max(self.z, other.z)
        t = max(self.t, other.t)

        x_length = min(self.x + self.x_length, other.x + other.x_length) - x
        y_length = min(self.y + self.y_length, other.y + other.y_length) - y
        z_length = min(self.z + self.z_length, other.z + other.z_length) - z
        t_length = min(self.t + self.t_length, other.t + other.t_length) - t

        if x_length <= 0 or y_length <= 0 or z_length <= 0 or t_length <= 0:
            # No intersection
            return None

        # Find label
        if self.label is not None and other.label is not None:
            if self.label != other.label:
                raise NgioValueError(
                    "Cannot calculate intersection of ROIs with different labels."
                )
        label = self.label or other.label

        return RoiPixels(
            name=f"[{self.name}_x_{other.name}]",
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            label=label,
        )


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
        roi = roi.to_pixel_roi(pixel_size=pixel_size, dimensions=dimensions)

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
    print("label", roi.label)
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
    )
