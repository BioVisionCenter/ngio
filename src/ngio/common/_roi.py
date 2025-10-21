"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from typing import TypeVar
from warnings import warn

from pydantic import BaseModel, ConfigDict

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import DefaultSpaceUnit, PixelSize, SpaceUnits
from ngio.utils import NgioValueError


def _world_to_raster(value: float, pixel_size: float, eps: float = 1e-6) -> float:
    raster_value = value / pixel_size

    # If the value is very close to an integer, round it
    # This ensures that we don't have floating point precision issues
    # When loading ROIs that were originally defined in pixel coordinates
    _rounded = round(raster_value)
    if abs(_rounded - raster_value) < eps:
        return _rounded
    return raster_value


def _to_raster(value: float, length: float, pixel_size: float) -> tuple[float, float]:
    """Convert to raster coordinates."""
    raster_value = _world_to_raster(value, pixel_size)
    raster_length = _world_to_raster(length, pixel_size)
    return raster_value, raster_length


def _to_slice(start: float | None, length: float | None) -> slice:
    if length is not None:
        assert start is not None
        end = start + length
    else:
        end = None
    return slice(start, end)


def _raster_to_world(value: int | float, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


T = TypeVar("T", int, float)


class GenericRoi(BaseModel):
    """A generic Region of Interest (ROI) model."""

    name: str | None = None
    x: float
    y: float
    z: float | None = None
    t: float | None = None
    x_length: float
    y_length: float
    z_length: float | None = None
    t_length: float | None = None
    label: int | None = None
    unit: SpaceUnits | str | None = None

    model_config = ConfigDict(extra="allow")

    def intersection(self, other: "GenericRoi") -> "GenericRoi | None":
        """Calculate the intersection of this ROI with another ROI."""
        return roi_intersection(self, other)

    def _nice_str(self) -> str:
        if self.t is not None:
            t_start = self.t
        else:
            t_start = None
        if self.t_length is not None and t_start is not None:
            t_end = t_start + self.t_length
        else:
            t_end = None

        t_str = f"t={t_start}->{t_end}"

        if self.z is not None:
            z_start = self.z
        else:
            z_start = None
        if self.z_length is not None and z_start is not None:
            z_end = z_start + self.z_length
        else:
            z_end = None
        z_str = f"z={z_start}->{z_end}"

        y_str = f"y={self.y}->{self.y + self.y_length}"
        x_str = f"x={self.x}->{self.x + self.x_length}"

        if self.label is not None:
            label_str = f", label={self.label}"
        else:
            label_str = ""
        cls_name = self.__class__.__name__
        return f"{cls_name}({t_str}, {z_str}, {y_str}, {x_str}{label_str})"

    def get_name(self) -> str:
        """Get the name of the ROI, or a default if not set."""
        if self.name is not None:
            return self.name
        return self._nice_str()

    def __repr__(self) -> str:
        return self._nice_str()

    def __str__(self) -> str:
        return self._nice_str()

    def to_slicing_dict(self, pixel_size: PixelSize) -> dict[str, slice]:
        raise NotImplementedError


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


def roi_intersection(ref_roi: GenericRoi, other_roi: GenericRoi) -> GenericRoi | None:
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
    if x is None and x_length is None:
        # No intersection
        return None
    assert x is not None and x_length is not None

    y, y_length = _1d_intersection(
        ref_roi.y, ref_roi.y_length, other_roi.y, other_roi.y_length
    )
    if y is None and y_length is None:
        # No intersection
        return None
    assert y is not None and y_length is not None

    z, z_length = _1d_intersection(
        ref_roi.z, ref_roi.z_length, other_roi.z, other_roi.z_length
    )
    t, t_length = _1d_intersection(
        ref_roi.t, ref_roi.t_length, other_roi.t, other_roi.t_length
    )

    if (z_length is not None and z_length <= 0) or (
        t_length is not None and t_length <= 0
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

    if ref_roi.name is not None and other_roi.name is not None:
        name = f"{ref_roi.name}:{other_roi.name}"
    else:
        name = ref_roi.name or other_roi.name

    cls_ref = ref_roi.__class__
    return cls_ref(
        name=name,
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


class Roi(GenericRoi):
    x: float = 0.0
    y: float = 0.0
    unit: SpaceUnits | str | None = DefaultSpaceUnit

    def to_roi_pixels(self, pixel_size: PixelSize) -> "RoiPixels":
        """Convert to raster coordinates."""
        x, x_length = _to_raster(self.x, self.x_length, pixel_size.x)
        y, y_length = _to_raster(self.y, self.y_length, pixel_size.y)

        if self.z is None:
            z, z_length = None, None
        else:
            assert self.z_length is not None
            z, z_length = _to_raster(self.z, self.z_length, pixel_size.z)

        if self.t is None:
            t, t_length = None, None
        else:
            assert self.t_length is not None
            t, t_length = _to_raster(self.t, self.t_length, pixel_size.t)
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

        return self.to_roi_pixels(pixel_size=pixel_size)

    def zoom(self, zoom_factor: float = 1) -> "Roi":
        """Zoom the ROI by a factor.

        Args:
            zoom_factor: The zoom factor. If the zoom factor
                is less than 1 the ROI will be zoomed in.
                If the zoom factor is greater than 1 the ROI will be zoomed out.
                If the zoom factor is 1 the ROI will not be changed.
        """
        return zoom_roi(self, zoom_factor)

    def to_slicing_dict(self, pixel_size: PixelSize) -> dict[str, slice]:
        """Convert to a slicing dictionary."""
        roi_pixels = self.to_roi_pixels(pixel_size)
        return roi_pixels.to_slicing_dict(pixel_size)


class RoiPixels(GenericRoi):
    """Region of interest (ROI) in pixel coordinates."""

    x: float = 0
    y: float = 0
    unit: SpaceUnits | str | None = None

    def to_roi(self, pixel_size: PixelSize) -> "Roi":
        """Convert to raster coordinates."""
        x = _raster_to_world(self.x, pixel_size.x)
        x_length = _raster_to_world(self.x_length, pixel_size.x)
        y = _raster_to_world(self.y, pixel_size.y)
        y_length = _raster_to_world(self.y_length, pixel_size.y)

        if self.z is None:
            z = None
        else:
            z = _raster_to_world(self.z, pixel_size.z)

        if self.z_length is None:
            z_length = None
        else:
            z_length = _raster_to_world(self.z_length, pixel_size.z)

        if self.t is None:
            t = None
        else:
            t = _raster_to_world(self.t, pixel_size.t)

        if self.t_length is None:
            t_length = None
        else:
            t_length = _raster_to_world(self.t_length, pixel_size.t)

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

    def to_slicing_dict(self, pixel_size: PixelSize) -> dict[str, slice]:
        """Convert to a slicing dictionary."""
        x_slice = _to_slice(self.x, self.x_length)
        y_slice = _to_slice(self.y, self.y_length)
        z_slice = _to_slice(self.z, self.z_length)
        t_slice = _to_slice(self.t, self.t_length)
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
        raise NgioValueError("Zoom factor must be greater than 0.")

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
