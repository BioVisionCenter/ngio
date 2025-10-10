"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from typing import Self
from warnings import warn

from pydantic import BaseModel, ConfigDict

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import DefaultSpaceUnit, PixelSize, SpaceUnits
from ngio.utils import NgioValueError


class RoiSlice(BaseModel):
    """A slice of a ROI along a single dimension."""

    start: float
    length: float

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_slice(cls, slice_: slice) -> "RoiSlice":
        """Create a RoiSlice from a slice object."""
        if slice_.start is None or slice_.stop is None:
            raise NgioValueError("Cannot create RoiSlice from open-ended slice.")
        start = float(slice_.start)
        length = float(slice_.stop - slice_.start)
        return cls(start=start, length=length)

    @classmethod
    def from_tuple(cls, tup: tuple[float, float]) -> "RoiSlice":
        """Create a RoiSlice from a tuple of (start, length)."""
        if len(tup) != 2:
            raise NgioValueError("Tuple must be of length 2 (start, length).")
        start, length = tup
        return cls(start=start, length=length)

    @property
    def end(self) -> float:
        """Get the end of the slice."""
        return self.start + self.length

    def to_slice(self) -> slice:
        """Convert to a slice object."""
        return slice(self.start, self.end)

    def to_pixel(self, pixel_size: float) -> "RoiSlice":
        """Convert to pixel coordinates."""
        raster_start = self.start / pixel_size
        raster_length = self.length / pixel_size
        return RoiSlice(start=raster_start, length=raster_length)

    def to_world(self, pixel_size: float) -> "RoiSlice":
        """Convert to world coordinates."""
        world_start = self.start * pixel_size
        world_length = self.length * pixel_size
        return RoiSlice(start=world_start, length=world_length)

    def intersects(self, other: "RoiSlice") -> "RoiSlice | None":
        """Check if this slice intersects with another slice."""
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        length = end - start
        if length <= 0:
            # No intersection
            return None
        return RoiSlice(start=start, length=length)

    def union(self, other: "RoiSlice") -> "RoiSlice":
        """Calculate the union of this slice with another slice."""
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        length = end - start
        return RoiSlice(start=start, length=length)

    def __repr__(self) -> str:
        return f"{self.start}->{self.end}"


class GenericRoi(BaseModel):
    """A generic Region of Interest (ROI) model."""

    name: str | None = None
    x: RoiSlice
    y: RoiSlice
    z: RoiSlice | None = None
    t: RoiSlice | None = None
    label: int | None = None
    unit: SpaceUnits | str | None = None

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_values(
        cls,
        x: tuple[float, float],
        y: tuple[float, float],
        z: tuple[float, float] | None = None,
        t: tuple[float, float] | None = None,
        name: str | None = None,
        label: int | None = None,
        unit: SpaceUnits | str | None = None,
        **extras,
    ) -> Self:
        """Create a GenericRoi from individual values."""
        return cls(
            name=name,
            x=RoiSlice.from_tuple(x),
            y=RoiSlice.from_tuple(y),
            z=RoiSlice.from_tuple(z) if z is not None else None,
            t=RoiSlice.from_tuple(t) if t is not None else None,
            label=label,
            unit=unit,
            **extras,
        )

    def intersection(self, other: "GenericRoi") -> "GenericRoi | None":
        """Calculate the intersection of this ROI with another ROI."""
        return roi_intersection(self, other)

    def union(self, other: "GenericRoi") -> "GenericRoi":
        """Calculate the union of this ROI with another ROI."""
        return roi_union(self, other)

    def _nice_str(self) -> str:
        cls_name = self.__class__.__name__
        name_str = f"name={self.name}, "

        t_str = f"t={self.t}, "
        z_str = f"z={self.z}, "
        y_str = f"y={self.y}, "
        x_str = f"x={self.x}, "

        roi_repr = f"{cls_name}({name_str}{t_str}{z_str}{y_str}{x_str}"
        if self.label is not None:
            roi_repr = f"{roi_repr}label={self.label})"
        else:
            roi_repr = f"{roi_repr})"

        return roi_repr

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


def _name_safe_union(name1: str | None, name2: str | None) -> str | None:
    """Create a name for the union of two ROIs."""
    if name1 is not None and name2 is not None:
        if name1 != name2:
            return f"{name1}:{name2}"
        else:
            return name1
    return name1 or name2


def _label_safe_union(label1: int | None, label2: int | None) -> int | None:
    if label1 is not None and label2 is not None:
        if label1 != label2:
            raise NgioValueError("Cannot create union of ROIs with different labels.")
        else:
            return label1
    return label1 or label2


def _1d_intersection(a: RoiSlice | None, b: RoiSlice | None) -> RoiSlice | None:
    """Calculate the intersection of two 1D intervals."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a.intersects(b)


def _1d_union(a: RoiSlice | None, b: RoiSlice | None) -> RoiSlice | None:
    """Calculate the union of two 1D intervals."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a.union(b)


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

    x = ref_roi.x.intersects(other_roi.x)
    if x is None:
        # No intersection
        return None
    y = ref_roi.y.intersects(other_roi.y)
    if y is None:
        # No intersection
        return None

    z = _1d_intersection(ref_roi.z, other_roi.z)
    t = _1d_intersection(ref_roi.t, other_roi.t)

    # Find label
    label = _label_safe_union(ref_roi.label, other_roi.label)
    name = _name_safe_union(ref_roi.name, other_roi.name)

    cls_ref = ref_roi.__class__
    return cls_ref(
        name=name,
        x=x,
        y=y,
        z=z,
        t=t,
        unit=ref_roi.unit,
        label=label,
    )


def roi_union(ref_roi: GenericRoi, other_roi: GenericRoi) -> GenericRoi:
    """Calculate the intersection of two ROIs."""
    if (
        ref_roi.unit is not None
        and other_roi.unit is not None
        and ref_roi.unit != other_roi.unit
    ):
        raise NgioValueError(
            "Cannot calculate intersection of ROIs with different units."
        )
    x = ref_roi.x.union(other_roi.x)
    y = ref_roi.y.union(other_roi.y)
    z = _1d_union(ref_roi.z, other_roi.z)
    t = _1d_union(ref_roi.t, other_roi.t)

    # Find label
    label = _label_safe_union(ref_roi.label, other_roi.label)
    name = _name_safe_union(ref_roi.name, other_roi.name)

    cls_ref = ref_roi.__class__
    return cls_ref(
        name=name,
        x=x,
        y=y,
        z=z,
        t=t,
        unit=ref_roi.unit,
        label=label,
    )


class Roi(GenericRoi):
    unit: SpaceUnits | str | None = DefaultSpaceUnit

    def to_roi_pixels(self, pixel_size: PixelSize) -> "RoiPixels":
        """Convert to raster coordinates."""
        x = self.x.to_pixel(pixel_size.x)
        y = self.y.to_pixel(pixel_size.y)

        z = self.z.to_pixel(pixel_size.z) if self.z else None
        t = self.t.to_pixel(pixel_size.t) if self.t else None

        extra_dict = self.model_extra if self.model_extra else {}

        return RoiPixels(
            name=self.name,
            x=x,
            y=y,
            z=z,
            t=t,
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

    unit: SpaceUnits | str | None = None

    def to_roi(self, pixel_size: PixelSize) -> "Roi":
        """Convert to raster coordinates."""
        x = self.x.to_world(pixel_size.x)
        y = self.y.to_world(pixel_size.y)
        z = self.z.to_world(pixel_size.z) if self.z else None
        t = self.t.to_world(pixel_size.t) if self.t else None
        extra_dict = self.model_extra if self.model_extra else {}
        return Roi(
            name=self.name,
            x=x,
            y=y,
            z=z,
            t=t,
            label=self.label,
            unit=self.unit,
            **extra_dict,
        )

    def to_slicing_dict(self, pixel_size: PixelSize) -> dict[str, slice]:
        """Convert to a slicing dictionary."""
        x_slice = self.x.to_slice()
        y_slice = self.y.to_slice()
        z_slice = self.z.to_slice() if self.z else slice(None)
        t_slice = self.t.to_slice() if self.t else slice(None)
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
    x_start, x_length = roi.x.start, roi.x.length
    y_start, y_length = roi.y.start, roi.y.length
    diff_x = x_length * zoom_factor
    diff_y = y_length * zoom_factor

    new_x_start = max(x_start - diff_x / 2, 0)
    new_y_start = max(y_start - diff_y / 2, 0)

    new_x = RoiSlice(start=new_x_start, length=x_length + diff_x)
    new_y = RoiSlice(start=new_y_start, length=y_length + diff_y)

    new_roi = Roi(
        name=roi.name,
        x=new_x,
        y=new_y,
        z=roi.z,
        t=roi.t,
        label=roi.label,
        unit=roi.unit,
    )
    return new_roi
