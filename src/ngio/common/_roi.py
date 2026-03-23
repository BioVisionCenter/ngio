"""Region of interest (ROI) metadata.

These are the interfaces between the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Callable, Mapping
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ngio.ome_zarr_meta import PixelSize
from ngio.utils import NgioValueError

SliceValueType: TypeAlias = float | tuple[float | None, float | None] | slice


def world_to_pixel(value: float, pixel_size: float, eps: float = 1e-2) -> float:
    raster_value = value / pixel_size

    # Round to nearest integer if within eps, this allows to keep pixel-aligned ROIs
    # when machine precision issues would otherwise make them slightly off.
    # Since errors in precision are both on the value and the pixel size,
    # the tolerance must be large enough to accommodate compounded errors,
    # we default to 1e-2 which allows for ~1% error.
    _rounded = round(raster_value)
    if abs(_rounded - raster_value) < eps:
        return _rounded
    return raster_value


def pixel_to_world(value: float, pixel_size: float) -> float:
    return value * pixel_size


def _join_roi_names(name1: str | None, name2: str | None) -> str | None:
    if name1 is not None and name2 is not None:
        if name1 == name2:
            return name1
        return f"{name1}:{name2}"
    return name1 or name2


def _join_roi_labels(label1: int | None, label2: int | None) -> int | None:
    if label1 is not None and label2 is not None:
        if label1 == label2:
            return label1
        raise NgioValueError("Cannot join ROIs with different labels")
    return label1 or label2


class RoiSlice(BaseModel):
    """A 1-D slice along a named axis for use within a Region of Interest.

    Represents a contiguous interval [start, start + length) along a
    single named axis. Either bound may be None to indicate an open
    (unbounded) interval.

    Attributes:
        axis_name: Name of the axis this slice applies to (e.g. "x", "y", "z").
        start: Start coordinate of the interval (inclusive). None means
            "from the beginning".
        length: Length of the interval. Must be non-negative. None means
            "to the end".
    """

    axis_name: str
    start: float | None = Field(default=None)
    length: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def _from_slice(
        cls,
        axis_name: str,
        selection: slice,
    ) -> "RoiSlice":
        start = selection.start
        length = (
            None
            if selection.stop is None or selection.start is None
            else selection.stop - selection.start
        )
        return cls(axis_name=axis_name, start=start, length=length)

    @classmethod
    def from_value(
        cls,
        axis_name: str,
        value: "SliceValueType | RoiSlice",
    ) -> "RoiSlice":
        """Create a RoiSlice from a variety of input types.

        Args:
            axis_name: Name of the axis this slice applies to.
            value: The interval to represent. Accepted types:
                - slice: converted directly via start / stop.
                - tuple[float | None, float | None]: interpreted as (start, length).
                - int | float: a single-unit interval starting at value (length=1).
                - RoiSlice: returned as-is.

        Returns:
            A new RoiSlice instance.

        Raises:
            TypeError: If value is not one of the supported types.
        """
        if isinstance(value, slice):
            return cls._from_slice(axis_name=axis_name, selection=value)
        elif isinstance(value, tuple):
            return cls(axis_name=axis_name, start=value[0], length=value[1])
        elif isinstance(value, int | float):
            return cls(axis_name=axis_name, start=value, length=1)
        elif isinstance(value, RoiSlice):
            return value
        else:
            raise TypeError(f"Unsupported type for slice value: {type(value)}")

    def __repr__(self) -> str:
        return f"{self.axis_name}: {self.start}->{self.end}"

    @property
    def end(self) -> float | None:
        """Exclusive end coordinate of the interval.

        Returns:
            start + length, or None if either bound is unset.
        """
        if self.start is None or self.length is None:
            return None
        return self.start + self.length

    def to_slice(self) -> slice:
        """Convert to a standard Python slice object.

        Returns:
            A slice(start, end) representing this interval.
        """
        return slice(self.start, self.end)

    def _is_compatible(self, other: "RoiSlice", msg: str) -> None:
        if self.axis_name != other.axis_name:
            raise NgioValueError(
                f"{msg}: Cannot operate on RoiSlices with different axis names"
            )

    def union(self, other: "RoiSlice") -> "RoiSlice":
        """Return the smallest interval that contains both slices.

        Args:
            other: Another RoiSlice on the same axis.

        Returns:
            A new RoiSlice spanning from the minimum start to the maximum
            end of the two inputs.

        Raises:
            NgioValueError: If the two slices are on different axes.
        """
        self._is_compatible(other, "RoiSlice union failed")
        start = min(self.start or 0, other.start or 0)
        end = max(self.end or float("inf"), other.end or float("inf"))
        length = end - start if end > start else 0
        if length == float("inf"):
            length = None
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def intersection(self, other: "RoiSlice") -> "RoiSlice | None":
        """Return the overlap between this slice and other, or None.

        Args:
            other: Another RoiSlice on the same axis.

        Returns:
            A new RoiSlice representing the overlapping interval, or None
            if the two slices do not overlap.

        Raises:
            NgioValueError: If the two slices are on different axes.
        """
        self._is_compatible(other, "RoiSlice intersection failed")
        start = max(self.start or 0, other.start or 0)
        end = min(self.end or float("inf"), other.end or float("inf"))
        if end <= start:
            # No intersection
            return None
        length = end - start
        if length == float("inf"):
            length = None
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def to_world(self, pixel_size: float) -> "RoiSlice":
        """Convert from pixel coordinates to world (physical) coordinates.

        Args:
            pixel_size: Physical size of one pixel along this axis.

        Returns:
            A new RoiSlice with start and length expressed in world units.
        """
        start = (
            pixel_to_world(self.start, pixel_size) if self.start is not None else None
        )
        length = (
            pixel_to_world(self.length, pixel_size) if self.length is not None else None
        )
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def to_pixel(self, pixel_size: float) -> "RoiSlice":
        """Convert from world (physical) coordinates to pixel coordinates.

        Args:
            pixel_size: Physical size of one pixel along this axis.

        Returns:
            A new RoiSlice with start and length expressed in pixel units.
        """
        start = (
            world_to_pixel(self.start, pixel_size) if self.start is not None else None
        )
        length = (
            world_to_pixel(self.length, pixel_size) if self.length is not None else None
        )
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def zoom(self, zoom_factor: float = 1.0) -> "RoiSlice":
        """Expand or shrink the slice symmetrically around its centre.

        A zoom_factor greater than 1 enlarges the interval; less than 1
        shrinks it. The centre of the interval is preserved, and the start
        is clamped to 0.

        Args:
            zoom_factor: Multiplicative scale applied to the length.
                Must be strictly positive.

        Returns:
            A new RoiSlice with the adjusted interval.

        Raises:
            NgioValueError: If zoom_factor is not greater than 0.
        """
        if zoom_factor <= 0:
            raise NgioValueError("Zoom factor must be greater than 0")
        zoom_factor -= 1.0
        if self.length is None:
            return self

        diff_length = self.length * zoom_factor
        length = self.length + diff_length
        start = max((self.start or 0) - (diff_length / 2), 0)
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)


class Roi(BaseModel):
    """A multi-dimensional Region of Interest (ROI).

    An Roi groups one RoiSlice per axis into a named, labelled region that
    can live in either world (physical) or pixel coordinate space.

    Attributes:
        name: Human-readable name for this ROI, or None.
        slices: List of per-axis slices. Must contain at least two entries
            and each axis name must be unique.
        label: Integer label identifying the ROI (e.g. from a label image).
            Must be non-negative, or None if unlabelled.
        space: Coordinate space of the slice values - either "world" for
            physical units or "pixel" for pixel indices.
    """

    name: str | None
    slices: list[RoiSlice] = Field(min_length=2)
    label: int | None = Field(default=None, ge=0)
    space: Literal["world", "pixel"] = "world"

    model_config = ConfigDict(extra="allow")

    @field_validator("slices")
    @classmethod
    def validate_no_duplicate_axes(cls, v: list[RoiSlice]) -> list[RoiSlice]:
        axis_names = [s.axis_name for s in v]
        if len(axis_names) != len(set(axis_names)):
            raise NgioValueError("Roi slices must have unique axis names")
        return v

    def _nice_repr__(self) -> str:
        slices_repr = ", ".join(repr(s) for s in self.slices)
        if self.label is None:
            label_str = ""
        else:
            label_str = f", label={self.label}"

        if self.name is None:
            name_str = ""
        else:
            name_str = f"name={self.name}, "
        return f"Roi({name_str}{slices_repr}{label_str}, space={self.space})"

    @classmethod
    def from_values(
        cls,
        slices: Mapping[str, SliceValueType | RoiSlice],
        name: str | None,
        label: int | None = None,
        space: Literal["world", "pixel"] = "world",
        **kwargs,
    ) -> Self:
        """Create an Roi from a mapping of axis names to slice values.

        Args:
            slices: Mapping from axis name to a slice value accepted by
                RoiSlice.from_value (slice, tuple, float, or RoiSlice).
            name: Human-readable name for the ROI.
            label: Integer label, or None if unlabelled.
            space: Coordinate space of the provided values ("world" or
                "pixel").
            **kwargs: Additional fields stored on the model.

        Returns:
            A new Roi instance.
        """
        _slices = []
        for axis, _slice in slices.items():
            _slices.append(RoiSlice.from_value(axis_name=axis, value=_slice))
        return cls.model_construct(
            name=name, slices=_slices, label=label, space=space, **kwargs
        )

    def __getitem__(self, key):
        """Allow dict-like access to slices by axis name."""
        _slice = self.get(key)
        if _slice is None:
            raise KeyError(f"Axis '{key}' not found in ROI slices")
        return _slice

    def get(self, axis_name: str, default: RoiSlice | None = None) -> RoiSlice | None:
        """Return the RoiSlice for a given axis, or None if not present.

        Args:
            axis_name: Name of the axis to look up.
            default: Value to return if the axis is not found.

        Returns:
            The matching RoiSlice, or the default value if no slice with
            the given axis name exists.
        """
        for roi_slice in self.slices:
            if roi_slice.axis_name == axis_name:
                return roi_slice
        return default

    def get_name(self) -> str:
        """Return a display name for this ROI.

        Falls back to the string label, then a full repr, when name is None.

        Returns:
            The ROI name, label as string, or a repr string.
        """
        if self.name is not None:
            return self.name
        if self.label is not None:
            return str(self.label)
        return self._nice_repr__()

    @staticmethod
    def _apply_sym_ops(
        self_slices: list[RoiSlice],
        other_slices: list[RoiSlice],
        op: Callable[[RoiSlice, RoiSlice], RoiSlice | None],
    ) -> list[RoiSlice] | None:
        self_axis_dict = {s.axis_name: s for s in self_slices}
        other_axis_dict = {s.axis_name: s for s in other_slices}
        common_axis_names = self_axis_dict.keys() | other_axis_dict.keys()
        new_slices = []
        for axis_name in common_axis_names:
            slice_a = self_axis_dict.get(axis_name)
            slice_b = other_axis_dict.get(axis_name)
            if slice_a is not None and slice_b is not None:
                result = op(slice_a, slice_b)
                if result is None:
                    return None
                new_slices.append(result)
            elif slice_a is not None:
                new_slices.append(slice_a)
            elif slice_b is not None:
                new_slices.append(slice_b)
        return new_slices

    def intersection(self, other: Self) -> Self | None:
        """Return the per-axis intersection of this ROI and other, or None.

        Axes present in both ROIs are intersected; axes present in only one
        are kept as-is. Returns None if any shared axis has no overlap.

        Args:
            other: Another Roi in the same coordinate space.

        Returns:
            A new Roi representing the intersection, or None if the ROIs
            do not overlap.

        Raises:
            NgioValueError: If the two ROIs are in different coordinate
                spaces, or if the labels conflict.
        """
        if self.space != other.space:
            raise NgioValueError(
                "Roi intersection failed: One ROI is in pixel space and the "
                "other in world space"
            )

        out_slices = self._apply_sym_ops(
            self.slices, other.slices, op=lambda a, b: a.intersection(b)
        )
        if out_slices is None:
            return None

        name = _join_roi_names(self.name, other.name)
        label = _join_roi_labels(self.label, other.label)
        return self.model_copy(
            update={"name": name, "slices": out_slices, "label": label}
        )

    def union(self, other: Self) -> Self:
        """Return the per-axis union (bounding box) of this ROI and other.

        Axes present in both ROIs are unioned; axes present in only one are
        kept as-is.

        Args:
            other: Another Roi in the same coordinate space.

        Returns:
            A new Roi whose slices span the combined extent of both inputs.

        Raises:
            NgioValueError: If the two ROIs are in different coordinate
                spaces, or if the labels conflict.
        """
        if self.space != other.space:
            raise NgioValueError(
                "Roi union failed: One ROI is in pixel space and the "
                "other in world space"
            )

        out_slices = self._apply_sym_ops(
            self.slices, other.slices, op=lambda a, b: a.union(b)
        )
        if out_slices is None:
            raise NgioValueError("Roi union failed: could not compute union")

        name = _join_roi_names(self.name, other.name)
        label = _join_roi_labels(self.label, other.label)
        return self.model_copy(
            update={"name": name, "slices": out_slices, "label": label}
        )

    def zoom(
        self, zoom_factor: float = 1.0, axes: tuple[str, ...] = ("x", "y")
    ) -> Self:
        """Expand or shrink the ROI symmetrically along the specified axes.

        Args:
            zoom_factor: Multiplicative scale applied to the length of each
                selected slice. Must be strictly positive.
            axes: Names of the axes to zoom. Defaults to ("x", "y").

        Returns:
            A new Roi with the zoom applied to the selected axes.
        """
        new_slices = []
        for roi_slice in self.slices:
            if roi_slice.axis_name in axes:
                new_slices.append(roi_slice.zoom(zoom_factor=zoom_factor))
            else:
                new_slices.append(roi_slice)
        return self.model_copy(update={"slices": new_slices})

    def to_world(self, pixel_size: PixelSize | None = None) -> Self:
        """Convert the ROI to world (physical) coordinate space.

        If the ROI is already in world space, a copy is returned unchanged.

        Args:
            pixel_size: Per-axis pixel sizes used for the conversion.
                Required when the ROI is currently in pixel space.

        Returns:
            A new Roi with all slices expressed in world units.

        Raises:
            NgioValueError: If the ROI is in pixel space and pixel_size is
                not provided.
        """
        if self.space == "world":
            return self.model_copy()
        if pixel_size is None:
            raise NgioValueError(
                "Pixel sizes must be provided to convert ROI from pixel to world"
            )
        new_slices = []
        for roi_slice in self.slices:
            pixel_size_ = pixel_size.get(roi_slice.axis_name, default=1.0)
            new_slices.append(roi_slice.to_world(pixel_size=pixel_size_))
        return self.model_copy(update={"slices": new_slices, "space": "world"})

    def to_pixel(self, pixel_size: PixelSize | None = None) -> Self:
        """Convert the ROI to pixel coordinate space.

        If the ROI is already in pixel space, a copy is returned unchanged.

        Args:
            pixel_size: Per-axis pixel sizes used for the conversion.
                Required when the ROI is currently in world space.

        Returns:
            A new Roi with all slices expressed in pixel units.

        Raises:
            NgioValueError: If the ROI is in world space and pixel_size is
                not provided.
        """
        if self.space == "pixel":
            return self.model_copy()

        if pixel_size is None:
            raise NgioValueError(
                "Pixel sizes must be provided to convert ROI from world to pixel"
            )

        new_slices = []
        for roi_slice in self.slices:
            pixel_size_ = pixel_size.get(roi_slice.axis_name, default=1.0)
            new_slices.append(roi_slice.to_pixel(pixel_size=pixel_size_))
        return self.model_copy(update={"slices": new_slices, "space": "pixel"})

    def to_slicing_dict(self, pixel_size: PixelSize | None = None) -> dict[str, slice]:
        """Convert the ROI to a dict of axis-name -> slice in pixel space.

        Converts to pixel coordinates first if necessary.

        Args:
            pixel_size: Per-axis pixel sizes, required when the ROI is in
                world space.

        Returns:
            A dict mapping each axis name to a Python slice object.
        """
        roi = self.to_pixel(pixel_size=pixel_size)
        return {roi_slice.axis_name: roi_slice.to_slice() for roi_slice in roi.slices}

    def update_slice(self, name: str, new_slice: SliceValueType | RoiSlice) -> Self:
        """Replace or add the slice for a given axis.

        If an axis with the given name already exists it is replaced;
        otherwise the new slice is appended.

        Args:
            name: Axis name of the slice to update or add.
            new_slice: New slice value, accepted by RoiSlice.from_value.

        Returns:
            A new Roi with the updated slice list.
        """
        new_roi_slice = RoiSlice.from_value(axis_name=name, value=new_slice)
        slices = []
        found = False
        for roi_slice in self.slices:
            if roi_slice.axis_name == name:
                slices.append(new_roi_slice)
                found = True
            else:
                slices.append(roi_slice)
        if not found:
            slices.append(new_roi_slice)
        return self.model_copy(update={"slices": slices})

    def remove_slice(self, name: str) -> Self:
        """Remove the slice for a given axis.

        Args:
            name: Axis name of the slice to remove.

        Returns:
            A new Roi with the named slice removed.

        Raises:
            NgioValueError: If no slice with the given axis name exists.
        """
        slices = [s for s in self.slices if s.axis_name != name]
        if len(slices) == len(self.slices):
            raise NgioValueError(f"Axis '{name}' not found in ROI slices")
        return self.model_copy(update={"slices": slices})
