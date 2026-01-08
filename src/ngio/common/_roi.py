"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Callable, Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ngio.ome_zarr_meta import PixelSize
from ngio.utils import NgioValueError


def world_to_pixel(value: float, pixel_size: float, eps: float = 1e-6) -> float:
    raster_value = value / pixel_size

    # If the value is very close to an integer, round it
    # This ensures that we don't have floating point precision issues
    # When loading ROIs that were originally defined in pixel coordinates
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
        value: float | tuple[float | None, float | None] | slice,
    ) -> "RoiSlice":
        if isinstance(value, slice):
            return cls._from_slice(axis_name=axis_name, selection=value)
        elif isinstance(value, tuple):
            return cls(axis_name=axis_name, start=value[0], length=value[1])
        elif isinstance(value, int | float):
            return cls(axis_name=axis_name, start=value, length=1)
        else:
            raise TypeError(f"Unsupported type for slice value: {type(value)}")

    def __repr__(self) -> str:
        return f"{self.axis_name}: {self.start}->{self.end}"

    @property
    def end(self) -> float | None:
        if self.start is None or self.length is None:
            return None
        return self.start + self.length

    def to_slice(self) -> slice:
        return slice(self.start, self.end)

    def _is_compatible(self, other: "RoiSlice", msg: str) -> None:
        if self.axis_name != other.axis_name:
            raise NgioValueError(
                f"{msg}: Cannot operate on RoiSlices with different axis names"
            )

    def union(self, other: "RoiSlice") -> "RoiSlice":
        self._is_compatible(other, "RoiSlice union failed")
        start = min(self.start or 0, other.start or 0)
        end = max(self.end or float("inf"), other.end or float("inf"))
        length = end - start if end > start else 0
        if length == float("inf"):
            length = None
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def intersection(self, other: "RoiSlice") -> "RoiSlice | None":
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
        start = (
            pixel_to_world(self.start, pixel_size) if self.start is not None else None
        )
        length = (
            pixel_to_world(self.length, pixel_size) if self.length is not None else None
        )
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def to_pixel(self, pixel_size: float) -> "RoiSlice":
        start = (
            world_to_pixel(self.start, pixel_size) if self.start is not None else None
        )
        length = (
            world_to_pixel(self.length, pixel_size) if self.length is not None else None
        )
        return RoiSlice(axis_name=self.axis_name, start=start, length=length)

    def zoom(self, zoom_factor: float = 1.0) -> "RoiSlice":
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
        slices: Mapping[str, float | tuple[float | None, float | None] | slice],
        name: str | None,
        label: int | None = None,
        space: Literal["world", "pixel"] = "world",
        **kwargs,
    ) -> Self:
        _slices = []
        for axis, _slice in slices.items():
            _slices.append(RoiSlice.from_value(axis_name=axis, value=_slice))
        return cls.model_construct(
            name=name, slices=_slices, label=label, space=space, **kwargs
        )

    def get(self, axis_name: str) -> RoiSlice | None:
        for roi_slice in self.slices:
            if roi_slice.axis_name == axis_name:
                return roi_slice
        return None

    def get_name(self) -> str:
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
        new_slices = []
        for roi_slice in self.slices:
            if roi_slice.axis_name in axes:
                new_slices.append(roi_slice.zoom(zoom_factor=zoom_factor))
            else:
                new_slices.append(roi_slice)
        return self.model_copy(update={"slices": new_slices})

    def to_world(self, pixel_size: PixelSize | None = None) -> Self:
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
        roi = self.to_pixel(pixel_size=pixel_size)
        return {roi_slice.axis_name: roi_slice.to_slice() for roi_slice in roi.slices}
