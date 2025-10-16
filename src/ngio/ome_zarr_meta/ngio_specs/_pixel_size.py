"""Fractal internal module for dataset metadata handling."""

import math
from functools import total_ordering
from typing import overload

import numpy as np
from pydantic import BaseModel

from ngio.ome_zarr_meta.ngio_specs import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
)

################################################################################################
#
# PixelSize model
# The PixelSize model is used to store the pixel size in 3D space.
# The model does not store scaling factors and units for other axes.
#
#################################################################################################


@total_ordering
class PixelSize(BaseModel):
    """PixelSize class to store the pixel size in 3D space."""

    x: float
    y: float
    z: float
    t: float = 1
    space_unit: SpaceUnits | str | None = DefaultSpaceUnit
    time_unit: TimeUnits | str | None = DefaultTimeUnit

    def __repr__(self) -> str:
        """Return a string representation of the pixel size."""
        return f"PixelSize(x={self.x}, y={self.y}, z={self.z}, t={self.t})"

    def __eq__(self, other) -> bool:
        """Check if two pixel sizes are equal."""
        if not isinstance(other, PixelSize):
            raise TypeError("Can only compare PixelSize with PixelSize.")

        if (
            self.time_unit is not None
            and other.time_unit is None
            and self.time_unit != other.time_unit
        ):
            return False

        if self.space_unit != other.space_unit:
            return False
        return math.isclose(self.distance(other), 0)

    def __lt__(self, other: "PixelSize") -> bool:
        """Check if one pixel size is less than the other."""
        if not isinstance(other, PixelSize):
            raise TypeError("Can only compare PixelSize with PixelSize.")
        ref = PixelSize(
            x=0,
            y=0,
            z=0,
            t=0,
            space_unit=self.space_unit,
            time_unit=self.time_unit,  # type: ignore
        )
        return self.distance(ref) < other.distance(ref)

    def as_dict(self) -> dict[str, float]:
        """Return the pixel size as a dictionary."""
        return {"t": self.t, "z": self.z, "y": self.y, "x": self.x}

    @overload
    def get(self, axis: str, default: float) -> float: ...

    @overload
    def get(self, axis: str, default: None = None) -> float | None: ...

    def get(self, axis: str, default: float | None = None) -> float | None:
        """Get the pixel size for a given axis (in canonical name)."""
        px_size = self.as_dict().get(axis, default)
        if px_size is None:
            raise ValueError(
                f"Invalid axis name: {axis}, must be one of 'x', 'y', 'z', 't'."
            )
        return px_size

    @property
    def tzyx(self) -> tuple[float, float, float, float]:
        """Return the voxel size in t, z, y, x order."""
        return self.t, self.z, self.y, self.x

    @property
    def zyx(self) -> tuple[float, float, float]:
        """Return the voxel size in z, y, x order."""
        return self.z, self.y, self.x

    @property
    def yx(self) -> tuple[float, float]:
        """Return the xy plane pixel size in y, x order."""
        return self.y, self.x

    @property
    def voxel_volume(self) -> float:
        """Return the volume of a voxel."""
        return self.y * self.x * self.z

    @property
    def xy_plane_area(self) -> float:
        """Return the area of the xy plane."""
        return self.y * self.x

    @property
    def time_spacing(self) -> float | None:
        """Return the time spacing."""
        return self.t

    def distance(self, other: "PixelSize") -> float:
        """Return the distance between two pixel sizes."""
        return float(np.linalg.norm(np.array(self.tzyx) - np.array(other.tzyx)))
