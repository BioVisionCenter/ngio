"""Fractal internal module for dataset metadata handling."""

import math
import warnings
from functools import total_ordering

import numpy as np
from pydantic import BaseModel

from ngio.ome_zarr_meta.ngio_specs import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
)
from ngio.utils import NgioUserWarning
from ngio.utils._warnings import stacklevel_of_first_caller

#: Meters per space unit, for cross-unit comparisons.
_METERS_PER_UNIT: dict[str, float] = {
    "yoctometer": 1e-24,
    "zeptometer": 1e-21,
    "attometer": 1e-18,
    "femtometer": 1e-15,
    "picometer": 1e-12,
    "angstrom": 1e-10,
    "nanometer": 1e-9,
    "micrometer": 1e-6,
    "millimeter": 1e-3,
    "centimeter": 1e-2,
    "decimeter": 1e-1,
    "meter": 1.0,
    "hectometer": 1e2,
    "kilometer": 1e3,
    "megameter": 1e6,
    "gigameter": 1e9,
    "terameter": 1e12,
    "petameter": 1e15,
    "exameter": 1e18,
    "zettameter": 1e21,
    "yottameter": 1e24,
    "inch": 0.0254,
    "foot": 0.3048,
    "yard": 0.9144,
    "mile": 1609.344,
    "parsec": 3.0856775814913673e16,
}


def _space_factor(unit: object) -> float | None:
    """Meters per unit; `None` for an unknown string or `None` unit."""
    if isinstance(unit, str):
        return _METERS_PER_UNIT.get(unit)
    return None


def _warn_space_mismatch(left: object, right: object) -> None:
    warnings.warn(
        f"Comparing pixel sizes with different space units ({left!r} vs "
        f"{right!r}); magnitudes are unit-converted for the comparison.",
        NgioUserWarning,
        stacklevel=stacklevel_of_first_caller(),
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
        """Spatial equality, unit-converted when both space units are known.

        Differing time units, unconvertible space units, or a differing `t`
        compare unequal.
        """
        if not isinstance(other, PixelSize):
            return NotImplemented
        if self.time_unit != other.time_unit or not math.isclose(self.t, other.t):
            return False
        if self.space_unit != other.space_unit and (
            _space_factor(self.space_unit) is None
            or _space_factor(other.space_unit) is None
        ):
            return False
        scale = max(float(np.linalg.norm(self.zyx)), 1e-30)
        return self.distance(other) <= 1e-9 * scale

    def __lt__(self, other: "PixelSize") -> bool:
        """Order by spatial magnitude, unit-converted when both are known."""
        if not isinstance(other, PixelSize):
            raise TypeError("Can only compare PixelSize with PixelSize.")
        self_norm = float(np.linalg.norm(self.zyx))
        other_norm = float(np.linalg.norm(other.zyx))
        if self.space_unit != other.space_unit:
            self_factor = _space_factor(self.space_unit)
            other_factor = _space_factor(other.space_unit)
            if self_factor is not None and other_factor is not None:
                _warn_space_mismatch(self.space_unit, other.space_unit)
                return self_norm * self_factor < other_norm * other_factor
        return self_norm < other_norm

    def as_dict(self) -> dict[str, float]:
        """Return the pixel size as a dictionary."""
        return {"t": self.t, "z": self.z, "y": self.y, "x": self.x}

    def get(self, axis: str, default: float | None = None) -> float:
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
        """Spatial (z/y/x) distance in `self`'s unit.

        `other` is unit-converted when both space units are known (with a
        warning); `t` never participates.
        """
        other_zyx = np.array(other.zyx, dtype=float)
        if self.space_unit != other.space_unit:
            self_factor = _space_factor(self.space_unit)
            other_factor = _space_factor(other.space_unit)
            if self_factor is not None and other_factor is not None:
                _warn_space_mismatch(self.space_unit, other.space_unit)
                other_zyx = other_zyx * (other_factor / self_factor)
        return float(np.linalg.norm(np.array(self.zyx) - other_zyx))
