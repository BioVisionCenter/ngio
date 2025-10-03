"""Dimension metadata.

This is not related to the NGFF metadata,
but it is based on the actual metadata of the image data.
"""

import math
from typing import overload

from ngio.ome_zarr_meta import (
    AxesHandler,
)
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValueError


def _are_compatible(shape1: int, shape2: int, scaling: float) -> bool:
    """Check if shape2 is consistent with shape1 given pixel sizes.

    Since we only deal with shape discrepancies due to rounding, we
    shape1, needs to be larger than shape2.
    """
    if shape1 < shape2:
        return _are_compatible(shape2, shape1, 1 / scaling)
    expected_shape2 = shape1 * scaling
    expected_shape2_floor = math.floor(expected_shape2)
    expected_shape2_ceil = math.ceil(expected_shape2)
    return shape2 in {expected_shape2_floor, expected_shape2_ceil}


def require_axes_match(reference: "Dimensions", other: "Dimensions") -> None:
    """Check if two Dimensions objects have the same axes.

    Besides the channel axis (which is a special case), all axes must be
    present in both Dimensions objects.

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.

    Raises:
        NgioValueError: If the axes do not match.
    """
    for s_axis in reference.axes_handler.axes:
        if s_axis.axis_type == "channel":
            continue
        o_axis = other.axes_handler.get_axis(s_axis.name)
        if o_axis is None:
            raise NgioValueError(
                f"Axes do not match. The axis {s_axis.name} "
                f"is not present in either dimensions."
            )
    # Check for axes present in the other dimensions but not in this one
    for o_axis in other.axes_handler.axes:
        if o_axis.axis_type == "channel":
            continue
        s_axis = reference.axes_handler.get_axis(o_axis.name)
        if s_axis is None:
            raise NgioValueError(
                f"Axes do not match. The axis {o_axis.name} "
                f"is not present in either dimensions."
            )


def check_if_axes_match(reference: "Dimensions", other: "Dimensions") -> bool:
    """Check if two Dimensions objects have the same axes.

    Besides the channel axis (which is a special case), all axes must be
    present in both Dimensions objects.

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.

    Returns:
        bool: True if the axes match, False otherwise.
    """
    try:
        require_axes_match(reference, other)
        return True
    except NgioValueError:
        return False


def require_dimensions_match(
    reference: "Dimensions", other: "Dimensions", allow_singleton: bool = False
) -> None:
    """Check if two Dimensions objects have the same axes and dimensions.

    Besides the channel axis, all axes must have the same dimension in
    both images.

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.
        allow_singleton (bool): Whether to allow singleton dimensions to be
            different. For example, if the input image has shape
            (5, 100, 100) and the label has shape (1, 100, 100).

    Raises:
        NgioValueError: If the dimensions do not match.
    """
    require_axes_match(reference, other)
    for r_axis in reference.axes_handler.axes:
        if r_axis.axis_type == "channel":
            continue
        o_axis = other.axes_handler.get_axis(r_axis.name)
        assert o_axis is not None  # already checked in assert_axes_match

        r_dim = reference.get(r_axis.name, default=1)
        o_dim = other.get(o_axis.name, default=1)

        if r_dim != o_dim:
            if allow_singleton and (r_dim == 1 or o_dim == 1):
                continue
            raise NgioValueError(
                f"Dimensions do not match for axis "
                f"{r_axis.name}. Got {r_dim} and {o_dim}."
            )


def check_if_dimensions_match(
    reference: "Dimensions", other: "Dimensions", allow_singleton: bool = False
) -> bool:
    """Check if two Dimensions objects have the same axes and dimensions.

    Besides the channel axis, all axes must have the same dimension in
    both images.

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.
        allow_singleton (bool): Whether to allow singleton dimensions to be
            different. For example, if the input image has shape
            (5, 100, 100) and the label has shape (1, 100, 100).

    Returns:
        bool: True if the dimensions match, False otherwise.
    """
    try:
        require_dimensions_match(reference, other, allow_singleton)
        return True
    except NgioValueError:
        return False


def require_rescalable(reference: "Dimensions", other: "Dimensions") -> None:
    """Assert that two images can be rescaled.

    For this to be true, the images must have the same axes, and
    the pixel sizes must be compatible (i.e. one can be scaled to the other).

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.

    """
    require_axes_match(reference, other)
    for ax_r in reference.axes_handler.axes:
        if ax_r.axis_type == "channel":
            continue
        ax_o = other.axes_handler.get_axis(ax_r.name)
        assert ax_o is not None, "Axes do not match."
        px_r = reference.pixel_size.get(ax_r.name, default=1.0)
        px_o = other.pixel_size.get(ax_o.name, default=1.0)
        shape_r = reference.get(ax_r.name, default=1)
        shape_o = other.get(ax_o.name, default=1)
        scale = px_r / px_o
        if not _are_compatible(
            shape1=shape_r,
            shape2=shape_o,
            scaling=scale,
        ):
            raise NgioValueError(
                f"Reference image with shape {reference.shape}, "
                f"and pixel size {reference.pixel_size}, "
                f"cannot be rescaled to "
                f"image with shape {other.shape} "
                f"and pixel size {other.pixel_size}. "
            )


def check_if_rescalable(reference: "Dimensions", other: "Dimensions") -> bool:
    """Check if two images can be rescaled.

    For this to be true, the images must have the same axes, and
    the pixel sizes must be compatible (i.e. one can be scaled to the other).

    Args:
        reference (Dimensions): The reference dimensions object to compare against.
        other (Dimensions): The other dimensions object to compare against.

    Returns:
        bool: True if the images can be rescaled, False otherwise.
    """
    try:
        require_rescalable(reference, other)
        return True
    except NgioValueError:
        return False


class Dimensions:
    """Dimension metadata Handling Class.

    This class is used to handle and manipulate dimension metadata.
    It provides methods to access and validate dimension information,
    such as shape, axes, and properties like is_2d, is_3d, is_time_series, etc.
    """

    require_axes_match = require_axes_match
    check_if_axes_match = check_if_axes_match
    require_dimensions_match = require_dimensions_match
    check_if_dimensions_match = check_if_dimensions_match
    require_rescalable = require_rescalable
    check_if_rescalable = check_if_rescalable

    def __init__(
        self,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dataset: Dataset,
    ) -> None:
        """Create a Dimension object from a Zarr array.

        Args:
            shape: The shape of the Zarr array.
            chunks: The chunks of the Zarr array.
            dataset: The dataset object.
        """
        self._shape = shape
        self._chunks = chunks
        self._axes_handler = dataset.axes_handler
        self._pixel_size = dataset.pixel_size

        if len(self._shape) != len(self._axes_handler.axes):
            raise NgioValueError(
                "The number of dimensions must match the number of axes. "
                f"Expected Axis {self._axes_handler.axes_names} but got shape "
                f"{self._shape}."
            )

    def __str__(self) -> str:
        """Return the string representation of the object."""
        dims = ", ".join(
            f"{ax.name}: {s}"
            for ax, s in zip(self._axes_handler.axes, self._shape, strict=True)
        )
        return f"Dimensions({dims})"

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)

    @property
    def axes_handler(self) -> AxesHandler:
        """Return the axes handler object."""
        return self._axes_handler

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size object."""
        return self._pixel_size

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return self._shape

    @property
    def chunks(self) -> tuple[int, ...]:
        """Return the chunks as a tuple."""
        return self._chunks

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the axes as a tuple of strings."""
        return self.axes_handler.axes_names

    @property
    def is_time_series(self) -> bool:
        """Return whether the image is a time series."""
        if self.get("t", default=1) == 1:
            return False
        return True

    @property
    def is_2d(self) -> bool:
        """Return whether the image is 2D."""
        if self.get("z", default=1) != 1:
            return False
        return True

    @property
    def is_2d_time_series(self) -> bool:
        """Return whether the image is a 2D time series."""
        return self.is_2d and self.is_time_series

    @property
    def is_3d(self) -> bool:
        """Return whether the image is 3D."""
        return not self.is_2d

    @property
    def is_3d_time_series(self) -> bool:
        """Return whether the image is a 3D time series."""
        return self.is_3d and self.is_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return whether the image has multiple channels."""
        if self.get("c", default=1) == 1:
            return False
        return True

    @overload
    def get(self, axis_name: str, default: None = None) -> int | None:
        pass

    @overload
    def get(self, axis_name: str, default: int) -> int:
        pass

    def get(self, axis_name: str, default: int | None = None) -> int | None:
        """Return the dimension/shape of the given axis name.

        Args:
            axis_name: The name of the axis (either canonical or non-canonical).
            default: The default value to return if the axis does not exist.
        """
        index = self.axes_handler.get_index(axis_name)
        if index is None:
            return default
        return self._shape[index]
