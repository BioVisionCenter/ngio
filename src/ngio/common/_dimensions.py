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


class Dimensions:
    """Dimension metadata Handling Class.

    This class is used to handle and manipulate dimension metadata.
    It provides methods to access and validate dimension information,
    such as shape, axes, and properties like is_2d, is_3d, is_time_series, etc.
    """

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
        """Return whether the data is a time series."""
        if self.get("t", default=1) == 1:
            return False
        return True

    @property
    def is_2d(self) -> bool:
        """Return whether the data is 2D."""
        if self.get("z", default=1) != 1:
            return False
        return True

    @property
    def is_2d_time_series(self) -> bool:
        """Return whether the data is a 2D time series."""
        return self.is_2d and self.is_time_series

    @property
    def is_3d(self) -> bool:
        """Return whether the data is 3D."""
        return not self.is_2d

    @property
    def is_3d_time_series(self) -> bool:
        """Return whether the data is a 3D time series."""
        return self.is_3d and self.is_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return whether the data has multiple channels."""
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
        """Return the dimension of the given axis name.

        Args:
            axis_name: The name of the axis (either canonical or non-canonical).
            default: The default value to return if the axis does not exist.
        """
        index = self.axes_handler.get_index(axis_name)
        if index is None:
            return default
        return self._shape[index]

    def get_index(self, axis_name: str) -> int | None:
        """Return the index of the given axis name.

        Args:
            axis_name: The name of the axis (either canonical or non-canonical).
        """
        return self.axes_handler.get_index(axis_name)

    def has_axis(self, axis_name: str) -> bool:
        """Return whether the axis exists."""
        index = self.axes_handler.get_index(axis_name)
        if index is None:
            return False
        return True

    def require_axes_match(self, other: "Dimensions") -> None:
        """Check if two Dimensions objects have the same axes.

        Besides the channel axis (which is a special case), all axes must be
        present in both Dimensions objects.

        Args:
            other (Dimensions): The other dimensions object to compare against.

        Raises:
            NgioValueError: If the axes do not match.
        """
        require_axes_match(self, other)

    def require_dimensions_match(
        self, other: "Dimensions", allow_singleton: bool = False
    ) -> None:
        """Check if two Dimensions objects have the same axes and dimensions.

        Besides the channel axis, all axes must have the same dimension in
        both images.

        Args:
            other (Dimensions): The other dimensions object to compare against.
            allow_singleton (bool): Whether to allow singleton dimensions to be
                different. For example, if the input image has shape
                (5, 100, 100) and the label has shape (1, 100, 100).

        Raises:
            NgioValueError: If the dimensions do not match.
        """
        require_dimensions_match(self, other, allow_singleton)

    def require_can_be_rescaled(self, other: "Dimensions") -> None:
        """Assert that two images can be rescaled.

        For this to be true, the images must have the same axes, and
        the pixel sizes must be compatible (i.e. one can be scaled to the other).

        Args:
            other (Dimensions): The other dimensions object to compare against.

        """
        require_rescalable(self, other)


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


def require_axes_match(dimensions1: Dimensions, dimensions2: Dimensions) -> None:
    """Check if two Dimensions objects have the same axes.

    Besides the channel axis (which is a special case), all axes must be
    present in both Dimensions objects.

    Args:
        dimensions1 (Dimensions): The first dimensions object to compare against.
        dimensions2 (Dimensions): The second dimensions object to compare against.

    Raises:
        NgioValueError: If the axes do not match.
    """
    for s_axis in dimensions1.axes_handler.axes:
        if s_axis.axis_type == "channel":
            continue
        o_axis = dimensions2.axes_handler.get_axis(s_axis.name)
        if o_axis is None:
            raise NgioValueError(
                f"Axes do not match. The axis {s_axis.name} "
                f"is not present in either dimensions."
            )
    # Check for axes present in the other dimensions but not in this one
    for o_axis in dimensions2.axes_handler.axes:
        if o_axis.axis_type == "channel":
            continue
        s_axis = dimensions1.axes_handler.get_axis(o_axis.name)
        if s_axis is None:
            raise NgioValueError(
                f"Axes do not match. The axis {o_axis.name} "
                f"is not present in either dimensions."
            )


def require_dimensions_match(
    dimensions1: Dimensions, dimensions2: Dimensions, allow_singleton: bool = False
) -> None:
    """Check if two Dimensions objects have the same axes and dimensions.

    Besides the channel axis, all axes must have the same dimension in
    both images.

    Args:
        dimensions1 (Dimensions): The first dimensions object to compare against.
        dimensions2 (Dimensions): The second dimensions object to compare against.
        allow_singleton (bool): Whether to allow singleton dimensions to be
            different. For example, if the input image has shape
            (5, 100, 100) and the label has shape (1, 100, 100).

    Raises:
        NgioValueError: If the dimensions do not match.
    """
    require_axes_match(dimensions1, dimensions2)
    for s_axis in dimensions1.axes_handler.axes:
        if s_axis.axis_type == "channel":
            continue
        o_axis = dimensions2.axes_handler.get_axis(s_axis.name)
        assert o_axis is not None  # already checked in assert_axes_match

        i_dim = dimensions1.get(s_axis.name, default=1)
        o_dim = dimensions2.get(o_axis.name, default=1)

        if i_dim != o_dim:
            if allow_singleton and (i_dim == 1 or o_dim == 1):
                continue
            raise NgioValueError(
                f"Dimensions do not match for axis "
                f"{s_axis.name}. Got {i_dim} and {o_dim}."
            )


def require_rescalable(dimensions1: Dimensions, dimensions2: Dimensions) -> None:
    """Assert that two images can be rescaled.

    For this to be true, the images must have the same axes, and
    the pixel sizes must be compatible (i.e. one can be scaled to the other).

    Args:
        dimensions1 (Dimensions): The first dimensions object to compare against.
        dimensions2 (Dimensions): The second dimensions object to compare against.

    """
    require_axes_match(dimensions1, dimensions2)
    for ax1 in dimensions1.axes_handler.axes:
        if ax1.axis_type == "channel":
            continue
        ax2 = dimensions2.axes_handler.get_axis(ax1.name)
        assert ax2 is not None, "Axes do not match."
        px1 = dimensions1.pixel_size.get(ax1.name, default=1.0)
        px2 = dimensions2.pixel_size.get(ax2.name, default=1.0)
        shape1 = dimensions1.get(ax1.name, default=1)
        shape2 = dimensions2.get(ax2.name, default=1)
        scale = px1 / px2
        if not _are_compatible(
            shape1=shape1,
            shape2=shape2,
            scaling=scale,
        ):
            raise NgioValueError(
                f"Image1 with shape {dimensions1.shape}, "
                f"and pixel size {dimensions1.pixel_size}, "
                f"cannot be rescaled to "
                f"Image2 with shape {dimensions2.shape}, "
                f"and pixel size {dimensions2.pixel_size}. "
            )
