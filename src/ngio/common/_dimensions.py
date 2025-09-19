"""Dimension metadata.

This is not related to the NGFF metadata,
but it is based on the actual metadata of the image data.
"""

from typing import overload

from ngio.ome_zarr_meta import AxesMapper
from ngio.utils import NgioValueError


class Dimensions:
    """Dimension metadata."""

    def __init__(
        self,
        shape: tuple[int, ...],
        axes_mapper: AxesMapper,
    ) -> None:
        """Create a Dimension object from a Zarr array.

        Args:
            shape: The shape of the Zarr array.
            axes_mapper: The axes mapper object.
        """
        self._shape = shape
        self._axes_mapper = axes_mapper

        if len(self._shape) != len(self._axes_mapper.axes):
            raise NgioValueError(
                "The number of dimensions must match the number of axes. "
                f"Expected Axis {self._axes_mapper.axes_names} but got shape "
                f"{self._shape}."
            )

    def __str__(self) -> str:
        """Return the string representation of the object."""
        dims = ", ".join(
            f"{ax.on_disk_name}: {s}"
            for ax, s in zip(self._axes_mapper.axes, self._shape, strict=True)
        )
        return f"Dimensions({dims})"

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
        index = self._axes_mapper.get_index(axis_name)
        if index is None:
            return default
        return self._shape[index]

    def get_index(self, axis_name: str) -> int | None:
        """Return the index of the given axis name.

        Args:
            axis_name: The name of the axis (either canonical or non-canonical).
        """
        return self._axes_mapper.get_index(axis_name)

    def has_axis(self, axis_name: str) -> bool:
        """Return whether the axis exists."""
        index = self._axes_mapper.get_axis(axis_name)
        if index is None:
            return False
        return True

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)

    @property
    def axes_mapper(self) -> AxesMapper:
        """Return the axes mapper object."""
        return self._axes_mapper

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return tuple(self._shape)

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the axes as a tuple of strings."""
        return self._axes_mapper.axes_names

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

    def assert_axes_match(self, other: "Dimensions") -> None:
        """Check if two Dimensions objects have the same axes.

        Besides the channel axis (which is a special case), all axes must be
        present in both Dimensions objects.

        Args:
            other (Dimensions): The other dimensions object to compare against.

        Raises:
            NgioValueError: If the axes do not match.
        """
        for s_axis in self.axes_mapper.axes:
            if s_axis.axis_type == "channel":
                continue
            o_axis = other.axes_mapper.get_axis(s_axis.on_disk_name)
            if o_axis is None:
                raise NgioValueError(
                    f"Axes do not match. The axis {s_axis.on_disk_name} "
                    f"is not present in either dimensions."
                )
        # Check for axes present in the other dimensions but not in this one
        for o_axis in other.axes_mapper.axes:
            if o_axis.axis_type == "channel":
                continue
            s_axis = self.axes_mapper.get_axis(o_axis.on_disk_name)
            if s_axis is None:
                raise NgioValueError(
                    f"Axes do not match. The axis {o_axis.on_disk_name} "
                    f"is not present in either dimensions."
                )

    def assert_dimensions_match(
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
        self.assert_axes_match(other)
        for s_axis in self.axes_mapper.axes:
            o_axis = other.axes_mapper.get_axis(s_axis.on_disk_name)
            assert o_axis is not None  # already checked in assert_axes_match

            i_dim = self.get(s_axis.on_disk_name, default=1)
            o_dim = other.get(o_axis.on_disk_name, default=1)

            if i_dim != o_dim:
                if allow_singleton and (i_dim == 1 or o_dim == 1):
                    continue
                raise NgioValueError(
                    f"Dimensions do not match for axis "
                    f"{s_axis.on_disk_name}. Got {i_dim} and {o_dim}."
                )
