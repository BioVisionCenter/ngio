import math
from typing import TypeAlias, assert_never

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict

from ngio.utils import NgioValueError

SlicingType: TypeAlias = slice | tuple[int, ...] | int


def _int_boundary_check(value: int, shape: int) -> int:
    """Ensure that the integer value is within the boundaries of the array shape."""
    if value < 0 or value >= shape:
        raise NgioValueError(
            f"Invalid index {value}. Index is out of bounds for axis with size {shape}."
        )
    return value


def _slicing_tuple_boundary_check(
    slicing_tuple: tuple[SlicingType, ...],
    array_shape: tuple[int, ...],
) -> tuple[SlicingType, ...]:
    """Ensure that the slicing tuple is within the boundaries of the array shape.

    This function normalizes the slicing tuple to ensure that the selection
    is within the boundaries of the array shape.
    """
    if len(slicing_tuple) != len(array_shape):
        raise NgioValueError(
            f"Invalid slicing tuple {slicing_tuple}. "
            f"Length {len(slicing_tuple)} does not match array shape {array_shape}."
        )
    out_slicing_tuple = []
    for sl, sh in zip(slicing_tuple, array_shape, strict=True):
        if isinstance(sl, slice):
            start, stop, step = sl.start, sl.stop, sl.step
            if start is not None:
                start = math.floor(start)
                start = max(0, min(start, sh))
            if stop is not None:
                stop = math.ceil(stop)
                stop = max(0, min(stop, sh))
            out_slicing_tuple.append(slice(start, stop, step))
        elif isinstance(sl, int):
            _int_boundary_check(sl, shape=sh)
            out_slicing_tuple.append(sl)
        elif isinstance(sl, tuple):
            [_int_boundary_check(i, shape=sh) for i in sl]
            out_slicing_tuple.append(sl)
        else:
            assert_never(sl)

    return tuple(out_slicing_tuple)


class SlicingOps(BaseModel):
    """Class to hold slicing operations."""

    on_disk_axes: tuple[str, ...]
    slicing_tuple: tuple[SlicingType, ...] | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def normalize_slicing_tuple(
        self, array_shape: tuple[int, ...]
    ) -> None | tuple[SlicingType, ...]:
        """Normalize the slicing tuple to be within the array shape boundaries."""
        if self.slicing_tuple is not None:
            return _slicing_tuple_boundary_check(
                slicing_tuple=self.slicing_tuple,
                array_shape=array_shape,
            )
        return None

    def get(self, ax_name: str) -> SlicingType:
        """Get the slicing tuple."""
        if self.slicing_tuple is None:
            return slice(None)
        ax_index = self.on_disk_axes.index(ax_name)
        return self.slicing_tuple[ax_index]


def get_slice_as_numpy(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> np.ndarray:
    slicing_tuple = slicing_ops.normalize_slicing_tuple(array_shape=zarr_array.shape)
    if slicing_tuple is None:
        return zarr_array[...]

    if all(not isinstance(s, tuple) for s in slicing_tuple):
        return zarr_array[slicing_tuple]

    # If there are tuple with int we need to handle them separately
    # this is a workaround for the fact that zarr does not support
    # non-contiguous slicing with tuples/lists.
    # TODO to be redone properly
    first_slice_tuple = []
    for s in slicing_tuple:
        if isinstance(s, tuple):
            first_slice_tuple.append(slice(None))
        else:
            first_slice_tuple.append(s)
    second_slice_tuple = []
    for s in slicing_tuple:
        if isinstance(s, tuple):
            second_slice_tuple.append(s)
        else:
            second_slice_tuple.append(slice(None))

    return zarr_array[tuple(first_slice_tuple)][tuple(second_slice_tuple)]


def get_slice_as_dask(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> da.Array:
    da_array = da.from_zarr(zarr_array)
    slice_tuple = slicing_ops.normalize_slicing_tuple(array_shape=zarr_array.shape)
    if slice_tuple is None:
        return da_array

    slice_tuple = _slicing_tuple_boundary_check(slice_tuple, zarr_array.shape)
    # TODO add support for non-contiguous slicing with tuples/lists
    if any(isinstance(s, tuple) for s in slice_tuple):
        raise NotImplementedError(
            "Slicing with non-contiguous tuples/lists "
            "is not supported yet for Dask arrays. Use the "
            "numpy api to get the correct array slice."
        )
    return da_array[slice_tuple]


def set_slice_as_numpy(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
) -> None:
    slice_tuple = slicing_ops.normalize_slicing_tuple(array_shape=zarr_array.shape)
    if slice_tuple is None:
        zarr_array[...] = patch
        return

    slice_tuple = _slicing_tuple_boundary_check(slice_tuple, zarr_array.shape)
    zarr_array[slice_tuple] = patch


def set_slice_as_dask(
    zarr_array: zarr.Array, patch: da.Array, slicing_ops: SlicingOps
) -> None:
    slice_tuple = slicing_ops.normalize_slicing_tuple(array_shape=zarr_array.shape)
    if slice_tuple is not None:
        slice_tuple = _slicing_tuple_boundary_check(slice_tuple, zarr_array.shape)
    da.to_zarr(arr=patch, url=zarr_array, region=slice_tuple)
