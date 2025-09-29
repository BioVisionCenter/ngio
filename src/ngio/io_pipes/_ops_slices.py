import math
from typing import TypeAlias, assert_never
from warnings import warn

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
    on_disk_shape: tuple[int, ...]
    slicing_tuple: tuple[SlicingType, ...] | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def normalized_slicing_tuple(self) -> None | tuple[SlicingType, ...]:
        """Normalize the slicing tuple to be within the array shape boundaries."""
        if self.slicing_tuple is not None:
            return _slicing_tuple_boundary_check(
                slicing_tuple=self.slicing_tuple,
                array_shape=self.on_disk_shape,
            )
        return None

    def get(self, ax_name: str, normalize: bool = False) -> SlicingType:
        """Get the slicing tuple."""
        slicing_tuple = (
            self.slicing_tuple if not normalize else self.normalized_slicing_tuple
        )
        if slicing_tuple is None:
            return slice(None)
        if ax_name not in self.on_disk_axes:
            return slice(None)
        ax_index = self.on_disk_axes.index(ax_name)
        return slicing_tuple[ax_index]


def _check_tuple_in_slicing_tuple(
    slicing_tuple: tuple[SlicingType, ...],
) -> tuple[None, None] | tuple[int, tuple[int, ...]]:
    """Check if there are any tuple in the slicing tuple.

    The zarr python api only supports int or slices, not tuples.
    Ngio support a single tuple in the slicing tuple to allow non-contiguous
    selection (main use case: selecting multiple channels).
    """
    # Find if the is any tuple in the slicing tuple
    # If there is one we need to handle it differently
    tuple_in_slice = [
        (i, s) for i, s in enumerate(slicing_tuple) if isinstance(s, tuple)
    ]
    if not tuple_in_slice:
        # No tuple in the slicing tuple
        return None, None

    if len(tuple_in_slice) > 1:
        raise NotImplementedError(
            "Slicing with multiple non-contiguous tuples/lists "
            "is not supported yet in Ngio. Use directly the "
            "zarr.Array api to get the correct array slice."
        )
    # Complex case, we have exactly one tuple in the slicing tuple
    ax, first_tuple = tuple_in_slice[0]
    if len(first_tuple) > 100:
        warn(
            "Performance warning: "
            "Non-contiguous slicing with a tuple/list with more than 100 elements is "
            "not natively supported by zarr. This is implemented by Ngio by performing "
            "multiple reads and stacking the result.",
            stacklevel=2,
        )
    return ax, first_tuple


def get_slice_as_numpy(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> np.ndarray:
    slicing_tuple = slicing_ops.normalized_slicing_tuple
    if slicing_tuple is None:
        # Base case, no slicing, return the full array
        return zarr_array[...]

    # Find if the is any tuple in the slicing tuple
    # If there is one we need to handle it differently
    ax, first_tuple = _check_tuple_in_slicing_tuple(slicing_tuple)
    if ax is None:
        # Simple case, no tuple in the slicing tuple
        return zarr_array[slicing_tuple]

    assert first_tuple is not None
    slices = [
        zarr_array[(*slicing_tuple[:ax], idx, *slicing_tuple[ax + 1 :])]
        for idx in first_tuple
    ]
    out_array = np.stack(slices, axis=ax)
    return out_array


def get_slice_as_dask(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> da.Array:
    da_array = da.from_zarr(zarr_array)
    slicing_tuple = slicing_ops.normalized_slicing_tuple
    if slicing_tuple is None:
        # Base case, no slicing, return the full array
        return da_array[...]

    # Find if the is any tuple in the slicing tuple
    # If there is one we need to handle it differently
    ax, first_tuple = _check_tuple_in_slicing_tuple(slicing_tuple)
    if ax is None:
        # Base case, no tuple in the slicing tuple
        return da_array[slicing_tuple]

    assert first_tuple is not None
    slices = [
        da_array[(*slicing_tuple[:ax], idx, *slicing_tuple[ax + 1 :])]
        for idx in first_tuple
    ]
    out_array = da.stack(slices, axis=ax)
    return out_array


def set_slice_as_numpy(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
) -> None:
    slice_tuple = slicing_ops.normalized_slicing_tuple
    if slice_tuple is None:
        # Base case, no slicing, write the full array
        zarr_array[...] = patch
        return

    ax, first_tuple = _check_tuple_in_slicing_tuple(slice_tuple)
    if ax is None:
        # Base case, no tuple in the slicing tuple
        zarr_array[slice_tuple] = patch
        return

    # Complex case, we have exactly one tuple in the slicing tuple
    assert first_tuple is not None
    for i, idx in enumerate(first_tuple):
        _sub_slice = (*slice_tuple[:ax], idx, *slice_tuple[ax + 1 :])
        zarr_array[_sub_slice] = np.take(patch, indices=i, axis=ax)


def set_slice_as_dask(
    zarr_array: zarr.Array, patch: da.Array, slicing_ops: SlicingOps
) -> None:
    slice_tuple = slicing_ops.normalized_slicing_tuple
    if slice_tuple is None:
        # Base case, no slicing, write the full array
        da.to_zarr(arr=patch, url=zarr_array)
        return
    ax, first_tuple = _check_tuple_in_slicing_tuple(slice_tuple)
    if ax is None:
        # Base case, no tuple in the slicing tuple
        da.to_zarr(arr=patch, url=zarr_array, region=slice_tuple)
        return

    # Complex case, we have exactly one tuple in the slicing tuple
    assert first_tuple is not None
    for i, idx in enumerate(first_tuple):
        _sub_slice = (*slice_tuple[:ax], slice(idx, idx + 1), *slice_tuple[ax + 1 :])
        sub_patch = da.take(patch, indices=i, axis=ax)
        sub_patch = da.expand_dims(sub_patch, axis=ax)
        da.to_zarr(arr=sub_patch, url=zarr_array, region=_sub_slice)
