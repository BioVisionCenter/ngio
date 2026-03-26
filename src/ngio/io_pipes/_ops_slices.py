import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.indexing import OrthogonalIndexer

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import Axis
from ngio.utils import NgioUserWarning, NgioValueError

logger = logging.getLogger(f"ngio:{__name__}")


SlicingInputType: TypeAlias = slice | Sequence[int] | int | None
SlicingType: TypeAlias = slice | list[int] | int

##############################################################
#
# "SlicingOps" model
#
##############################################################


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
            if start is not None and stop is not None and start > stop:
                stop = start
            out_slicing_tuple.append(slice(start, stop, step))
        elif isinstance(sl, int):
            _int_boundary_check(sl, shape=sh)
            out_slicing_tuple.append(sl)
        elif isinstance(sl, list):
            for i in sl:
                _int_boundary_check(i, shape=sh)
            out_slicing_tuple.append(sl)
        else:
            raise NgioValueError(
                f"Invalid slice {sl} of type {type(sl)} in slicing tuple\n"
                f"{slicing_tuple}. Allowed types are: int, slice or list of int."
            )

    return tuple(out_slicing_tuple)


class SlicingOps(BaseModel):
    """Class to hold slicing operations."""

    on_disk_axes: tuple[str, ...]
    on_disk_shape: tuple[int, ...]
    chunk_grid: ChunkGrid
    slicing_tuple: tuple[SlicingType, ...]
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def normalized_slicing_tuple(self) -> tuple[SlicingType, ...]:
        """Normalize the slicing tuple to be within the array shape boundaries."""
        return _slicing_tuple_boundary_check(
            slicing_tuple=self.slicing_tuple,
            array_shape=self.on_disk_shape,
        )

    @property
    def slice_axes(self) -> tuple[str, ...]:
        """The axes after slicing."""
        in_memory_axes = []
        for ax, sl in zip(self.on_disk_axes, self.slicing_tuple, strict=True):
            if isinstance(sl, int):
                continue
            in_memory_axes.append(ax)
        return tuple(in_memory_axes)

    def slice_chunks(self) -> set[tuple[int, ...]]:
        """The chunk coordinates required to read or write the slice."""
        indexer = OrthogonalIndexer(
            selection=self.normalized_slicing_tuple,
            shape=self.on_disk_shape,
            chunk_grid=self.chunk_grid,
        )
        return {cc.chunk_coords for cc in indexer}

    def get(self, ax_name: str, normalize: bool = False) -> SlicingType:
        """Get the slicing tuple."""
        slicing_tuple = (
            self.slicing_tuple if not normalize else self.normalized_slicing_tuple
        )
        if ax_name not in self.on_disk_axes:
            return slice(None)
        ax_index = self.on_disk_axes.index(ax_name)
        return slicing_tuple[ax_index]


def _check_list_in_slicing_tuple(
    slicing_tuple: tuple[SlicingType, ...],
) -> tuple[None, None] | tuple[int, list[int]]:
    """Check if there are any lists in the slicing tuple.

    Dask regions when setting data do not support non-contiguous
    selections natively.
    Ngio support a single list in the slicing tuple to allow non-contiguous
    selection (main use case: selecting multiple channels).
    """
    list_in_slice = [(i, s) for i, s in enumerate(slicing_tuple) if isinstance(s, list)]
    if not list_in_slice:
        return None, None

    if len(list_in_slice) > 1:
        raise NotImplementedError(
            "Slicing with multiple non-contiguous tuples/lists "
            "is not supported yet in Ngio. Use directly the "
            "zarr.Array api to get the correct array slice."
        )
    # Complex case, we have exactly one list in the slicing tuple
    ax, first_tuple = list_in_slice[0]
    if len(first_tuple) > 100:
        warnings.warn(
            "Performance warning: "
            "Non-contiguous slicing with a tuple/list with more than 100 elements is "
            "not natively supported by zarr. This is implemented by Ngio by performing "
            "multiple reads and stacking the result.",
            NgioUserWarning,
            stacklevel=2,
        )
    return ax, first_tuple


##############################################################
#
# Slicing implementations
#
##############################################################


def get_slice_as_numpy(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> np.ndarray:
    """Get a slice of a zarr array as a numpy array."""
    slicing_tuple = slicing_ops.normalized_slicing_tuple
    array = zarr_array[slicing_tuple]
    if not isinstance(array, np.ndarray):
        raise NgioValueError(
            f"Expected numpy array from zarr slice, got {type(array)}."
        )
    return array


def get_slice_as_dask(zarr_array: zarr.Array, slicing_ops: SlicingOps) -> da.Array:
    """Get a slice of a zarr array as a dask array."""
    da_array = da.from_zarr(zarr_array)
    slicing_tuple = slicing_ops.normalized_slicing_tuple
    return da_array[slicing_tuple]


def _check_compatibility_of_shapes(
    shape_zarr: tuple[int, ...],
    shape_patch: tuple[int, ...],
    slice_tuple: tuple[SlicingType, ...],
) -> None:
    """Check the compatibility zarr array (slices) and the patch."""
    expected_shape = []
    for sl, sh in zip(slice_tuple, shape_zarr, strict=True):
        if isinstance(sl, slice):
            start, stop, step = sl.start, sl.stop, sl.step
            if start is None:
                start = 0
            if stop is None:
                stop = sh
            expected_dim = math.ceil((stop - start) / (step or 1))
        elif isinstance(sl, int):
            continue  # int index reduces rank; not included in expected shape
        elif isinstance(sl, list):
            expected_dim = len(sl)
        else:
            raise NgioValueError(
                f"Invalid slice {sl} of type {type(sl)} in slicing tuple\n"
                f"{slice_tuple}. Allowed types are: int, slice or list of int."
            )
        expected_shape.append(expected_dim)

    expected_shape = tuple(expected_shape)
    if expected_shape != shape_patch:
        raise NgioValueError(
            f"Incompatible shapes for patch and slice.\n"
            f"- Patch shape: {shape_patch}\n"
            f"- Zarr array shape: {shape_zarr}\n"
            f"- Slice tuple: {slice_tuple}\n"
            f"- Expected shape: {shape_zarr}[{slice_tuple}] {expected_shape}\n"
        )


def set_slice_as_numpy(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
) -> None:
    slice_tuple = slicing_ops.normalized_slicing_tuple
    _check_compatibility_of_shapes(zarr_array.shape, patch.shape, slice_tuple)
    zarr_array[slice_tuple] = patch


def handle_int_set_as_dask(
    patch: da.Array,
    slicing_tuple: tuple[SlicingType, ...],
) -> tuple[da.Array, tuple[SlicingType, ...]]:
    """Handle the case where the slicing tuple contains integers.

    In this case we need to expand the patch array to match the slicing tuple.
    """
    new_slicing_tuple = list(slicing_tuple)
    for i, sl in enumerate(slicing_tuple):
        if isinstance(sl, int):
            patch = da.expand_dims(patch, axis=i)
            new_slicing_tuple[i] = slice(sl, sl + 1)
    return patch, tuple(new_slicing_tuple)


def set_slice_as_dask(
    zarr_array: zarr.Array, patch: da.Array, slicing_ops: SlicingOps
) -> None:
    slice_tuple = slicing_ops.normalized_slicing_tuple
    _check_compatibility_of_shapes(zarr_array.shape, patch.shape, slice_tuple)
    ax, first_tuple = _check_list_in_slicing_tuple(slice_tuple)
    patch, slice_tuple = handle_int_set_as_dask(patch, slice_tuple)
    if ax is None:
        da.to_zarr(arr=patch, url=zarr_array, region=slice_tuple)
        return

    # Complex case, we have exactly one list in the slicing tuple
    assert first_tuple is not None
    for i, idx in enumerate(first_tuple):
        _sub_slice = (*slice_tuple[:ax], slice(idx, idx + 1), *slice_tuple[ax + 1 :])
        sub_patch = da.take(patch, indices=i, axis=ax)
        sub_patch = da.expand_dims(sub_patch, axis=ax)
        da.to_zarr(arr=sub_patch, url=zarr_array, region=_sub_slice)


##############################################################
#
# Builder functions
#
##############################################################


def _try_to_slice(value: Sequence[int]) -> slice | list[int]:
    """Try to convert a sequence of integers into a slice if they are contiguous.

    - If the input is sorted and contains contiguous integers,
      return a slice from the minimum to the maximum integer.
    - Otherwise, return the input as a list of integers.
    """
    if not value:
        raise NgioValueError("Ngio does not support empty sequences as slice input.")

    if not all(isinstance(i, int) for i in value):
        _value = []
        for i in value:
            try:
                _value.append(int(i))
            except Exception as e:
                raise NgioValueError(
                    f"Invalid value {i} of type {type(i)} in sequence {value}"
                ) from e
        value = _value
    max_input = max(value)
    min_input = min(value)
    if min_input < 0:
        raise NgioValueError("Input must contain non-negative integers.")

    if sorted(value) == list(range(min_input, max_input + 1)):
        return slice(min_input, max_input + 1)

    return list(value)


def _remove_channel_slicing(
    slicing_dict: dict[str, SlicingInputType],
    dimensions: Dimensions,
) -> dict[str, SlicingInputType]:
    """Remove channel selection from the slicing dict for single-channel data."""
    if dimensions.is_multi_channels:
        return slicing_dict

    slicing_dict.pop("c", None)
    return slicing_dict


def _check_slicing_virtual_axes(slice_: SlicingInputType) -> bool:
    """Check if the slice_ is compatible with virtual axes.

    Virtual axes are axes that are not present in the actual data,
    such as time or channel axes in some datasets.
    So the only valid slices for virtual axes are:
    - None: means all data along the axis
    - 0: means the first element along the axis
    - slice([0, None], [1, None])
    """
    if slice_ is None or slice_ == 0:
        return True
    if isinstance(slice_, slice):
        if slice_.start is None and slice_.stop is None:
            return True
        if slice_.start == 0 and slice_.stop is None:
            return True
        if slice_.start == 0 and slice_.stop == 1:
            return True
    if isinstance(slice_, Sequence):
        if len(slice_) == 1 and slice_[0] == 0:
            return True
    return False


def _clean_slicing_dict(
    dimensions: Dimensions,
    slicing_dict: Mapping[str, SlicingInputType],
    remove_channel_selection: bool = False,
) -> dict[str, SlicingInputType]:
    """Clean the slicing dict.

    This function will:
        - Validate that the axes in the slicing_dict are present in the dimensions.
        - Make sure that the slicing_dict uses the on-disk axis names.
        - Check for duplicate axis names in the slicing_dict.
        - Remove channel selection if requested and data is single-channel.
    """
    clean_slicing_dict: dict[str, SlicingInputType] = {}
    for axis_name, slice_ in slicing_dict.items():
        axis = dimensions.axes_handler.get_axis(axis_name)
        if axis is None:
            # Virtual axes should be allowed to be selected
            # Common use case is still allowing channel_selection
            # When the zarr has not channel axis.
            if not _check_slicing_virtual_axes(slice_):
                raise NgioValueError(
                    f"Invalid axis selection:{axis_name}={slice_}. "
                    f"Not found on the on-disk axes {dimensions.axes}."
                )
            # Virtual axes can be safely ignored
            continue
        if axis.name in clean_slicing_dict:
            raise NgioValueError(
                f"Duplicate axis {axis.name} in slice kwargs. "
                "Please provide unique axis names."
            )
        clean_slicing_dict[axis.name] = slice_

    if remove_channel_selection:
        clean_slicing_dict = _remove_channel_slicing(
            slicing_dict=clean_slicing_dict, dimensions=dimensions
        )
    return clean_slicing_dict


def _normalize_slicing_tuple(
    axis: Axis,
    slicing_dict: dict[str, SlicingInputType],
) -> SlicingType:
    """Normalize the slicing dict to tuple.

    Since the slicing dict can contain different types of values
    We need to normalize them to more predictable types.
    The output types are:
    - slice
    - int
    - list of int (for non-contiguous selection)
    """
    axis_name = axis.name
    if axis_name not in slicing_dict:
        # If no slice is provided for the axis, use a full slice
        return slice(None)

    value = slicing_dict[axis_name]
    if value is None:
        return slice(None)
    if isinstance(value, slice) or isinstance(value, int):
        return value
    elif isinstance(value, Sequence):
        # If a contiguous sequence of integers is provided,
        # convert it to a slice for simplicity.
        # Alternatively, it will be converted to a list of ints
        return _try_to_slice(value)

    raise NgioValueError(
        f"Invalid slice definition {value} of type {type(value)}. "
        "Allowed types are: int, slice, sequence of int or None."
    )


def build_slicing_ops(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType] | None,
    remove_channel_selection: bool = False,
) -> SlicingOps:
    """Assemble slices to be used to query the array."""
    slicing_dict = slicing_dict or {}
    _slicing_dict = _clean_slicing_dict(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    slicing_tuple = tuple(
        _normalize_slicing_tuple(
            axis=axis,
            slicing_dict=_slicing_dict,
        )
        for axis in dimensions.axes_handler.axes
    )
    return SlicingOps(
        on_disk_axes=dimensions.axes_handler.axes_names,
        on_disk_shape=dimensions.shape,
        chunk_grid=dimensions.chunk_grid,
        slicing_tuple=slicing_tuple,
    )
