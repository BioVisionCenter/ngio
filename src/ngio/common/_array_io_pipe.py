from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._io_transforms import (
    TransformProtocol,
    apply_dask_transforms,
    apply_numpy_transforms,
)
from ngio.ome_zarr_meta.ngio_specs import Axis, SlicingOps
from ngio.utils import NgioValueError

SlicingInputType: TypeAlias = slice | Sequence[int] | int | None
SlicingType: TypeAlias = slice | tuple[int, ...] | int
ArrayLike: TypeAlias = np.ndarray | DaskArray

##############################################################
#
# Slicing Operations
#
##############################################################


def _validate_int(value: int, shape: int) -> int:
    """Validate an integer value for slicing."""
    if not isinstance(value, int):
        raise NgioValueError(f"Invalid value {value} of type {type(value)}")
    if value < 0 or value >= shape:
        raise NgioValueError(
            f"Invalid value {value}. Index out of bounds for axis of shape {shape}"
        )
    return value


def _try_to_slice(input: Sequence[int]) -> slice | tuple[int, ...]:
    """Try to convert a list of integers into a slice if they are contiguous.

    - If the input is empty, return an empty tuple.
    - If the input is sorted, and contains contiguous integers,
      return a slice from the minimum to the maximum integer.
    - Otherwise, return the input as a tuple.

    This is useful for optimizing array slicing operations
    by allowing the use of slices when possible, which can be more efficient.
    """
    if not input:
        return ()

    # If the input is not sorted, return it as a tuple
    max_input = max(input)
    min_input = min(input)
    assert min_input >= 0, "Input must contain non-negative integers"
    assert max_input >= 0, "Input must contain non-negative integers"

    if sorted(input) == list(range(min_input, max_input + 1)):
        return slice(min_input, max_input + 1)

    return tuple(input)


def _validate_iter_of_ints(value: Sequence, shape: int) -> slice | tuple[int, ...]:
    value = [_validate_int(v, shape=shape) for v in value]
    return _try_to_slice(value)


def _validate_slice(value: slice, shape: int) -> slice:
    """Validate a slice object and return it with adjusted start and stop."""
    start = value.start if value.start is not None else 0
    start = max(start, 0)
    stop = value.stop if value.stop is not None else shape
    return slice(start, stop)


def _remove_channel_slicing(
    slicing_dict: dict[str, SlicingInputType],
    dimensions: Dimensions,
) -> dict[str, SlicingInputType]:
    """This utility function removes the channel selection from the slice kwargs.

    if ignore_channel_selection is True, it will remove the channel selection
    regardless of the dimensions. If the ignore_channel_selection is False
    it will fail.
    """
    if dimensions.is_multi_channels:
        return slicing_dict

    if "c" in slicing_dict:
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
        if slice_.start is None and slice_.stop == 0:
            return True
        if slice_.start == 0 and slice_.stop == 1:
            return True
    if isinstance(slice_, Sequence):
        if len(slice_) == 1 and slice_[0] == 0:
            return True
    return False


def _normalize_slicing_dict(
    dimensions: Dimensions,
    slicing_dict: Mapping[str, SlicingInputType],
    remove_channel_selection: bool = False,
) -> dict[str, SlicingInputType]:
    """Convert slice kwargs to the on-disk axes names."""
    normalized_slicing_dict: dict[str, SlicingInputType] = {}
    for axis_name, slice_ in slicing_dict.items():
        axis = dimensions.axes_mapper.get_axis(axis_name)
        if axis is None:
            # Virtual axes should be allowed to be selected
            # Common use case is still allowing channel_selection
            # When the zarr has not channel axis.
            if not _check_slicing_virtual_axes(slice_):
                raise NgioValueError(
                    f"Invalid axis selection:{axis_name}={slice_}. "
                    f"Not found on the on-disk axes {dimensions.on_disk_axes}."
                )
            # Virtual axes can be safely ignored
            continue
        on_disk_name = axis.on_disk_name
        if on_disk_name in normalized_slicing_dict:
            raise NgioValueError(
                f"Duplicate axis {on_disk_name} in slice kwargs. "
                "Please provide unique axis names."
            )
        normalized_slicing_dict[axis.on_disk_name] = slice_

    if remove_channel_selection:
        normalized_slicing_dict = _remove_channel_slicing(
            slicing_dict=normalized_slicing_dict, dimensions=dimensions
        )
    return normalized_slicing_dict


def _normalize_axes_order(
    dimensions: Dimensions,
    axes_order: Sequence[str],
) -> list[str]:
    """Convert axes order to the on-disk axes names.

    In this way there is not unambiguity in the axes order.
    """
    new_axes_order = []
    for axis_name in axes_order:
        axis = dimensions.axes_mapper.get_axis(axis_name)
        if axis is None:
            new_axes_order.append(axis_name)
        else:
            new_axes_order.append(axis.on_disk_name)
    return new_axes_order


def _normalize_slice_input(
    axis: Axis,
    slicing_dict: dict[str, SlicingInputType],
    dimensions: Dimensions,
    requires_axes_ops: bool,
    axes_order: list[str],
) -> SlicingType:
    """Normalize a slice input to a tuple of slices.

    Make sure that the slice is valid for the given axis and dimensions.
    And transform it to either a slice or a tuple of integers.
    If the axis is not present in the slicing_dict, return a full slice.
    """
    axis_name = axis.on_disk_name
    if axis_name not in slicing_dict:
        # If no slice is provided for the axis, use a full slice
        return slice(None)

    value = slicing_dict[axis_name]
    if isinstance(value, int):
        value = _validate_int(value, dimensions.get(axis_name))
        if requires_axes_ops or axis_name in axes_order:
            # Axes ops require all dimensions to be preserved
            value = slice(value, value + 1)
        return value
    elif isinstance(value, Sequence):
        return _validate_iter_of_ints(value, dimensions.get(axis_name))
    elif isinstance(value, slice):
        return _validate_slice(value, dimensions.get(axis_name))
    elif value is None:
        return slice(None)

    raise NgioValueError(f"Invalid slice definition {value} of type {type(value)}")


def _build_slicing_tuple(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType],
    axes_order: list[str] | None = None,
    requires_axes_ops: bool = False,
    remove_channel_selection: bool = False,
) -> tuple[SlicingType, ...] | None:
    """Assemble slices to be used to query the array."""
    if len(slicing_dict) == 0:
        # Skip unnecessary computation if no slicing is requested
        return None
    _axes_order = (
        _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)
        if axes_order is not None
        else []
    )
    _slicing_dict = _normalize_slicing_dict(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    slicing_tuple = tuple(
        _normalize_slice_input(
            axis=axis,
            slicing_dict=_slicing_dict,
            dimensions=dimensions,
            requires_axes_ops=requires_axes_ops,
            axes_order=_axes_order,
        )
        for axis in dimensions.axes_mapper.on_disk_axes
    )
    return slicing_tuple


def _get_slice_as_numpy(
    zarr_array: zarr.Array, slice_tuple: tuple[SlicingType, ...] | None
) -> np.ndarray:
    if slice_tuple is None:
        return zarr_array[...]

    if all(not isinstance(s, tuple) for s in slice_tuple):
        return zarr_array[slice_tuple]

    # If there are tuple[int, ...] we need to handle them separately
    # this is a workaround for the fact that zarr does not support
    # non-contiguous slicing with tuples/lists.
    first_slice_tuple = []
    for s in slice_tuple:
        if isinstance(s, tuple):
            first_slice_tuple.append(slice(None))
        else:
            first_slice_tuple.append(s)
    second_slice_tuple = []
    for s in slice_tuple:
        if isinstance(s, tuple):
            second_slice_tuple.append(s)
        else:
            second_slice_tuple.append(slice(None))

    return zarr_array[tuple(first_slice_tuple)][tuple(second_slice_tuple)]


def _get_slice_as_dask(
    zarr_array: zarr.Array, slice_tuple: tuple[SlicingType, ...] | None
) -> da.Array:
    da_array = da.from_zarr(zarr_array)
    if slice_tuple is None:
        return da_array

    if any(isinstance(s, tuple) for s in slice_tuple):
        raise NgioValueError(
            "Slicing with non-contiguous tuples/lists "
            "is not supported for Dask arrays. Use the "
            "numpy api to get the correct array slice."
        )
    return da_array[slice_tuple]


def _set_numpy_patch(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slice_tuple: tuple[SlicingType, ...] | None,
) -> None:
    if slice_tuple is None:
        zarr_array[...] = patch
        return
    zarr_array[slice_tuple] = patch


def _set_dask_patch(
    zarr_array: zarr.Array, patch: da.Array, slice_tuple: tuple[SlicingType, ...] | None
) -> None:
    da.to_zarr(arr=patch, url=zarr_array, region=slice_tuple)


##############################################################
#
# Array Axes Operations
#
##############################################################


def _apply_numpy_axes_ops(
    array: np.ndarray,
    squeeze_axes: tuple[int, ...] | None = None,
    transpose_axes: tuple[int, ...] | None = None,
    expand_axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply axes operations to a numpy array."""
    if squeeze_axes is not None:
        array = np.squeeze(array, axis=squeeze_axes)
    if transpose_axes is not None:
        array = np.transpose(array, axes=transpose_axes)
    if expand_axes is not None:
        array = np.expand_dims(array, axis=expand_axes)
    return array


def _apply_dask_axes_ops(
    array: da.Array,
    squeeze_axes: tuple[int, ...] | None = None,
    transpose_axes: tuple[int, ...] | None = None,
    expand_axes: tuple[int, ...] | None = None,
) -> da.Array:
    """Apply axes operations to a dask array."""
    if squeeze_axes is not None:
        array = da.squeeze(array, axis=squeeze_axes)
    if transpose_axes is not None:
        array = da.transpose(array, axes=transpose_axes)
    if expand_axes is not None:
        array = da.expand_dims(array, axis=expand_axes)
    return array


##############################################################
#
# Concrete "From Disk" Pipes
#
##############################################################


def setup_from_disk_pipe(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType],
    axes_order: Sequence[str] | None = None,
    remove_channel_selection: bool = False,
) -> SlicingOps:
    if axes_order is not None:
        axes_order = _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)
        slicing_ops = dimensions.axes_mapper.to_order(axes_order)
    else:
        slicing_ops = SlicingOps()

    slicing_tuple = _build_slicing_tuple(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        axes_order=axes_order,
        requires_axes_ops=slicing_ops.requires_axes_ops,
        remove_channel_selection=remove_channel_selection,
    )
    slicing_ops.slice_tuple = slicing_tuple
    return slicing_ops


def _numpy_get_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    _array = _get_slice_as_numpy(zarr_array, slice_tuple=slicing_ops.slice_tuple)
    _array = _apply_numpy_axes_ops(
        _array,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )

    _array = apply_numpy_transforms(
        _array, transforms=transforms, slicing_ops=slicing_ops
    )
    return _array


def _dask_get_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> DaskArray:
    _array = _get_slice_as_dask(zarr_array, slice_tuple=slicing_ops.slice_tuple)
    _array = _apply_dask_axes_ops(
        _array,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )

    _array = apply_dask_transforms(
        _array, transforms=transforms, slicing_ops=slicing_ops
    )
    return _array


def build_numpy_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]:
    """Get a numpy array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_from_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda: _numpy_get_pipe(
        zarr_array=zarr_array,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


def build_dask_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_from_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda: _dask_get_pipe(
        zarr_array=zarr_array,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


##############################################################
#
# Concrete "To Disk" Pipes
#
##############################################################


def setup_to_disk_pipe(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType],
    axes_order: Sequence[str] | None = None,
    remove_channel_selection: bool = False,
) -> SlicingOps:
    if axes_order is not None:
        axes_order = _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)
        slicing_ops = dimensions.axes_mapper.from_order(axes_order)
    else:
        slicing_ops = SlicingOps()

    slicing_tuple = _build_slicing_tuple(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        requires_axes_ops=slicing_ops.requires_axes_ops,
        remove_channel_selection=remove_channel_selection,
    )
    slicing_ops.slice_tuple = slicing_tuple
    return slicing_ops


def _numpy_set_pipe(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> None:
    _patch = apply_numpy_transforms(
        patch, transforms=transforms, slicing_ops=slicing_ops
    )
    _patch = _apply_numpy_axes_ops(
        _patch,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )
    _set_numpy_patch(zarr_array, _patch, slicing_ops.slice_tuple)


def _dask_set_pipe(
    zarr_array: zarr.Array,
    patch: DaskArray,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> None:
    _patch = apply_dask_transforms(
        patch, transforms=transforms, slicing_ops=slicing_ops
    )
    _patch = _apply_dask_axes_ops(
        _patch,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )
    _set_dask_patch(zarr_array, _patch, slicing_ops.slice_tuple)


def build_numpy_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]:
    """Set a numpy array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_to_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda patch: _numpy_set_pipe(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


def build_dask_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[DaskArray], None]:
    """Set a dask array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_to_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda patch: _dask_set_pipe(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


################################################################
#
# Masked Array Pipes
#
################################################################


def _label_to_bool_mask_numpy(
    label_data: np.ndarray | DaskArray,
    label: int | None = None,
    data_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Convert label data to a boolean mask."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    if data_shape is not None and label_data.shape != data_shape:
        bool_mask = np.broadcast_to(bool_mask, data_shape)
    return bool_mask


def build_masked_numpy_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    fill_value: int | float = 0,
) -> Callable[[], np.ndarray]:
    """Get a numpy array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    data_getter = build_numpy_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    label_data_getter = build_numpy_getter(
        zarr_array=label_zarr_array,
        dimensions=label_dimensions,
        axes_order=axes_order,
        transforms=label_transforms,
        slicing_dict=label_slicing_dict,
        remove_channel_selection=True,
    )

    def get_masked_data_as_numpy() -> np.ndarray:
        data = data_getter()
        label_data = label_data_getter()
        bool_mask = _label_to_bool_mask_numpy(
            label_data=label_data, label=label_id, data_shape=data.shape
        )
        masked_data = np.where(bool_mask, data, fill_value)
        return masked_data

    return get_masked_data_as_numpy


def _label_to_bool_mask_dask(
    label_data: DaskArray,
    label: int | None = None,
    data_shape: tuple[int, ...] | None = None,
) -> DaskArray:
    """Convert label data to a boolean mask for Dask arrays."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    if data_shape is not None and label_data.shape != data_shape:
        bool_mask = da.broadcast_to(bool_mask, data_shape)
    return bool_mask


def build_masked_dask_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    fill_value: int | float = 0,
) -> Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    data_getter = build_dask_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    label_data_getter = build_dask_getter(
        zarr_array=label_zarr_array,
        dimensions=label_dimensions,
        axes_order=axes_order,
        transforms=label_transforms,
        slicing_dict=label_slicing_dict,
        remove_channel_selection=True,
    )

    def get_masked_data_as_dask() -> DaskArray:
        data = data_getter()
        label_data = label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)
        bool_mask = _label_to_bool_mask_dask(
            label_data=label_data, label=label_id, data_shape=data_shape
        )
        masked_data = da.where(bool_mask, data, fill_value)
        return masked_data

    return get_masked_data_as_dask


def build_masked_numpy_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    data_getter: Callable[[], np.ndarray] | None = None,
    label_data_getter: Callable[[], np.ndarray] | None = None,
) -> Callable[[np.ndarray], None]:
    """Set a numpy array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_numpy_getter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=False,
        )

    if label_data_getter is None:
        label_data_getter = build_numpy_getter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    masked_data_setter = build_numpy_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    def set_patch_masked_as_numpy(patch: np.ndarray) -> None:
        """Set a numpy patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        bool_mask = _label_to_bool_mask_numpy(
            label_data=label_data, label=label_id, data_shape=data.shape
        )
        mask_data = np.where(bool_mask, patch, data)
        masked_data_setter(mask_data)

    return set_patch_masked_as_numpy


def build_masked_dask_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    data_getter: Callable[[], DaskArray] | None = None,
    label_data_getter: Callable[[], DaskArray] | None = None,
) -> Callable[[DaskArray], None]:
    """Set a dask array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_dask_getter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=False,
        )

    if label_data_getter is None:
        label_data_getter = build_dask_getter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    data_setter = build_dask_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    def set_patch_masked_as_dask(patch: DaskArray) -> None:
        """Set a dask patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)
        bool_mask = _label_to_bool_mask_dask(
            label_data=label_data, label=label_id, data_shape=data_shape
        )
        mask_data = da.where(bool_mask, patch, data)
        data_setter(mask_data)

    return set_patch_masked_as_dask
