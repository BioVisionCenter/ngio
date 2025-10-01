from collections.abc import Mapping, Sequence
from typing import TypeAlias

from ngio.common._dimensions import Dimensions
from ngio.io_pipes._ops_slices import SlicingOps, SlicingType
from ngio.ome_zarr_meta.ngio_specs import Axis
from ngio.ome_zarr_meta.ngio_specs._axes import AxesOps
from ngio.utils import NgioValueError

SlicingInputType: TypeAlias = slice | Sequence[int] | int | None


def _try_to_slice(value: Sequence[int]) -> slice | tuple[int, ...]:
    """Try to convert a list of integers into a slice if they are contiguous.

    - If the input is empty, return an empty tuple.
    - If the input is sorted, and contains contiguous integers,
      return a slice from the minimum to the maximum integer.
    - Otherwise, return the input as a tuple.

    This is useful for optimizing array slicing operations
    by allowing the use of slices when possible, which can be more efficient.
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
    # If the input is not sorted, return it as a tuple
    max_input = max(value)
    min_input = min(value)
    assert min_input >= 0, "Input must contain non-negative integers"

    if sorted(value) == list(range(min_input, max_input + 1)):
        return slice(min_input, max_input + 1)

    return tuple(value)


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
        - Clean up channel selection if the dimensions
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


def _normalize_axes_order(
    dimensions: Dimensions,
    axes_order: Sequence[str],
) -> list[str]:
    """Convert axes order to the on-disk axes names.

    In this way there is not unambiguity in the axes order.
    """
    new_axes_order = []
    for axis_name in axes_order:
        axis = dimensions.axes_handler.get_axis(axis_name)
        if axis is None:
            new_axes_order.append(axis_name)
        else:
            new_axes_order.append(axis.name)
    return new_axes_order


def _normalize_slicing_tuple(
    axis: Axis,
    slicing_dict: dict[str, SlicingInputType],
    no_axes_ops: bool,
    axes_order: list[str],
) -> tuple[SlicingType, str | None]:
    """Normalize the slicing dict to tuple.

    Since the slicing dict can contain different types of values
    We need to normalize them to more predictable types.
    The output types are:
    - slice
    - int
    - tuple of int (for non-contiguous selection)
    """
    axis_name = axis.name
    if axis_name not in slicing_dict:
        # If no slice is provided for the axis, use a full slice
        return slice(None), None

    value = slicing_dict[axis_name]
    if value is None:
        return slice(None), None

    if isinstance(value, slice):
        return value, None
    elif isinstance(value, int):
        # If axes ops are requested, we need to preserve the dimension
        # When we slice because the axes ops will be applied later
        # If no axes ops are requested, we can safely keep the integer
        # which will remove the dimension
        if (not no_axes_ops) or (axis_name in axes_order):
            # Axes ops require all dimensions to be preserved
            value = slice(value, value + 1)
            return value, None
        return value, axis_name
    elif isinstance(value, Sequence):
        # If a contiguous sequence of integers is provided,
        # convert it to a slice for efficiency
        # Alternatively, it will be converted to a tuple of ints
        return _try_to_slice(value), None

    raise NgioValueError(
        f"Invalid slice definition {value} of type {type(value)}. "
        "Allowed types are: int, slice, sequence of int or None."
    )


def _build_slicing_tuple(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType],
    axes_order: list[str] | None = None,
    no_axes_ops: bool = False,
    remove_channel_selection: bool = False,
) -> tuple[tuple[SlicingType, ...] | None, list[str]]:
    """Assemble slices to be used to query the array."""
    if len(slicing_dict) == 0:
        # Skip unnecessary computation if no slicing is requested
        return None, []
    _axes_order = (
        _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)
        if axes_order is not None
        else []
    )
    _slicing_dict = _clean_slicing_dict(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    slicing_tuple = []
    axes_to_remove = []
    for axis in dimensions.axes_handler.axes:
        sl, ax_to_remove = _normalize_slicing_tuple(
            axis=axis,
            slicing_dict=_slicing_dict,
            no_axes_ops=no_axes_ops,
            axes_order=_axes_order,
        )
        slicing_tuple.append(sl)
        if ax_to_remove is not None:
            axes_to_remove.append(ax_to_remove)
    slicing_tuple = tuple(slicing_tuple)
    # Slicing tuple can have only one element of type tuple
    # If multiple tuple are present it will lead to errors
    # when querying the array
    if sum(isinstance(s, tuple) for s in slicing_tuple) > 1:
        raise NgioValueError(
            f"Invalid slicing tuple {slicing_tuple}. Ngio does not support "
            "multiple non-contiguous selections (tuples) in the slicing tuple. "
            "Please use slices or single integer selections instead."
        )
    return slicing_tuple, axes_to_remove


def _build_axes_ops(
    *,
    axes_order: Sequence[str] | None,
    dimensions: Dimensions,
) -> tuple[list[str] | None, AxesOps]:
    if axes_order is None:
        return None, AxesOps(
            on_disk_axes=dimensions.axes_handler.axes_names,
            in_memory_axes=dimensions.axes_handler.axes_names,
        )

    axes_order = _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)
    axes_ops = dimensions.axes_handler.get_axes_ops(axes_order)
    return axes_order, axes_ops


def setup_io_pipe(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    axes_order: Sequence[str] | None = None,
    remove_channel_selection: bool = False,
) -> tuple[SlicingOps, AxesOps]:
    """Setup the slicing tuple and axes ops for an IO pipe."""
    slicing_dict = slicing_dict or {}
    axes_order, axes_ops = _build_axes_ops(
        axes_order=axes_order,
        dimensions=dimensions,
    )

    slicing_tuple, axes_to_remove = _build_slicing_tuple(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        axes_order=axes_order,
        no_axes_ops=axes_ops.is_no_op,
        remove_channel_selection=remove_channel_selection,
    )

    if axes_to_remove:
        in_memory_axes = tuple(
            ax for ax in axes_ops.in_memory_axes if ax not in axes_to_remove
        )
        axes_ops = AxesOps(
            on_disk_axes=axes_ops.on_disk_axes,
            in_memory_axes=in_memory_axes,
        )
    slicing_ops = SlicingOps(
        on_disk_axes=dimensions.axes_handler.axes_names,
        slicing_tuple=slicing_tuple,
        on_disk_shape=dimensions.shape,
    )
    return slicing_ops, axes_ops
