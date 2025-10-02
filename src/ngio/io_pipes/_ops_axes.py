from collections.abc import Sequence
from typing import TypeVar

import dask.array as da
import numpy as np
from pydantic import BaseModel, ConfigDict

from ngio.common._dimensions import Dimensions
from ngio.utils import NgioValueError

##############################################################
#
# "AxesOps" Model
#
##############################################################


class AxesOps(BaseModel):
    """Model to represent axes operations.

    This model will be used to transform objects from on disk axes to in memory axes.
    """

    input_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    transpose_op: tuple[int, ...] | None = None
    expand_op: tuple[int, ...] | None = None
    squeeze_op: tuple[int, ...] | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def is_no_op(self) -> bool:
        """Check if all operations are no ops."""
        if (
            self.transpose_op is None
            and self.expand_op is None
            and self.squeeze_op is None
        ):
            return True
        return False

    @property
    def get_transpose_op(self) -> tuple[int, ...] | None:
        """Get the transpose axes."""
        return self.transpose_op

    @property
    def get_expand_op(self) -> tuple[int, ...] | None:
        """Get the expand axes."""
        return self.expand_op

    @property
    def get_squeeze_op(self) -> tuple[int, ...] | None:
        """Get the squeeze axes."""
        return self.squeeze_op

    @property
    def set_transpose_op(self) -> tuple[int, ...] | None:
        """Set the transpose axes."""
        if self.transpose_op is None:
            return None
        return tuple(np.argsort(self.transpose_op))

    @property
    def set_expand_op(self) -> tuple[int, ...] | None:
        """Set the expand axes."""
        return self.squeeze_op

    @property
    def set_squeeze_op(self) -> tuple[int, ...] | None:
        """Set the squeeze axes."""
        return self.expand_op


##############################################################
#
# Axes Operations implementations
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


T = TypeVar("T")


def _apply_sequence_axes_ops(
    input_: Sequence[T],
    default: T,
    squeeze_axes: tuple[int, ...] | None = None,
    transpose_axes: tuple[int, ...] | None = None,
    expand_axes: tuple[int, ...] | None = None,
) -> list[T]:
    input_list = list(input_)
    if squeeze_axes is not None:
        for offset, ax in enumerate(squeeze_axes):
            input_list.pop(ax - offset)

    if transpose_axes is not None:
        input_list = [input_list[i] for i in transpose_axes]

    if expand_axes is not None:
        for ax in expand_axes:
            input_list.insert(ax, default)

    return input_list


def get_as_numpy_axes_ops(
    array: np.ndarray,
    axes_ops: AxesOps,
) -> np.ndarray:
    """Apply axes operations to a numpy array."""
    return _apply_numpy_axes_ops(
        array,
        squeeze_axes=axes_ops.get_squeeze_op,
        transpose_axes=axes_ops.get_transpose_op,
        expand_axes=axes_ops.get_expand_op,
    )


def get_as_dask_axes_ops(
    array: da.Array,
    axes_ops: AxesOps,
) -> da.Array:
    """Apply axes operations to a dask array."""
    return _apply_dask_axes_ops(
        array,
        squeeze_axes=axes_ops.get_squeeze_op,
        transpose_axes=axes_ops.get_transpose_op,
        expand_axes=axes_ops.get_expand_op,
    )


def get_as_sequence_axes_ops(
    input_: Sequence[T],
    axes_ops: AxesOps,
    default: T,
) -> list[T]:
    """Apply axes operations to a sequence."""
    return _apply_sequence_axes_ops(
        input_,
        default=default,
        squeeze_axes=axes_ops.get_squeeze_op,
        transpose_axes=axes_ops.get_transpose_op,
        expand_axes=axes_ops.get_expand_op,
    )


def set_as_numpy_axes_ops(
    array: np.ndarray,
    axes_ops: AxesOps,
) -> np.ndarray:
    """Apply inverse axes operations to a numpy array."""
    return _apply_numpy_axes_ops(
        array,
        squeeze_axes=axes_ops.set_squeeze_op,
        transpose_axes=axes_ops.set_transpose_op,
        expand_axes=axes_ops.set_expand_op,
    )


def set_as_dask_axes_ops(
    array: da.Array,
    axes_ops: AxesOps,
) -> da.Array:
    """Apply inverse axes operations to a dask array."""
    return _apply_dask_axes_ops(
        array,
        squeeze_axes=axes_ops.set_squeeze_op,
        transpose_axes=axes_ops.set_transpose_op,
        expand_axes=axes_ops.set_expand_op,
    )


def set_as_sequence_axes_ops(
    input_: Sequence[T],
    axes_ops: AxesOps,
    default: T,
) -> list[T]:
    """Apply inverse axes operations to a sequence."""
    return _apply_sequence_axes_ops(
        input_,
        default=default,
        squeeze_axes=axes_ops.set_squeeze_op,
        transpose_axes=axes_ops.set_transpose_op,
        expand_axes=axes_ops.set_expand_op,
    )


##############################################################
#
# Builder functions
#
##############################################################


def _check_output_axes(axes: Sequence[str]) -> None:
    """Check that the input axes are valid."""
    unique_names = set(axes)
    if len(unique_names) != len(axes):
        raise NgioValueError(
            "Duplicate axis names found. Please provide unique names for each axis."
        )
    for name in axes:
        if not isinstance(name, str):
            raise NgioValueError(
                f"Invalid axis name '{name}'. Axis names must be strings."
            )


def _build_squeeze_tuple(
    input_axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Build a tuple of axes to squeeze."""
    axes_to_squeeze = []
    axes_after_squeeze = []
    for i, ax in enumerate(input_axes):
        if ax not in output_axes:
            axes_to_squeeze.append(i)
        else:
            axes_after_squeeze.append(ax)
    return tuple(axes_to_squeeze), tuple(axes_after_squeeze)


def _build_transpose_tuple(
    input_axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Build a tuple of axes to transpose."""
    transposition_order = []
    axes_names_after_transpose = []
    for ax in output_axes:
        if ax in input_axes:
            transposition_order.append(input_axes.index(ax))
            axes_names_after_transpose.append(ax)
    return tuple(transposition_order), tuple(axes_names_after_transpose)


def _build_expand_tuple(
    input_axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> tuple[int, ...]:
    """Build a tuple of axes to expand."""
    axes_to_expand = []
    for i, ax in enumerate(output_axes):
        if ax not in input_axes:
            axes_to_expand.append(i)
    return tuple(axes_to_expand)


def _build_axes_ops(
    input_axes: tuple[str, ...], output_axes: tuple[str, ...]
) -> AxesOps:
    """Change the order of the axes."""
    # Validate the names
    _check_output_axes(output_axes)
    # Step 1: Check find squeeze axes
    axes_to_squeeze, input_axes = _build_squeeze_tuple(input_axes, output_axes)
    # Step 2: Find the transposition order
    transposition_order, input_axes = _build_transpose_tuple(input_axes, output_axes)
    # Step 3: Find axes to expand
    axes_to_expand = _build_expand_tuple(input_axes, output_axes)

    # If the operations are empty, make them None
    if len(axes_to_squeeze) == 0:
        axes_to_squeeze = None

    if np.allclose(transposition_order, np.arange(len(transposition_order))):
        # If the transposition order is the identity, we don't need to transpose
        transposition_order = None
    if len(axes_to_expand) == 0:
        axes_to_expand = None

    return AxesOps(
        input_axes=input_axes,
        output_axes=output_axes,
        transpose_op=transposition_order,
        expand_op=axes_to_expand,
        squeeze_op=axes_to_squeeze,
    )


def _normalize_axes_order(
    dimensions: Dimensions,
    axes_order: Sequence[str],
) -> tuple[str, ...]:
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
    return tuple(new_axes_order)


def build_axes_ops(
    *,
    dimensions: Dimensions,
    input_axes: tuple[str, ...],
    axes_order: Sequence[str] | None,
) -> AxesOps:
    if axes_order is None:
        return AxesOps(
            input_axes=input_axes,
            output_axes=input_axes,
        )
    output_axes = _normalize_axes_order(dimensions=dimensions, axes_order=axes_order)

    axes_ops = _build_axes_ops(input_axes=input_axes, output_axes=output_axes)
    return axes_ops
