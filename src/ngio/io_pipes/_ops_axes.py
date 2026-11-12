from collections.abc import Sequence
from typing import TypeVar

import dask.array as da
import numpy as np
from pydantic import BaseModel, ConfigDict

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import canonical_axes_order
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


def _require_singleton_squeeze(
    shape: tuple[int, ...], squeeze_axes: tuple[int, ...]
) -> None:
    """`axes_order` drops axes it omits; only size-1 axes may be dropped."""
    oversized = [axis for axis in squeeze_axes if shape[axis] != 1]
    if oversized:
        sizes = ", ".join(f"axis {axis} (size {shape[axis]})" for axis in oversized)
        raise NgioValueError(
            f"axes_order omits non-singleton axes: {sizes}. An omitted axis "
            "is dropped, which needs size 1 — slice it down first, or "
            "include it in axes_order."
        )


def _apply_numpy_axes_ops(
    array: np.ndarray,
    squeeze_axes: tuple[int, ...] | None = None,
    transpose_axes: tuple[int, ...] | None = None,
    expand_axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply axes operations to a numpy array."""
    if squeeze_axes is not None:
        _require_singleton_squeeze(array.shape, squeeze_axes)
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
        _require_singleton_squeeze(array.shape, squeeze_axes)
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
    for name in axes:
        if not isinstance(name, str):
            raise NgioValueError(
                f"Invalid axis name '{name}'. Axis names must be strings."
            )
    duplicates = sorted({name for name in axes if axes.count(name) > 1})
    if duplicates:
        raise NgioValueError(
            f"Duplicate axis names {duplicates} in axes_order={list(axes)}. "
            "Please provide unique names for each axis."
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

    An axis that is not on disk is kept as a virtual (singleton) axis only if
    it is a canonical axis, by its canonical name or the name the `AxesSetup`
    maps to it. Any other name is an error rather than a phantom dimension.
    """
    axes_handler = dimensions.axes_handler
    new_axes_order = []
    for axis_name in axes_order:
        axis = axes_handler.get_axis(axis_name)
        if axis is not None:
            new_axes_order.append(axis.name)
            continue
        canonical = axes_handler.get_canonical_name(axis_name) or axis_name
        if canonical not in canonical_axes_order():
            raise NgioValueError(
                f"axes_order={list(axes_order)} contains '{axis_name}', which is "
                f"neither an axis of this image {list(axes_handler.axes_names)} "
                f"nor a canonical axis {list(canonical_axes_order())}."
            )
        new_axes_order.append(canonical)
    return tuple(new_axes_order)


def validate_axes_order(dimensions: Dimensions, axes_order: Sequence[str]) -> None:
    """Fail now, not at the first read, if `axes_order` cannot be honoured."""
    _check_output_axes(_normalize_axes_order(dimensions, axes_order))


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
