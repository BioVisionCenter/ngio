from collections.abc import Sequence
from typing import TypeVar

import dask.array as da
import numpy as np

from ngio.ome_zarr_meta.ngio_specs._axes import AxesOps


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
