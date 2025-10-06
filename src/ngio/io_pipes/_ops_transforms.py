from collections.abc import Sequence
from typing import Protocol

import dask.array as da
import numpy as np

from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps


class TransformProtocol(Protocol):
    """Protocol for a generic transform."""

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """A transformation to be applied after loading a numpy array."""
        ...

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """A transformation to be applied after loading a dask array."""
        ...

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """A transformation to be applied before writing a numpy array."""
        ...

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """A transformation to be applied before writing a dask array."""
        ...


def get_as_numpy_transform(
    array: np.ndarray,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    """Apply a numpy transform to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.get_as_numpy_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array


def get_as_dask_transform(
    array: da.Array,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> da.Array:
    """Apply a dask transform to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.get_as_dask_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array


def set_as_numpy_transform(
    array: np.ndarray,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    """Apply inverse numpy transforms to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.set_as_numpy_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array


def set_as_dask_transform(
    array: da.Array,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> da.Array:
    """Apply inverse dask transforms to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.set_as_dask_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array
