from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable

import dask.array as da
import numpy as np

from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.utils import NgioValueError


@runtime_checkable
class NumpyTransformProtocol(Protocol):
    """Protocol for a transform on the numpy data path."""

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """A transformation to be applied after loading a numpy array."""
        ...

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """A transformation to be applied before writing a numpy array."""
        ...


@runtime_checkable
class DaskTransformProtocol(Protocol):
    """Protocol for a transform on the dask data path."""

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """A transformation to be applied after loading a dask array."""
        ...

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """A transformation to be applied before writing a dask array."""
        ...


TransformProtocol: TypeAlias = NumpyTransformProtocol | DaskTransformProtocol
"""A transform for either data path; each pipe requires the protocol it uses."""


def require_numpy_transforms(
    transforms: Sequence[TransformProtocol] | None,
) -> Sequence[NumpyTransformProtocol] | None:
    """Validate that every transform implements the numpy transform protocol.

    Returns:
        The same transforms, narrowed to the numpy protocol.

    Raises:
        NgioValueError: If a transform lacks the numpy transform methods.
    """
    if transforms is None:
        return None
    for transform in transforms:
        if not isinstance(transform, NumpyTransformProtocol):
            raise NgioValueError(
                f"Transform '{type(transform).__name__}' does not implement the "
                "numpy transform protocol (get_as_numpy_transform / "
                "set_as_numpy_transform), which numpy pipes require. "
                "Dask-only transforms work with the dask API "
                "(e.g. get_as_dask / iter_as_dask)."
            )
    return [t for t in transforms if isinstance(t, NumpyTransformProtocol)]


def require_dask_transforms(
    transforms: Sequence[TransformProtocol] | None,
) -> Sequence[DaskTransformProtocol] | None:
    """Validate that every transform implements the dask transform protocol.

    Returns:
        The same transforms, narrowed to the dask protocol.

    Raises:
        NgioValueError: If a transform lacks the dask transform methods.
    """
    if transforms is None:
        return None
    for transform in transforms:
        if not isinstance(transform, DaskTransformProtocol):
            raise NgioValueError(
                f"Transform '{type(transform).__name__}' does not implement the "
                "dask transform protocol (get_as_dask_transform / "
                "set_as_dask_transform), which dask pipes require. "
                "Numpy-only transforms work with the numpy API "
                "(e.g. get_as_numpy / iter_as_numpy)."
            )
    return [t for t in transforms if isinstance(t, DaskTransformProtocol)]


def get_as_numpy_transform(
    array: np.ndarray,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[NumpyTransformProtocol] | None = None,
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
    transforms: Sequence[DaskTransformProtocol] | None = None,
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
    transforms: Sequence[NumpyTransformProtocol] | None = None,
) -> np.ndarray:
    """Apply inverse numpy transforms to an array."""
    if transforms is None:
        return array

    # Reading applies [A, B] as B(A(x)), so writing must invert the chain
    # outermost-first: A_inv(B_inv(y)).
    for transform in reversed(transforms):
        array = transform.set_as_numpy_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array


def set_as_dask_transform(
    array: da.Array,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[DaskTransformProtocol] | None = None,
) -> da.Array:
    """Apply inverse dask transforms to an array."""
    if transforms is None:
        return array

    # Reading applies [A, B] as B(A(x)), so writing must invert the chain
    # outermost-first: A_inv(B_inv(y)).
    for transform in reversed(transforms):
        array = transform.set_as_dask_transform(
            array, slicing_ops=slicing_ops, axes_ops=axes_ops
        )
    return array
