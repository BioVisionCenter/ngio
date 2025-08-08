from collections.abc import Sequence
from typing import Protocol

import dask.array as da
import numpy as np

from ngio.ome_zarr_meta.ngio_specs import SlicingOps


class AbstractTransform:
    """Abstract base class for a generic transform."""

    def transform_numpy(self, array: np.ndarray, slicing_ops: SlicingOps) -> np.ndarray:
        """Transform a numpy array."""
        raise NotImplementedError("Subclasses should implement this method.")

    def transform_dask(self, array: da.Array, slicing_ops: SlicingOps) -> da.Array:
        """Transform a dask array."""
        raise NotImplementedError("Subclasses should implement this method.")


class TransformProtocol(Protocol):
    """Protocol numpy, dask, or delayed array transforms."""

    def transform_numpy(self, array: np.ndarray, slicing_ops: SlicingOps) -> np.ndarray:
        """Transform a numpy array."""
        ...

    def transform_dask(self, array: da.Array, slicing_ops: SlicingOps) -> da.Array:
        """Transform a dask array."""
        ...


def apply_numpy_transforms(
    array: np.ndarray,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    """Apply a numpy transform to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.transform_numpy(array, slicing_ops=slicing_ops)
    return array


def apply_dask_transforms(
    array: da.Array,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> da.Array:
    """Apply a dask transform to an array."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.transform_dask(array, slicing_ops=slicing_ops)
    return array
