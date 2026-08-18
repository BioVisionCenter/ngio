from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._io_pipe_ops import (
    read_as_dask,
    read_as_numpy,
    setup_io_pipe,
    write_from_dask,
    write_from_numpy,
)
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingInputType, SlicingOps
from ngio.io_pipes._ops_transforms import (
    IoPipeContext,
    TransformProtocol,
    normalize_transforms,
)

T = TypeVar("T")


class _IoPipe:
    """State shared by every io pipe: one context plus the transform chain."""

    def __init__(
        self,
        zarr_array: zarr.Array,
        slicing_ops: SlicingOps,
        axes_ops: AxesOps,
        transforms: Sequence[TransformProtocol] | None = None,
        roi: Roi | None = None,
    ) -> None:
        self._ctx = IoPipeContext(
            zarr_array=zarr_array, slicing=slicing_ops, axes_ops=axes_ops, roi=roi
        )
        self._transforms = normalize_transforms(transforms)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(zarr_array={self._ctx.zarr_array}, "
            f"slicing_ops={self._ctx.slicing}, "
            f"axes_ops={self._ctx.axes_ops}, "
            f"transforms={self._transforms})"
        )

    @property
    def zarr_array(self) -> zarr.Array:
        return self._ctx.zarr_array

    @property
    def slicing_ops(self) -> SlicingOps:
        return self._ctx.slicing

    @property
    def axes_ops(self) -> AxesOps:
        return self._ctx.axes_ops

    @property
    def transforms(self) -> Sequence[TransformProtocol] | None:
        return self._transforms

    @property
    def roi(self) -> Roi:
        if self._ctx.roi is None:
            name = self.__class__.__name__
            raise ValueError(f"No ROI defined for {name}.")
        return self._ctx.roi


class DataGetter(_IoPipe, ABC, Generic[T]):
    def __call__(self) -> T:
        return self.get()

    @abstractmethod
    def get(self) -> T:
        pass


class DataSetter(_IoPipe, ABC, Generic[T]):
    def __call__(self, patch: T) -> None:
        return self.set(patch)

    @abstractmethod
    def set(self, patch: T) -> None:
        pass


class _FromDimensionsInit(_IoPipe):
    """The concrete pipes' shared constructor: dimensions in, context out."""

    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
        roi: Roi | None = None,
    ) -> None:
        """Build a pipe to read or write a slice of a zarr array.

        When a `roi` is given, it defines the slicing (converted at this
        image's pixel size); explicit `slicing_dict` entries override the
        ROI-derived ones per axis.
        """
        ctx = setup_io_pipe(
            zarr_array=zarr_array,
            dimensions=dimensions,
            slicing_dict=slicing_dict,
            axes_order=axes_order,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )
        super().__init__(
            zarr_array=ctx.zarr_array,
            slicing_ops=ctx.slicing,
            axes_ops=ctx.axes_ops,
            transforms=transforms,
            roi=roi,
        )


class NumpyGetter(_FromDimensionsInit, DataGetter[np.ndarray]):
    def get(self) -> np.ndarray:
        """Get a numpy array from the zarr array with ops."""
        return read_as_numpy(self._ctx, self._transforms)


class DaskGetter(_FromDimensionsInit, DataGetter[DaskArray]):
    def get(self) -> DaskArray:
        """Get a dask array from the zarr array with ops."""
        return read_as_dask(self._ctx, self._transforms)


class NumpySetter(_FromDimensionsInit, DataSetter[np.ndarray]):
    def set(self, patch: np.ndarray) -> None:
        """Write a numpy array to the zarr array with ops."""
        write_from_numpy(self._ctx, self._transforms, patch)


class DaskSetter(_FromDimensionsInit, DataSetter[DaskArray]):
    def set(self, patch: DaskArray) -> None:
        """Write a dask array to the zarr array with ops."""
        write_from_dask(self._ctx, self._transforms, patch)
