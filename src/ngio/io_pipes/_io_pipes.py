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
from ngio.io_pipes._merge_policy import MergeInput, MergePolicy, resolve_merge
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingInputType, SlicingOps
from ngio.io_pipes._ops_transforms import (
    IoPipeContext,
    TransformProtocol,
    normalize_transforms,
)
from ngio.utils import NgioValueError

T = TypeVar("T")


def _prepare_transforms(
    transforms: Sequence[TransformProtocol] | None,
) -> Sequence[TransformProtocol] | None:
    """Normalize the chain, refusing anything that is not a transform.

    A merge policy depends on the destination's contents and belongs in the
    pipe's `merge=` slot (see `ngio.transforms` for the split); catching it
    here replaces a confusing "not a transform" failure. Both protocols are
    `runtime_checkable`, so `isinstance` only probes attribute names — an
    object exposing `on_get`/`on_set` *and* `reconcile` is taken at its
    placement here and treated as a transform.
    """
    if transforms:
        policies = [
            t
            for t in transforms
            if isinstance(t, MergePolicy) and not isinstance(t, TransformProtocol)
        ]
        if policies:
            names = ", ".join(type(t).__name__ for t in policies)
            raise NgioValueError(
                f"{names} is a merge policy, not a transform: it combines the "
                "patch with what is already on disk, which the transform chain "
                "has no place for. Pass it as `merge=` on the write instead of "
                "in `transforms=`."
            )
    return normalize_transforms(transforms)


class _IoPipe:
    """State shared by every io pipe: one context plus the transform chain."""

    def __init__(
        self,
        zarr_array: zarr.Array,
        slicing_ops: SlicingOps,
        axes_ops: AxesOps,
        transforms: Sequence[TransformProtocol] | None = None,
        roi: Roi | None = None,
        merge: MergeInput | None = None,
    ) -> None:
        self._ctx = IoPipeContext(
            zarr_array=zarr_array, slicing=slicing_ops, axes_ops=axes_ops, roi=roi
        )
        self._transforms = _prepare_transforms(transforms)
        self._merge = resolve_merge(merge)
        if self._merge is not None and isinstance(self, DataGetter):
            raise NgioValueError(
                f"{self.__class__.__name__} is a getter: a merge decides how a "
                "write combines with the destination, so it would be silently "
                "ignored here. Pass `merge=` on the setter instead."
            )

    def __repr__(self) -> str:
        name = self.__class__.__name__
        merge = f", merge={self._merge!r}" if self._merge is not None else ""
        return (
            f"{name}(zarr_array={self._ctx.zarr_array}, "
            f"slicing_ops={self._ctx.slicing}, "
            f"axes_ops={self._ctx.axes_ops}, "
            f"transforms={self._transforms}{merge})"
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
    def merge(self) -> MergePolicy | None:
        """How writes combine with the destination, or `None` to overwrite."""
        return self._merge

    @property
    def roi(self) -> Roi:
        if self._ctx.roi is None:
            name = self.__class__.__name__
            raise NgioValueError(f"No ROI defined for {name}.")
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
        merge: MergeInput | None = None,
    ) -> None:
        """Build a pipe to read or write a slice of a zarr array.

        When a `roi` is given, it defines the slicing (converted at this
        image's pixel size); explicit `slicing_dict` entries override the
        ROI-derived ones per axis, and such an override drops the pipe's
        `roi` (see `setup_io_pipe`). `merge` applies to setters only and
        decides how the patch combines with what is already there.
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
            roi=ctx.roi,
            merge=merge,
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
        write_from_numpy(self._ctx, self._transforms, patch, self._merge)


class DaskSetter(_FromDimensionsInit, DataSetter[DaskArray]):
    def set(self, patch: DaskArray) -> None:
        """Write a dask array to the zarr array with ops."""
        write_from_dask(self._ctx, self._transforms, patch, self._merge)
