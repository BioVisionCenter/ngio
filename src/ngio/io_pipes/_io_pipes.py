from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._ops_axes import (
    AxesOps,
    build_axes_ops,
    get_as_dask_axes_ops,
    get_as_numpy_axes_ops,
    set_as_dask_axes_ops,
    set_as_numpy_axes_ops,
)
from ngio.io_pipes._ops_slices import (
    SlicingInputType,
    SlicingOps,
    build_slicing_ops,
    get_slice_as_dask,
    get_slice_as_numpy,
    set_slice_as_dask,
    set_slice_as_numpy,
)
from ngio.io_pipes._ops_transforms import (
    TransformProtocol,
    get_as_dask_transform,
    get_as_numpy_transform,
    set_as_dask_transform,
    set_as_numpy_transform,
)


def setup_io_pipe(
    *,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    axes_order: Sequence[str] | None = None,
    remove_channel_selection: bool = False,
) -> tuple[SlicingOps, AxesOps]:
    """Setup the slicing tuple and axes ops for an IO pipe."""
    slicing_ops = build_slicing_ops(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    axes_ops = build_axes_ops(
        dimensions=dimensions,
        input_axes=slicing_ops.slice_axes,
        axes_order=axes_order,
    )
    return slicing_ops, axes_ops


##############################################################
#
# "From Disk" Pipes
#
##############################################################

T = TypeVar("T")


class DataGetter(ABC, Generic[T]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        slicing_ops: SlicingOps,
        axes_ops: AxesOps,
        transforms: Sequence[TransformProtocol] | None = None,
        roi: Roi | None = None,
    ) -> None:
        self._zarr_array = zarr_array
        self._slicing_ops = slicing_ops
        self._axes_ops = axes_ops
        self._transforms = transforms
        self._roi = roi

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(zarr_array={self._zarr_array}, "
            f"slicing_ops={self._slicing_ops}, "
            f"axes_ops={self._axes_ops}, "
            f"transforms={self._transforms})"
        )

    @property
    def zarr_array(self) -> zarr.Array:
        return self._zarr_array

    @property
    def slicing_ops(self) -> SlicingOps:
        return self._slicing_ops

    @property
    def axes_ops(self) -> AxesOps:
        return self._axes_ops

    @property
    def transforms(self) -> Sequence[TransformProtocol] | None:
        return self._transforms

    @property
    def roi(self) -> Roi:
        if self._roi is None:
            name = self.__class__.__name__
            raise ValueError(f"No ROI defined for {name}.")
        return self._roi

    def __call__(self) -> T:
        return self.get()

    @abstractmethod
    def get(self) -> T:
        pass


class DataSetter(ABC, Generic[T]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        slicing_ops: SlicingOps,
        axes_ops: AxesOps,
        transforms: Sequence[TransformProtocol] | None = None,
        roi: Roi | None = None,
    ) -> None:
        self._zarr_array = zarr_array
        self._slicing_ops = slicing_ops
        self._axes_ops = axes_ops
        self._transforms = transforms
        self._roi = roi

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(zarr_array={self._zarr_array}, "
            f"slicing_ops={self._slicing_ops}, "
            f"axes_ops={self._axes_ops}, "
            f"transforms={self._transforms})"
        )

    @property
    def zarr_array(self) -> zarr.Array:
        return self._zarr_array

    @property
    def slicing_ops(self) -> SlicingOps:
        return self._slicing_ops

    @property
    def axes_ops(self) -> AxesOps:
        return self._axes_ops

    @property
    def transforms(self) -> Sequence[TransformProtocol] | None:
        return self._transforms

    @property
    def roi(self) -> Roi:
        if self._roi is None:
            name = self.__class__.__name__
            raise ValueError(f"No ROI defined for {name}.")
        return self._roi

    def __call__(self, patch: T) -> None:
        return self.set(patch)

    @abstractmethod
    def set(self, patch: T) -> None:
        pass


class NumpyGetter(DataGetter[np.ndarray]):
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
        """Build a pipe to get a numpy or dask array from a zarr array."""
        slicing_ops, axes_ops = setup_io_pipe(
            dimensions=dimensions,
            slicing_dict=slicing_dict,
            axes_order=axes_order,
            remove_channel_selection=remove_channel_selection,
        )
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=slicing_ops,
            axes_ops=axes_ops,
            transforms=transforms,
            roi=roi,
        )

    def get(self) -> np.ndarray:
        """Get a numpy array from the zarr array with ops."""
        array = get_slice_as_numpy(self._zarr_array, slicing_ops=self._slicing_ops)
        array = get_as_numpy_axes_ops(array, axes_ops=self._axes_ops)
        array = get_as_numpy_transform(
            array,
            slicing_ops=self._slicing_ops,
            axes_ops=self._axes_ops,
            transforms=self._transforms,
        )
        return array


class DaskGetter(DataGetter[DaskArray]):
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
        """Build a pipe to get a numpy or dask array from a zarr array."""
        slicing_ops, axes_ops = setup_io_pipe(
            dimensions=dimensions,
            slicing_dict=slicing_dict,
            axes_order=axes_order,
            remove_channel_selection=remove_channel_selection,
        )
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=slicing_ops,
            axes_ops=axes_ops,
            transforms=transforms,
            roi=roi,
        )

    def get(self) -> DaskArray:
        """Get a dask array from the zarr array with ops.

        The order of operations is:
        * get slice will load the data from the zarr array
        * get axes ops will reorder, squeeze or expand the axes
        * get transform will apply any additional transforms

        """
        array = get_slice_as_dask(self._zarr_array, slicing_ops=self._slicing_ops)
        array = get_as_dask_axes_ops(array, axes_ops=self._axes_ops)
        array = get_as_dask_transform(
            array,
            slicing_ops=self._slicing_ops,
            axes_ops=self._axes_ops,
            transforms=self._transforms,
        )
        return array


##############################################################
#
# "To Disk" Pipes
#
##############################################################


class NumpySetter(DataSetter[np.ndarray]):
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
        """Build a pipe to get a numpy or dask array from a zarr array."""
        slicing_ops, axes_ops = setup_io_pipe(
            dimensions=dimensions,
            slicing_dict=slicing_dict,
            axes_order=axes_order,
            remove_channel_selection=remove_channel_selection,
        )
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=slicing_ops,
            axes_ops=axes_ops,
            transforms=transforms,
            roi=roi,
        )

    def set(self, patch: np.ndarray) -> None:
        """Get a numpy array from the zarr array with ops."""
        patch = set_as_numpy_transform(
            array=patch,
            slicing_ops=self._slicing_ops,
            axes_ops=self._axes_ops,
            transforms=self._transforms,
        )
        patch = set_as_numpy_axes_ops(
            array=patch,
            axes_ops=self._axes_ops,
        )
        set_slice_as_numpy(
            zarr_array=self._zarr_array,
            patch=patch,
            slicing_ops=self._slicing_ops,
        )


class DaskSetter(DataSetter[DaskArray]):
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
        """Build a pipe to get a numpy or dask array from a zarr array."""
        slicing_ops, axes_ops = setup_io_pipe(
            dimensions=dimensions,
            slicing_dict=slicing_dict,
            axes_order=axes_order,
            remove_channel_selection=remove_channel_selection,
        )
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=slicing_ops,
            axes_ops=axes_ops,
            transforms=transforms,
            roi=roi,
        )

    def set(self, patch: DaskArray) -> None:
        """Get a dask array from the zarr array with ops."""
        patch = set_as_dask_transform(
            array=patch,
            slicing_ops=self._slicing_ops,
            axes_ops=self._axes_ops,
            transforms=self._transforms,
        )
        patch = set_as_dask_axes_ops(
            array=patch,
            axes_ops=self._axes_ops,
        )
        set_slice_as_dask(
            zarr_array=self._zarr_array,
            patch=patch,
            slicing_ops=self._slicing_ops,
        )
