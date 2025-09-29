from collections.abc import Callable, Sequence
from typing import Literal, assert_never, overload

import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common import Dimensions
from ngio.io_pipes._io_pipes_utils import SlicingInputType, setup_io_pipe
from ngio.io_pipes._ops_axes import (
    get_as_dask_axes_ops,
    get_as_numpy_axes_ops,
    set_as_dask_axes_ops,
    set_as_numpy_axes_ops,
)
from ngio.io_pipes._ops_slices import (
    SlicingOps,
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
from ngio.ome_zarr_meta.ngio_specs._axes import AxesOps

##############################################################
#
# "From Disk" Pipes
#
##############################################################


def _get_as_numpy_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    """Get a numpy array from the zarr array with the given slice tuple and axes ops."""
    array = get_slice_as_numpy(zarr_array, slicing_ops=slicing_ops)
    array = get_as_numpy_axes_ops(
        array,
        axes_ops=axes_ops,
    )
    array = get_as_numpy_transform(
        array,
        slicing_ops=slicing_ops,
        axes_ops=axes_ops,
        transforms=transforms,
    )

    return array


def _get_as_dask_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> DaskArray:
    """Get a numpy array from the zarr array with the given slice tuple and axes ops."""
    array = get_slice_as_dask(zarr_array, slicing_ops=slicing_ops)
    array = get_as_dask_axes_ops(
        array,
        axes_ops=axes_ops,
    )
    array = get_as_dask_transform(
        array,
        slicing_ops=slicing_ops,
        axes_ops=axes_ops,
        transforms=transforms,
    )

    return array


@overload
def build_getter_pipe(
    *,
    mode: Literal["numpy"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]: ...


@overload
def build_getter_pipe(
    *,
    mode: Literal["dask"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], DaskArray]: ...


def build_getter_pipe(
    *,
    mode: Literal["numpy", "dask"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray] | Callable[[], DaskArray]:
    """Build a pipe to get a numpy or dask array from a zarr array."""
    slicing_ops, axes_ops = setup_io_pipe(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        axes_order=axes_order,
        remove_channel_selection=remove_channel_selection,
    )
    if mode == "numpy":

        def get_numpy_pipe() -> np.ndarray:
            """Closure to get a numpy array from the zarr array."""
            return _get_as_numpy_pipe(
                zarr_array=zarr_array,
                slicing_ops=slicing_ops,
                axes_ops=axes_ops,
                transforms=transforms,
            )

        return get_numpy_pipe

    elif mode == "dask":

        def get_dask_pipe() -> DaskArray:
            """Closure to get a dask array from the zarr array."""
            return _get_as_dask_pipe(
                zarr_array=zarr_array,
                slicing_ops=slicing_ops,
                axes_ops=axes_ops,
                transforms=transforms,
            )

        return get_dask_pipe
    assert_never(mode)


##############################################################
#
# "To Disk" Pipes
#
##############################################################


def _set_as_numpy_pipe(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> None:
    """Get a numpy array from the zarr array with the given slice tuple and axes ops."""
    patch = set_as_numpy_transform(
        array=patch,
        slicing_ops=slicing_ops,
        axes_ops=axes_ops,
        transforms=transforms,
    )
    patch = set_as_numpy_axes_ops(
        array=patch,
        axes_ops=axes_ops,
    )
    set_slice_as_numpy(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
    )


def _set_as_dask_pipe(
    zarr_array: zarr.Array,
    patch: DaskArray,
    slicing_ops: SlicingOps,
    axes_ops: AxesOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> None:
    """Get a numpy array from the zarr array with the given slice tuple and axes ops."""
    patch = set_as_dask_transform(
        array=patch,
        slicing_ops=slicing_ops,
        axes_ops=axes_ops,
        transforms=transforms,
    )
    patch = set_as_dask_axes_ops(
        array=patch,
        axes_ops=axes_ops,
    )
    set_slice_as_dask(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
    )


@overload
def build_setter_pipe(
    *,
    mode: Literal["numpy"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]: ...


@overload
def build_setter_pipe(
    *,
    mode: Literal["dask"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[DaskArray], None]: ...


def build_setter_pipe(
    *,
    mode: Literal["numpy", "dask"],
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None] | Callable[[DaskArray], None]:
    """Build a pipe to get a numpy or dask array from a zarr array."""
    slicing_ops, axes_ops = setup_io_pipe(
        dimensions=dimensions,
        slicing_dict=slicing_dict,
        axes_order=axes_order,
        remove_channel_selection=remove_channel_selection,
    )
    if mode == "numpy":

        def set_numpy_pipe(patch: np.ndarray) -> None:
            """Closure to set a numpy array into the zarr array."""
            return _set_as_numpy_pipe(
                zarr_array=zarr_array,
                patch=patch,
                slicing_ops=slicing_ops,
                axes_ops=axes_ops,
                transforms=transforms,
            )

        return set_numpy_pipe
    elif mode == "dask":

        def set_dask_pipe(patch: DaskArray) -> None:
            """Closure to set a dask array into the zarr array."""
            return _set_as_dask_pipe(
                zarr_array=zarr_array,
                patch=patch,
                slicing_ops=slicing_ops,
                axes_ops=axes_ops,
                transforms=transforms,
            )

        return set_dask_pipe
    assert_never(mode)
