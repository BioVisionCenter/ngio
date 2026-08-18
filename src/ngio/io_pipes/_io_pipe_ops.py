"""Context construction and the read/write chains shared by all io pipes.

This module sits below the pipe classes in the import graph so that
transforms needing to read data (e.g. the mask transform) can run the chains
without importing the pipes.
"""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._merge_policy import MergePolicy
from ngio.io_pipes._ops_axes import (
    build_axes_ops,
    get_as_dask_axes_ops,
    get_as_numpy_axes_ops,
    set_as_dask_axes_ops,
    set_as_numpy_axes_ops,
)
from ngio.io_pipes._ops_slices import (
    SlicingInputType,
    build_slicing_ops,
    get_slice_as_dask,
    get_slice_as_numpy,
    set_slice_as_dask,
    set_slice_as_numpy,
)
from ngio.io_pipes._ops_transforms import (
    IoPipeContext,
    TransformProtocol,
    apply_get_transforms,
    apply_set_transforms,
)
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValueError

ArrayT = TypeVar("ArrayT", np.ndarray, DaskArray)


def _merged(
    merge: MergePolicy,
    existing: ArrayT,
    patch: ArrayT,
    ctx: IoPipeContext,
    expected_type: type,
) -> ArrayT:
    """Apply the merge policy and verify it stayed on the same backend."""
    merged = cast("ArrayT", merge.reconcile(existing, patch, ctx))
    if not isinstance(merged, expected_type):
        raise NgioValueError(
            f"Merge policy '{type(merge).__name__}' returned a "
            f"{type(merged).__name__} instead of a {expected_type.__name__} — "
            "it does not support this data path (e.g. a numpy-only policy used "
            "through the dask API, or one materializing a dask array)."
        )
    return merged


def roi_to_slicing_dict(
    *,
    roi: Roi,
    pixel_size: PixelSize,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> dict[str, SlicingInputType]:
    """Convert a ROI to a slicing dictionary."""
    roi_slicing_dict: dict[str, SlicingInputType] = roi.to_slicing_dict(
        pixel_size=pixel_size
    )  # type: ignore
    if slicing_dict is None:
        return roi_slicing_dict

    # Additional slice kwargs can be provided
    # and will override the ones from the ROI
    roi_slicing_dict.update(slicing_dict)
    return roi_slicing_dict


def setup_io_pipe(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    axes_order: Sequence[str] | None = None,
    remove_channel_selection: bool = False,
    roi: Roi | None = None,
) -> IoPipeContext:
    """Build the context for an io pipe.

    When a `roi` is given it is converted to a slicing dict; explicit
    `slicing_dict` entries override the ROI-derived ones per axis.
    """
    if roi is not None:
        slicing_dict = roi_to_slicing_dict(
            roi=roi, pixel_size=dimensions.pixel_size, slicing_dict=slicing_dict
        )
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
    return IoPipeContext(
        zarr_array=zarr_array, slicing=slicing_ops, axes_ops=axes_ops, roi=roi
    )


def read_as_numpy(
    ctx: IoPipeContext, transforms: Sequence[TransformProtocol] | None
) -> np.ndarray:
    """Run the read chain: slice, axes ops, then get-transforms in order."""
    array = get_slice_as_numpy(ctx.zarr_array, slicing_ops=ctx.slicing)
    array = get_as_numpy_axes_ops(array, axes_ops=ctx.axes_ops)
    return apply_get_transforms(
        array, ctx=ctx, transforms=transforms, expected_type=np.ndarray
    )


def read_as_dask(
    ctx: IoPipeContext, transforms: Sequence[TransformProtocol] | None
) -> DaskArray:
    """Run the read chain: slice, axes ops, then get-transforms in order."""
    array = get_slice_as_dask(ctx.zarr_array, slicing_ops=ctx.slicing)
    array = get_as_dask_axes_ops(array, axes_ops=ctx.axes_ops)
    return apply_get_transforms(
        array, ctx=ctx, transforms=transforms, expected_type=DaskArray
    )


def write_from_numpy(
    ctx: IoPipeContext,
    transforms: Sequence[TransformProtocol] | None,
    patch: np.ndarray,
    merge: MergePolicy | None = None,
) -> None:
    """Run the write chain: set-transforms in reverse, merge, axes ops, write.

    The merge runs after the transforms so both it and the destination read are
    in the array's own space; `read_as_numpy(ctx, None)` deliberately replays no
    chain, and covers exactly the region about to be written.
    """
    patch = apply_set_transforms(
        patch, ctx=ctx, transforms=transforms, expected_type=np.ndarray
    )
    if merge is not None:
        existing = read_as_numpy(ctx, None)
        patch = _merged(merge, existing, patch, ctx, expected_type=np.ndarray)
    patch = set_as_numpy_axes_ops(array=patch, axes_ops=ctx.axes_ops)
    set_slice_as_numpy(zarr_array=ctx.zarr_array, patch=patch, slicing_ops=ctx.slicing)


def write_from_dask(
    ctx: IoPipeContext,
    transforms: Sequence[TransformProtocol] | None,
    patch: DaskArray,
    merge: MergePolicy | None = None,
) -> None:
    """Run the write chain: set-transforms in reverse, merge, axes ops, write."""
    patch = apply_set_transforms(
        patch, ctx=ctx, transforms=transforms, expected_type=DaskArray
    )
    if merge is not None:
        existing = read_as_dask(ctx, None)
        patch = _merged(merge, existing, patch, ctx, expected_type=DaskArray)
    patch = set_as_dask_axes_ops(array=patch, axes_ops=ctx.axes_ops)
    set_slice_as_dask(zarr_array=ctx.zarr_array, patch=patch, slicing_ops=ctx.slicing)
