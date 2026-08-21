"""Masking, in its two separate roles.

Masking looks like one idea but is two operations. On the read path it is a
function of the patch alone — replace outside-mask pixels with a fill value —
which is an ordinary transform. On the write path it is a function of the patch
*and* the destination — keep the on-disk pixels outside the mask — which is a
merge, and belongs in the pipe's `merge=` slot rather than in the transform
chain.

They share only the question "which pixels does the mask select here", which
`_MaskSelection` answers for both.
"""

from collections.abc import Sequence
from typing import TypeVar, cast

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.io_pipes._io_pipe_ops import (
    IoPipeContext,
    read_as_dask,
    read_as_numpy,
    setup_io_pipe,
)
from ngio.io_pipes._match_shape import dask_match_shape, numpy_match_shape
from ngio.io_pipes._ops_slices import SlicingInputType
from ngio.io_pipes._ops_transforms import (
    ArrayLike,
    TransformProtocol,
    elementwise,
    normalize_transforms,
)
from ngio.io_pipes._zoom_transform import BaseZoomTransform
from ngio.utils import NgioValueError

ArrayT = TypeVar("ArrayT", np.ndarray, DaskArray)


def _label_to_bool_mask(
    label_data: ArrayT,
    label: int | None,
    data_shape: tuple[int, ...],
    label_axes: tuple[str, ...],
    data_axes: tuple[str, ...],
) -> ArrayT:
    """Convert label data to a boolean mask matching the data shape."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    if isinstance(bool_mask, DaskArray):
        matched = dask_match_shape(
            array=bool_mask,
            reference_shape=data_shape,
            array_axes=label_axes,
            reference_axes=data_axes,
        )
    else:
        matched = numpy_match_shape(
            array=bool_mask,
            reference_shape=data_shape,
            array_axes=label_axes,
            reference_axes=data_axes,
        )
    return cast("ArrayT", matched)


class _MaskSelection:
    """Reads a label image and says which pixels its mask selects.

    Shared by the read-side transform and the write-side merge; the mask is
    chosen by the pipe's ROI — `label == roi.label`, or `label != 0` for an
    unlabelled ROI — so one instance serves every ROI.
    """

    def __init__(
        self,
        *,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        label_transforms: Sequence[TransformProtocol] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        axes_order: Sequence[str] | None = None,
        allow_rescaling: bool = True,
        target_dimensions: Dimensions | None = None,
    ) -> None:
        """Set up the label read.

        Args:
            label_zarr_array: The label zarr array holding the mask.
            label_dimensions: The label's dimensions; the ROI is sliced at the
                label's own pixel size.
            label_transforms: Extra transforms applied to the label read.
            label_slicing_dict: Per-axis overrides for the label slicing.
            axes_order: The axes order of the data pipe, so the label is read
                in matching orientation.
            allow_rescaling: Zoom the label to the data grid (nearest) when the
                two live at different pyramid levels. This rescales the *label*
                only; the data array is never reshaped.
            target_dimensions: The data's dimensions, required to rescale.
        """
        if allow_rescaling:
            if target_dimensions is None:
                raise NgioValueError(
                    "Masking needs the data's dimensions to rescale the label; "
                    "pass target_dimensions (or allow_rescaling=False)."
                )
            zoom = BaseZoomTransform(
                input_dimensions=label_dimensions,
                target_dimensions=target_dimensions,
                order="nearest",
            )
            if label_transforms is None or len(label_transforms) == 0:
                label_transforms = [zoom]
            else:
                label_transforms = [zoom, *label_transforms]

        self._label_zarr_array = label_zarr_array
        self._label_dimensions = label_dimensions
        self._label_transforms = normalize_transforms(label_transforms)
        self._label_slicing_dict = label_slicing_dict
        self._axes_order = axes_order

    def _label_ctx(self, ctx: IoPipeContext) -> IoPipeContext:
        if ctx.roi is None:
            raise NgioValueError(
                "Masking requires a ROI-scoped pipe; read or write through a "
                "roi method (or pass roi= to the pipe)."
            )
        return setup_io_pipe(
            zarr_array=self._label_zarr_array,
            dimensions=self._label_dimensions,
            slicing_dict=self._label_slicing_dict,
            axes_order=self._axes_order,
            remove_channel_selection=True,
            roi=ctx.roi,
        )

    def mask_for(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """The boolean mask for `ctx`'s ROI, broadcast to `array`'s shape."""
        label_ctx = self._label_ctx(ctx)
        assert ctx.roi is not None
        if isinstance(array, DaskArray):
            label_data = read_as_dask(label_ctx, self._label_transforms)
            shape = tuple(int(dim) for dim in array.shape)
            bool_mask = _label_to_bool_mask(
                label_data=label_data,
                label=ctx.roi.label,
                data_shape=shape,
                label_axes=label_ctx.axes_ops.output_axes,
                data_axes=ctx.axes_ops.output_axes,
            )
            if bool_mask.shape != array.shape:
                bool_mask = da.broadcast_to(bool_mask, array.shape)
            return bool_mask

        label_data = read_as_numpy(label_ctx, self._label_transforms)
        bool_mask = _label_to_bool_mask(
            label_data=label_data,
            label=ctx.roi.label,
            data_shape=array.shape,
            label_axes=label_ctx.axes_ops.output_axes,
            data_axes=ctx.axes_ops.output_axes,
        )
        if bool_mask.shape != array.shape:
            bool_mask = np.broadcast_to(bool_mask, array.shape)
        return bool_mask


class BaseMaskTransform(_MaskSelection):
    """Read-side masking: outside-mask pixels come back as `fill_value`.

    A get-only transform. Filling has no inverse — the values it replaced are
    gone — so there is nothing to apply on the write path, and write-protecting
    the pixels outside the mask is a `merge=` policy instead of a transform.
    """

    def __init__(self, *, fill_value: int | float = 0, **kwargs) -> None:
        """Build the transform; `fill_value` is what outside-mask pixels read as."""
        super().__init__(**kwargs)
        self._fill_value = fill_value

    def on_get(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Replace outside-mask pixels with the fill value."""
        bool_mask = self.mask_for(array, ctx)
        if isinstance(array, DaskArray):
            return da.where(bool_mask, array, self._fill_value)
        return elementwise(np.where, bool_mask, array, self._fill_value)

    def on_set(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Refuse: masking a write is a merge, not a transform."""
        raise NgioValueError(
            "MaskTransform only applies to reads — filling outside-mask pixels "
            "has no inverse to run on a write. To protect the pixels outside "
            "the mask when writing, pass `merge=MaskMerge(...)` instead of "
            "putting the transform in `transforms=`."
        )


class BaseMaskMerge(_MaskSelection):
    """Write-side masking: keep the on-disk pixels outside the mask.

    A merge policy, so it runs after the transform chain against the array's
    own contents — the protected pixels are carried through byte-identically
    rather than round-tripped through a transform's inverse.
    """

    def reconcile(
        self, existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
    ) -> ArrayLike:
        """Take the patch inside the mask, the on-disk data outside it."""
        bool_mask = self.mask_for(existing, ctx)
        if isinstance(patch, DaskArray):
            return da.where(bool_mask, patch, existing)
        return elementwise(np.where, bool_mask, patch, existing)
