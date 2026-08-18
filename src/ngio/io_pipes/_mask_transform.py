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
from ngio.io_pipes._ops_transforms import TransformProtocol, normalize_transforms
from ngio.io_pipes._rmw_transform import ReadModifyWriteTransform
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


class BaseMaskTransform(ReadModifyWriteTransform):
    """Mask the data flowing through an io pipe with a label image.

    On read, pixels outside the mask are replaced with `fill_value`; on
    write, they are protected by merging the patch with the on-disk data.
    The mask is selected by the pipe's ROI: `label_data == roi.label`, or
    `label_data != 0` for an unlabelled ROI.

    Must be the last transform of its pipe, and the only read-modify-write
    one — see `ngio.io_pipes._rmw_transform`. The pipes enforce both at
    construction.
    """

    def __init__(
        self,
        *,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        label_transforms: Sequence[TransformProtocol] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        axes_order: Sequence[str] | None = None,
        fill_value: int | float = 0,
        allow_rescaling: bool = True,
        target_dimensions: Dimensions | None = None,
        set_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Build a mask transform reading its mask from a label zarr array.

        Args:
            label_zarr_array: The label zarr array holding the mask.
            label_dimensions: The label's dimensions; the pipe's ROI is
                sliced at the label's own pixel size.
            label_transforms: Extra transforms applied to the label read.
            label_slicing_dict: Per-axis overrides for the label slicing.
            axes_order: The axes order of the data pipe, so the label is
                read in matching orientation.
            fill_value: Value for outside-mask pixels on read. Writes ignore
                it and keep the on-disk data instead.
            allow_rescaling: Zoom the label to the data grid (nearest) when
                the two live at different pyramid levels.
            target_dimensions: The data's dimensions, required to rescale.
            set_transforms: Override for the chain replayed on the write
                path's read-back; the pipe binds it for you.
        """
        super().__init__(set_transforms=set_transforms)
        if allow_rescaling:
            if target_dimensions is None:
                raise NgioValueError(
                    "MaskTransform needs the data's dimensions to rescale the "
                    "label; pass target_dimensions (or allow_rescaling=False)."
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
        self._fill_value = fill_value

    def _label_ctx(self, ctx: IoPipeContext) -> IoPipeContext:
        if ctx.roi is None:
            raise NgioValueError(
                "MaskTransform requires a ROI-scoped pipe; read or write "
                "through a roi method (or pass roi= to the pipe)."
            )
        return setup_io_pipe(
            zarr_array=self._label_zarr_array,
            dimensions=self._label_dimensions,
            slicing_dict=self._label_slicing_dict,
            axes_order=self._axes_order,
            remove_channel_selection=True,
            roi=ctx.roi,
        )

    def _bool_mask(
        self,
        ctx: IoPipeContext,
        label_ctx: IoPipeContext,
        label_data: ArrayT,
        data_shape: tuple[int, ...],
    ) -> ArrayT:
        assert ctx.roi is not None
        return _label_to_bool_mask(
            label_data=label_data,
            label=ctx.roi.label,
            data_shape=data_shape,
            label_axes=label_ctx.axes_ops.output_axes,
            data_axes=ctx.axes_ops.output_axes,
        )

    def on_get(
        self, array: np.ndarray | DaskArray, ctx: IoPipeContext
    ) -> np.ndarray | DaskArray:
        """Replace outside-mask pixels with the fill value."""
        label_ctx = self._label_ctx(ctx)
        if isinstance(array, DaskArray):
            label_data = read_as_dask(label_ctx, self._label_transforms)
            data_shape = tuple(int(dim) for dim in array.shape)
            bool_mask = self._bool_mask(ctx, label_ctx, label_data, data_shape)
            if bool_mask.shape != array.shape:
                bool_mask = da.broadcast_to(bool_mask, array.shape)
            return da.where(bool_mask, array, self._fill_value)

        label_data = read_as_numpy(label_ctx, self._label_transforms)
        bool_mask = self._bool_mask(ctx, label_ctx, label_data, array.shape)
        if bool_mask.shape != array.shape:
            bool_mask = np.broadcast_to(bool_mask, array.shape)
        return np.where(bool_mask, array, self._fill_value)

    def reconcile(
        self,
        existing: np.ndarray | DaskArray,
        patch: np.ndarray | DaskArray,
        ctx: IoPipeContext,
    ) -> np.ndarray | DaskArray:
        """Take the patch inside the mask, the on-disk data outside it."""
        label_ctx = self._label_ctx(ctx)
        if isinstance(patch, DaskArray):
            label_data = read_as_dask(label_ctx, self._label_transforms)
            data_shape = tuple(int(dim) for dim in existing.shape)
            bool_mask = self._bool_mask(ctx, label_ctx, label_data, data_shape)
            if bool_mask.shape != existing.shape:
                bool_mask = da.broadcast_to(bool_mask, existing.shape)
            return da.where(bool_mask, patch, existing)

        existing_np = cast("np.ndarray", existing)
        label_data = read_as_numpy(label_ctx, self._label_transforms)
        bool_mask = self._bool_mask(ctx, label_ctx, label_data, existing_np.shape)
        if bool_mask.shape != existing_np.shape:
            bool_mask = np.broadcast_to(bool_mask, existing_np.shape)
        return np.where(bool_mask, cast("np.ndarray", patch), existing_np)
