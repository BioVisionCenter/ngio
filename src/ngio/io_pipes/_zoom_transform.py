"""Rescaling between two pixel grids, as a transform on the io pipes.

The forward and inverse shape paths deliberately disagree about slice
normalization — the interlocking invariants live as comments on
`_compute_zoom_shape` and `_compute_inverse_zoom_shape`.
"""

import math
from collections.abc import Sequence

import dask.array as da
import numpy as np

from ngio.common._dimensions import Dimensions
from ngio.common._zoom import (
    InterpolationOrder,
    dask_zoom,
    numpy_zoom,
)
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.io_pipes._ops_transforms import TransformContext
from ngio.utils import NgioValueError


class BaseZoomTransform:
    def __init__(
        self,
        input_dimensions: Dimensions,
        target_dimensions: Dimensions,
        order: InterpolationOrder = "nearest",
    ) -> None:
        self._input_dimensions = input_dimensions
        self._target_dimensions = target_dimensions
        self._input_pixel_size = input_dimensions.pixel_size
        self._target_pixel_size = target_dimensions.pixel_size
        self._order: InterpolationOrder = order

    def _normalize_shape(
        self, slice_: slice | int | list[int], scale: float, max_dim: int
    ) -> int:
        if isinstance(slice_, slice):
            # Clamped like the read: a slice reaching past either edge is cut
            # there by the boundary check, so deriving the target from the
            # overhang would zoom at a silently wrong factor. A slice entirely
            # past the edge is an empty read, never a negative extent.
            _start = max(0.0, slice_.start or 0)
            _start_int = math.floor(_start * scale)
            if slice_.stop is not None:
                _stop = slice_.stop * scale
                _stop = min(_stop, max_dim)
            else:
                _stop = max_dim
            _stop_int = math.ceil(_stop)
            target_shape = max(0, _stop_int - _start_int)

        elif isinstance(slice_, int):
            target_shape = 1
        elif isinstance(slice_, list):
            if scale != 1:
                # Three non-contiguous rows are not a region: stretching them
                # as if adjacent resamples across pixels that were never
                # neighbours, silently.
                raise NgioValueError(
                    "Cannot zoom a non-contiguous list selection "
                    f"({slice_!r}): the selected elements are not adjacent, "
                    "so no zoom factor applies to them. Select a contiguous "
                    "slice, or zoom before selecting."
                )
            target_shape = len(slice_)
        else:
            raise NgioValueError(f"Unsupported slice type: {type(slice_)}")
        return math.ceil(target_shape)

    def _compute_zoom_shape(
        self,
        array_shape: Sequence[int],
        axes: Sequence[str],
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        # Derives the target shape from the **raw, un-normalized** slice: the
        # raw slice carries the ROI's sub-pixel world bounds
        # (`slice(5.25, 10.5)` on a 4x-coarser grid), which is what makes the
        # zoomed patch land exactly on the target grid — normalizing first
        # rounds to *this* array's integer pixels and derives a wrong factor.
        # Raw does not mean unclamped: like the read, a negative start cuts at
        # 0 and the stop at the target extent, so a slice entirely past the
        # edge yields an empty patch, never a negative extent. No shape
        # validation here — the patch is whatever the read produced. A list
        # selection has no geometry a zoom factor applies to, so it is refused
        # on any scaled axis.
        if len(array_shape) != len(axes):
            raise NgioValueError(
                f"Array has {len(array_shape)} dimensions but the transform "
                f"declares {len(axes)} output axes "
                f"({axes})."
            )

        target_shape = []
        for shape, ax_name in zip(array_shape, axes, strict=True):
            ax_type = self._input_dimensions.axes_handler.get_axis(ax_name)
            if ax_type is None:
                # Unknown axis can only be a virtual axis
                # So we set it to 1
                target_shape.append(1)
                continue
            elif ax_type.axis_type == "channel":
                # Do not scale channel axis
                target_shape.append(shape)
                continue
            t_dim = self._target_dimensions.get(ax_name, default=1)
            in_pix = self._input_pixel_size.get(ax_name, default=1.0)
            t_pix = self._target_pixel_size.get(ax_name, default=1.0)
            # Raw on purpose: the un-normalized slice keeps the ROI's
            # sub-pixel world bounds, which is what makes the zoomed shape
            # land exactly on the target grid. Normalizing first would round
            # to this array's integer pixels and derive a wrong factor.
            slice_ = slicing_ops.get(ax_name, normalize=False)
            scale = in_pix / t_pix
            _target_shape = self._normalize_shape(
                slice_=slice_, scale=scale, max_dim=t_dim
            )
            target_shape.append(_target_shape)
        return tuple(target_shape)

    def _compute_inverse_zoom_shape(
        self,
        array_shape: Sequence[int],
        axes: Sequence[str],
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        # The opposite of `_compute_zoom_shape` on purpose: this normalizes,
        # because a write must match the integer on-disk region the setter
        # will actually cover. Only this path carries the ±1 guard —
        # sub-pixel bounds legitimately round the two paths one pixel apart.
        if len(array_shape) != len(axes):
            raise NgioValueError(
                f"Array has {len(array_shape)} dimensions but the transform "
                f"declares {len(axes)} output axes "
                f"({axes})."
            )

        target_shape = []
        for shape, ax_name in zip(array_shape, axes, strict=True):
            ax_type = self._input_dimensions.axes_handler.get_axis(ax_name)
            if ax_type is not None and ax_type.axis_type == "channel":
                # Do not scale channel axis
                target_shape.append(shape)
                continue
            in_dim = self._input_dimensions.get(ax_name, default=1)
            slice_ = slicing_ops.get(ax_name=ax_name, normalize=True)
            target_shape.append(
                self._normalize_shape(slice_=slice_, scale=1, max_dim=in_dim)
            )

        # The rescaling is based on the slice, so the input patch must be
        # roughly the size the slice implies.
        expected_shape = self._compute_zoom_shape(
            array_shape=target_shape, axes=axes, slicing_ops=slicing_ops
        )
        if any(
            abs(es - s) > 1 for es, s in zip(expected_shape, array_shape, strict=True)
        ):
            raise NgioValueError(
                f"Input array shape {array_shape} is not compatible with the expected "
                f"shape {expected_shape} based on the zoom transform.\n"
            )
        return tuple(target_shape)

    def _numpy_zoom(
        self, array: np.ndarray, target_shape: tuple[int, ...]
    ) -> np.ndarray:
        if array.shape == target_shape:
            return array
        if array.size == 0 or 0 in target_shape:
            # Nothing to resample either way; the zoom kernels divide by the
            # source shape and would blow up on the zero.
            return np.empty(target_shape, dtype=array.dtype)
        return numpy_zoom(
            source_array=array, target_shape=target_shape, order=self._order
        )

    def _dask_zoom(
        self,
        array: da.Array,
        array_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ) -> da.Array:
        if array_shape == target_shape:
            return array
        if 0 in array_shape or 0 in target_shape:
            return da.empty(target_shape, dtype=array.dtype)
        return dask_zoom(
            source_array=array, target_shape=target_shape, order=self._order
        )

    def on_get(
        self, array: np.ndarray | da.Array, ctx: TransformContext
    ) -> np.ndarray | da.Array:
        """Apply the scaling transformation after reading."""
        if isinstance(array, da.Array):
            array_shape = tuple(int(s) for s in array.shape)
            target_shape = self._compute_zoom_shape(
                array_shape=array_shape, axes=ctx.axes, slicing_ops=ctx.slicing
            )
            return self._dask_zoom(
                array=array, array_shape=array_shape, target_shape=target_shape
            )
        target_shape = self._compute_zoom_shape(
            array_shape=array.shape, axes=ctx.axes, slicing_ops=ctx.slicing
        )
        return self._numpy_zoom(array=array, target_shape=target_shape)

    def on_set(
        self, array: np.ndarray | da.Array, ctx: TransformContext
    ) -> np.ndarray | da.Array:
        """Apply the inverse scaling transformation before writing."""
        if isinstance(array, da.Array):
            array_shape = tuple(int(s) for s in array.shape)
            target_shape = self._compute_inverse_zoom_shape(
                array_shape=array_shape, axes=ctx.axes, slicing_ops=ctx.slicing
            )
            return self._dask_zoom(
                array=array, array_shape=array_shape, target_shape=target_shape
            )
        target_shape = self._compute_inverse_zoom_shape(
            array_shape=array.shape, axes=ctx.axes, slicing_ops=ctx.slicing
        )
        return self._numpy_zoom(array=array, target_shape=target_shape)
