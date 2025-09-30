import math
from collections.abc import Sequence

import dask.array as da
import numpy as np

from ngio.common._zoom import (
    InterpolationOrder,
    dask_zoom,
    numpy_zoom,
)
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes import SlicingOps
from ngio.ome_zarr_meta import AxesOps


class ZoomTransform:
    def __init__(
        self,
        input_image: AbstractImage,
        target_image: AbstractImage,
        order: InterpolationOrder = "nearest",
    ) -> None:
        self._input_dimensions = input_image.dimensions
        self._target_dimensions = target_image.dimensions
        self._input_pixel_size = input_image.pixel_size
        self._target_pixel_size = target_image.pixel_size
        self._order: InterpolationOrder = order

    def _normalize_shape(
        self, slice_: slice | int | tuple, shape: int, scale: float, out_dim: int
    ) -> int:
        if isinstance(slice_, slice):
            _start = slice_.start or 0
            _stop = slice_.stop or shape
            out_shape = (_stop - _start) * scale
            max_out_shape = out_dim - _start * scale
            out_shape = min(out_shape, max_out_shape)
        elif isinstance(slice_, int):
            out_shape = 1
        elif isinstance(slice_, tuple):
            out_shape = len(slice_) * scale
        else:
            raise ValueError(f"Unsupported slice type: {type(slice_)}")
        return math.ceil(out_shape)

    def _compute_zoom_shape(
        self,
        array_shape: Sequence[int],
        axes_ops: AxesOps,
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(axes_ops.in_memory_axes)

        out_shape = []
        for shape, ax_name in zip(array_shape, axes_ops.in_memory_axes, strict=True):
            ax_type = self._input_dimensions.axes_handler.get_axis(ax_name)
            if ax_type is not None and ax_type.axis_type == "channel":
                # Do not scale channel axis
                out_shape.append(shape)
                continue
            out_dim = self._target_dimensions.get(ax_name, default=1)
            in_pix = self._input_pixel_size.get(ax_name, default=1.0)
            out_pix = self._target_pixel_size.get(ax_name, default=1.0)
            slice_ = slicing_ops.get(ax_name, normalize=False)
            scale = in_pix / out_pix
            _out_shape = self._normalize_shape(
                slice_=slice_, shape=shape, scale=scale, out_dim=out_dim
            )
            out_shape.append(_out_shape)
        return tuple(out_shape)

    def _compute_inverse_zoom_shape(
        self,
        array_shape: Sequence[int],
        axes_ops: AxesOps,
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(axes_ops.in_memory_axes)

        out_shape = []
        for shape, ax_name in zip(array_shape, axes_ops.in_memory_axes, strict=True):
            ax_type = self._input_dimensions.axes_handler.get_axis(ax_name)
            if ax_type is not None and ax_type.axis_type == "channel":
                # Do not scale channel axis
                out_shape.append(shape)
                continue
            in_dim = self._input_dimensions.get(ax_name, default=1)
            slice_ = slicing_ops.get(ax_name=ax_name, normalize=True)
            out_shape.append(
                self._normalize_shape(
                    slice_=slice_, shape=shape, scale=1, out_dim=in_dim
                )
            )

        # Since we are basing the rescaling on the slice, we need to ensure
        # that the input image we got is roughly the right size.
        # This is a safeguard against user errors.
        expected_shape = self._compute_zoom_shape(
            array_shape=out_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        if any(
            abs(es - s) > 1 for es, s in zip(expected_shape, array_shape, strict=True)
        ):
            raise ValueError(
                f"Input array shape {array_shape} is not compatible with the expected "
                f"shape {expected_shape} based on the zoom transform.\n"
            )
        return tuple(out_shape)

    def _numpy_zoom(
        self, array: np.ndarray, target_shape: tuple[int, ...]
    ) -> np.ndarray:
        if array.shape == target_shape:
            return array
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
        return dask_zoom(
            source_array=array, target_shape=target_shape, order=self._order
        )

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply the scaling transformation to a numpy array."""
        out_shape = self._compute_zoom_shape(
            array_shape=array.shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._numpy_zoom(array=array, target_shape=out_shape)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply the scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        out_shape = self._compute_zoom_shape(
            array_shape=array_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._dask_zoom(
            array=array, array_shape=array_shape, target_shape=out_shape
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply the inverse scaling transformation to a numpy array."""
        out_shape = self._compute_inverse_zoom_shape(
            array_shape=array.shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._numpy_zoom(array=array, target_shape=out_shape)

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply the inverse scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        out_shape = self._compute_inverse_zoom_shape(
            array_shape=array_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._dask_zoom(
            array=array, array_shape=array_shape, target_shape=out_shape
        )
