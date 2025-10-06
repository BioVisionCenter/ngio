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
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps


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
        self, slice_: slice | int | tuple, scale: float, max_dim: int
    ) -> int:
        if isinstance(slice_, slice):
            _start = slice_.start or 0
            _start_int = math.floor(_start * scale)
            if slice_.stop is not None:
                _stop = slice_.stop * scale
                _stop = min(_stop, max_dim)
            else:
                _stop = max_dim
            _stop_int = math.ceil(_stop)
            target_shape = _stop_int - _start_int

        elif isinstance(slice_, int):
            target_shape = 1
        elif isinstance(slice_, tuple):
            target_shape = len(slice_) * scale
        else:
            raise ValueError(f"Unsupported slice type: {type(slice_)}")
        return math.ceil(target_shape)

    def _compute_zoom_shape(
        self,
        array_shape: Sequence[int],
        axes_ops: AxesOps,
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(axes_ops.output_axes)

        target_shape = []
        for shape, ax_name in zip(array_shape, axes_ops.output_axes, strict=True):
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
        axes_ops: AxesOps,
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(axes_ops.output_axes)

        target_shape = []
        for shape, ax_name in zip(array_shape, axes_ops.output_axes, strict=True):
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

        # Since we are basing the rescaling on the slice, we need to ensure
        # that the input image we got is roughly the right size.
        # This is a safeguard against user errors.
        expected_shape = self._compute_zoom_shape(
            array_shape=target_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        if any(
            abs(es - s) > 1 for es, s in zip(expected_shape, array_shape, strict=True)
        ):
            raise ValueError(
                f"Input array shape {array_shape} is not compatible with the expected "
                f"shape {expected_shape} based on the zoom transform.\n"
            )
        return tuple(target_shape)

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
        target_shape = self._compute_zoom_shape(
            array_shape=array.shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._numpy_zoom(array=array, target_shape=target_shape)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply the scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        target_shape = self._compute_zoom_shape(
            array_shape=array_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._dask_zoom(
            array=array, array_shape=array_shape, target_shape=target_shape
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply the inverse scaling transformation to a numpy array."""
        target_shape = self._compute_inverse_zoom_shape(
            array_shape=array.shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._numpy_zoom(array=array, target_shape=target_shape)

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply the inverse scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        target_shape = self._compute_inverse_zoom_shape(
            array_shape=array_shape, axes_ops=axes_ops, slicing_ops=slicing_ops
        )
        return self._dask_zoom(
            array=array, array_shape=array_shape, target_shape=target_shape
        )
