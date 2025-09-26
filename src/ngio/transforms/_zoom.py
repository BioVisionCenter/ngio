import math
from collections.abc import Sequence

import dask.array as da
import numpy as np

from ngio.common._zoom import (
    InterpolationOrder,
    dask_zoom,
    numpy_zoom,
    scale_factor_from_pixel_size,
)
from ngio.images._abstract_image import AbstractImage

SlicingOps: None = None


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
        self._scale = scale_factor_from_pixel_size(
            original_dimension=self._input_dimensions,
            original_pixel_size=self._input_pixel_size,
            target_pixel_size=self._target_pixel_size,
        )

    @property
    def scale(self) -> tuple[float, ...]:
        return self._scale

    @property
    def inv_scale(self) -> tuple[float, ...]:
        return tuple([1 / s for s in self._scale])

    def _predict_zoomed_shape(
        self,
        array_shape: Sequence[int],
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(slicing_ops.in_memory_axes)

        if slicing_ops.slicing_tuple is None:
            slice_tuple = tuple([slice(None)] * len(array_shape))
        else:
            slice_tuple = slicing_ops.slicing_tuple
        print("fw before", slice_tuple)
        slice_tuple = tuple(
            apply_sequence_axes_ops(
                slice_tuple,
                default=slice(None),
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        print("fw after", slice_tuple)
        print("slicing", slicing_ops)
        out_shape = []
        for shape, ax_name, sl in zip(
            array_shape, slicing_ops.in_memory_axes, slice_tuple, strict=True
        ):
            out_dim = self._target_dimensions.get(ax_name, default=1)
            in_pix = self._input_pixel_size.get(ax_name, default=1.0)
            out_pix = self._target_pixel_size.get(ax_name, default=1.0)

            scale = in_pix / out_pix

            if isinstance(sl, slice):
                _start = sl.start or 0
                _stop = sl.stop or shape
                _shape = _stop - _start
                _shape = min(_shape * scale, out_dim)
            elif isinstance(sl, int):
                _shape = 1
            elif isinstance(sl, tuple):
                _shape = len(sl) * scale
            else:
                raise ValueError(f"Unsupported slice type: {type(sl)}")
            _shape = math.ceil(_shape)
            out_shape.append(_shape)

        return tuple(out_shape)

    def _predict_inverse_zoomed_shape(
        self,
        array_shape: Sequence[int],
        slicing_ops: SlicingOps,
    ) -> tuple[int, ...]:
        assert len(array_shape) == len(slicing_ops.in_memory_axes)

        if slicing_ops.sicing_tuple is None:
            slice_tuple = tuple([slice(None)] * len(array_shape))
        else:
            slice_tuple = slicing_ops.sicing_tuple

        print("inv before", slice_tuple)
        slice_tuple = tuple(
            apply_sequence_axes_ops(
                slice_tuple,
                default=slice(None),
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        print("inv after", slice_tuple)
        print("slicing", slicing_ops)
        out_shape = []
        for shape, ax_name, sl in zip(
            array_shape, slicing_ops.in_memory_axes, slice_tuple, strict=True
        ):
            in_dim = self._input_dimensions.get(ax_name, default=1)
            in_pix = self._input_pixel_size.get(ax_name, default=1.0)
            out_pix = self._target_pixel_size.get(ax_name, default=1.0)

            scale = out_pix / in_pix
            print(ax_name, scale, shape, sl, in_dim)

            if isinstance(sl, slice):
                _start = sl.start or 0
                _stop = sl.stop or shape
                _shape = _stop - _start
                _shape = min(_shape * scale, in_dim)
            elif isinstance(sl, int):
                _shape = 1
            elif isinstance(sl, tuple):
                _shape = len(sl) * scale
            else:
                raise ValueError(f"Unsupported slice type: {type(sl)}")
            _shape = math.ceil(_shape)
            out_shape.append(_shape)
            print(_shape)
        return tuple(out_shape)

    def apply_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps
    ) -> np.ndarray:
        """Apply the scaling transformation to a numpy array."""
        predicted_shape = self._predict_zoomed_shape(
            array_shape=array.shape, slicing_ops=slicing_ops
        )
        return numpy_zoom(
            source_array=array, target_shape=predicted_shape, order=self._order
        )

    def apply_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps
    ) -> da.Array:
        """Apply the scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        predicted_shape = self._predict_zoomed_shape(
            array_shape=array_shape, slicing_ops=slicing_ops
        )
        return dask_zoom(
            source_array=array, target_shape=predicted_shape, order=self._order
        )

    def apply_inverse_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps
    ) -> np.ndarray:
        """Apply the inverse scaling transformation to a numpy array."""
        predicted_shape = self._predict_inverse_zoomed_shape(
            array_shape=array.shape, slicing_ops=slicing_ops
        )
        return numpy_zoom(
            source_array=array, target_shape=predicted_shape, order=self._order
        )

    def apply_inverse_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps
    ) -> da.Array:
        """Apply the inverse scaling transformation to a dask array."""
        array_shape = tuple(int(s) for s in array.shape)
        predicted_shape = self._predict_inverse_zoomed_shape(
            array_shape=array_shape, slicing_ops=slicing_ops
        )
        return dask_zoom(
            source_array=array, target_shape=predicted_shape, order=self._order
        )
