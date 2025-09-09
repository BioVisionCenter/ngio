from collections.abc import Sequence
from typing import Literal

import dask.array as da
import numpy as np

from ngio.common._array_io_utils import apply_sequence_axes_ops
from ngio.common._dimensions import Dimensions
from ngio.common._zoom import dask_zoom, numpy_zoom
from ngio.ome_zarr_meta.ngio_specs import SlicingOps


class ZoomTransform:
    def __init__(self, scale: Sequence[float], order: Literal[0, 1, 2]):
        self._scale = tuple(scale)
        self._order: Literal[0, 1, 2] = order

    @property
    def scale(self) -> tuple[float, ...]:
        return self._scale

    @property
    def inv_scale(self) -> tuple[float, ...]:
        return tuple([1 / s for s in self._scale])

    @classmethod
    def from_dimensions(
        cls,
        original_dimension: Dimensions,
        target_dimension: Dimensions,
        order: Literal[0, 1, 2],
    ):
        scale = []
        for o_ax_name in original_dimension.axes_mapper.axes_names:
            t_ax = target_dimension.axes_mapper.get_axis(name=o_ax_name)
            if t_ax is None:
                _scale = 1
            else:
                t_shape = target_dimension.get(o_ax_name)
                o_shape = original_dimension.get(o_ax_name)
                assert t_shape is not None and o_shape is not None
                _scale = t_shape / o_shape
            scale.append(_scale)

        return cls(scale, order)

    def apply_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps
    ) -> np.ndarray:
        """Apply the scaling transformation to a numpy array."""
        scale = tuple(
            apply_sequence_axes_ops(
                self.scale,
                default=1,
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        array = numpy_zoom(source_array=array, scale=scale, order=self._order)
        return array

    def apply_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps
    ) -> da.Array:
        """Apply the scaling transformation to a dask array."""
        scale = tuple(
            apply_sequence_axes_ops(
                self.scale,
                default=1,
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        array = dask_zoom(source_array=array, scale=scale, order=self._order)
        return array

    def apply_inverse_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps
    ) -> np.ndarray:
        """Apply the inverse scaling transformation to a numpy array."""
        scale = tuple(
            apply_sequence_axes_ops(
                self.inv_scale,
                default=1,
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        array = numpy_zoom(source_array=array, scale=scale, order=self._order)
        return array

    def apply_inverse_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps
    ) -> da.Array:
        """Apply the inverse scaling transformation to a dask array."""
        scale = tuple(
            apply_sequence_axes_ops(
                self.inv_scale,
                default=1,
                squeeze_axes=slicing_ops.squeeze_axes,
                transpose_axes=slicing_ops.transpose_axes,
                expand_axes=slicing_ops.expand_axes,
            )
        )
        array = dask_zoom(source_array=array, scale=scale, order=self._order)
        return array
