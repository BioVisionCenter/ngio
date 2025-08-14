from functools import partial
from typing import Literal

import dask.array as da
import numpy as np
from scipy.ndimage import zoom as scipy_zoom

from ngio.utils import NgioValueError


def _stacked_zoom(x, zoom_y, zoom_x, order=1, mode="grid-constant", grid_mode=True):
    *rest, yshape, xshape = x.shape
    x = x.reshape(-1, yshape, xshape)
    scale_xy = (zoom_y, zoom_x)
    _x_out = [
        scipy_zoom(x[i], scale_xy, order=order, mode=mode, grid_mode=grid_mode)
        for i in range(x.shape[0])
    ]
    x_out = np.stack(_x_out)  # type: ignore (scipy_zoom returns np.ndarray, but type is not inferred correctly)
    return x_out.reshape(*rest, *x_out.shape[1:])


def fast_zoom(x, zoom, order=1, mode="grid-constant", grid_mode=True, auto_stack=True):
    """Fast zoom function.

    Scipy zoom function that can handle singleton dimensions
    but the performance degrades with the number of dimensions.

    This function has two small optimizations:
     - it removes singleton dimensions before calling zoom
     - if it detects that the zoom is only on the last two dimensions
         it stacks the first dimensions to call zoom only on the last two.
    """
    mask = np.isclose(x.shape, 1)
    zoom = np.array(zoom)
    singletons = tuple(np.where(mask)[0])
    xs = np.squeeze(x, axis=singletons)
    new_zoom = zoom[~mask]

    *zoom_rest, zoom_y, zoom_x = new_zoom
    if auto_stack and np.allclose(zoom_rest, 1):
        xs = _stacked_zoom(
            xs, zoom_y, zoom_x, order=order, mode=mode, grid_mode=grid_mode
        )
    else:
        xs = scipy_zoom(xs, new_zoom, order=order, mode=mode, grid_mode=grid_mode)
    x = np.expand_dims(xs, axis=singletons)  # type: ignore (scipy_zoom returns np.ndarray, but type is not inferred correctly)
    return x


def _zoom_inputs_check(
    source_array: np.ndarray | da.Array,
    scale: tuple[int | float, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if scale is None and target_shape is None:
        raise NgioValueError("Either scale or target_shape must be provided")

    if scale is not None and target_shape is not None:
        raise NgioValueError("Only one of scale or target_shape must be provided")

    if scale is None:
        assert target_shape is not None, "Target shape must be provided"
        if len(target_shape) != source_array.ndim:
            raise NgioValueError(
                "Target shape must have the "
                "same number of dimensions as "
                "the source array"
            )
        _scale = np.array(target_shape) / np.array(source_array.shape)
        _target_shape = target_shape
    else:
        _scale = np.array(scale)
        _target_shape = tuple(np.array(source_array.shape) * scale)

    if len(_scale) != source_array.ndim:
        raise NgioValueError(
            f"Cannot scale array of shape {source_array.shape} with factors {_scale}."
            " Target shape must have the same number of dimensions as the source array."
        )

    return _scale, _target_shape


def dask_zoom(
    source_array: da.Array,
    scale: tuple[float | int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    order: Literal[0, 1, 2] = 1,
) -> da.Array:
    """Dask implementation of zooming an array.

    Only one of scale or target_shape must be provided.

    Args:
        source_array (da.Array): The source array to zoom.
        scale (tuple[int, ...] | None): The scale factor to zoom by.
        target_shape (tuple[int, ...], None): The target shape to zoom to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.

    Returns:
        da.Array: The zoomed array.
    """
    # This function follow the implementation from:
    # https://github.com/ome/ome-zarr-py/blob/master/ome_zarr/dask_utils.py
    # The module was contributed by Andreas Eisenbarth @aeisenbarth
    # See https://github.com/toloudis/ome-zarr-py/pull/

    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=scale, target_shape=target_shape
    )

    # Rechunk to better match the scaling operation
    source_chunks = np.array(source_array.chunksize)  # type: ignore (da.Array.chunksize is a tuple of ints)
    better_source_chunks = np.maximum(1, np.round(source_chunks * _scale) / _scale)
    better_source_chunks = better_source_chunks.astype(int)
    source_array = source_array.rechunk(better_source_chunks)  # type: ignore (better_source_chunks is a valid input for rechunk)

    # Calculate the block output shape
    block_output_shape = tuple(np.ceil(better_source_chunks * _scale).astype(int))

    zoom_wrapper = partial(
        fast_zoom, zoom=_scale, order=order, mode="grid-constant", grid_mode=True
    )

    out_array = da.map_blocks(
        zoom_wrapper, source_array, chunks=block_output_shape, dtype=source_array.dtype
    )

    # Slice and rechunk to target
    slices = tuple(slice(0, ts, 1) for ts in _target_shape)
    out_array = out_array[slices]
    return out_array


def numpy_zoom(
    source_array: np.ndarray,
    scale: tuple[int | float, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    order: Literal[0, 1, 2] = 1,
) -> np.ndarray:
    """Numpy implementation of zooming an array.

    Only one of scale or target_shape must be provided.

    Args:
        source_array (np.ndarray): The source array to zoom.
        scale (tuple[int, ...] | None): The scale factor to zoom by.
        target_shape (tuple[int, ...], None): The target shape to zoom to.
        order (Literal[0, 1, 2]): The order of interpolation. Defaults to 1.

    Returns:
        np.ndarray: The zoomed array
    """
    _scale, _ = _zoom_inputs_check(
        source_array=source_array, scale=scale, target_shape=target_shape
    )

    out_array = fast_zoom(
        source_array, zoom=_scale, order=order, mode="grid-constant", grid_mode=True
    )
    assert isinstance(out_array, np.ndarray)
    return out_array
