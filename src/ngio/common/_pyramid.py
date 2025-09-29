import math
from collections.abc import Callable, Sequence
from typing import Literal

import dask.array as da
import numpy as np
import zarr
from zarr.types import DIMENSION_SEPARATOR

from ngio.common._zoom import (
    InterpolationOrder,
    _zoom_inputs_check,
    dask_zoom,
    numpy_zoom,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)


def _on_disk_numpy_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    target[...] = numpy_zoom(source[...], target_shape=target.shape, order=order)


def _on_disk_dask_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    source_array = da.from_zarr(source)
    target_array = dask_zoom(source_array, target_shape=target.shape, order=order)

    target_array = target_array.rechunk(target.chunks)
    target_array.compute_chunk_sizes()
    target_array.to_zarr(target)


def _on_disk_coarsen(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder = "linear",
    aggregation_function: Callable | None = None,
) -> None:
    """Apply a coarsening operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to coarsen.
        target (zarr.Array): The target array to save the coarsened result to.
        order (InterpolationOrder): The order of interpolation is not really implemented
            for coarsening, but it is kept for compatibility with the zoom function.
            order="linear" -> linear interpolation ~ np.mean
            order="nearest" -> nearest interpolation ~ np.max
        aggregation_function (np.ufunc): The aggregation function to use.
    """
    source_array = da.from_zarr(source)

    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=None, target_shape=target.shape
    )

    assert _target_shape == target.shape, (
        "Target shape must match the target array shape"
    )

    if aggregation_function is None:
        if order == "linear":
            aggregation_function = np.mean
        elif order == "nearest":
            aggregation_function = np.max
        elif order == "cubic":
            raise NgioValueError("Cubic interpolation is not supported for coarsening.")
        else:
            raise NgioValueError(
                f"Aggregation function must be provided for order {order}"
            )

    coarsening_setup = {}
    for i, s in enumerate(_scale):
        factor = 1 / s
        # This check is very strict, but it is necessary to avoid
        # a few pixels shift in the coarsening
        # We could add a tolerance
        if factor.is_integer():
            coarsening_setup[i] = int(factor)
        else:
            raise NgioValueError(
                f"Coarsening factor must be an integer, got {factor} on axis {i}"
            )

    out_target = da.coarsen(
        aggregation_function, source_array, coarsening_setup, trim_excess=True
    )
    out_target = out_target.rechunk(target.chunks)
    out_target.to_zarr(target)


def on_disk_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder = "linear",
    mode: Literal["dask", "numpy", "coarsen"] = "dask",
) -> None:
    """Apply a zoom operation from a source zarr array to a target zarr array.

    Args:
        source (zarr.Array): The source array to zoom.
        target (zarr.Array): The target array to save the zoomed result to.
        order (InterpolationOrder): The order of interpolation. Defaults to "linear".
        mode (Literal["dask", "numpy", "coarsen"]): The mode to use. Defaults to "dask".
    """
    if not isinstance(source, zarr.Array):
        raise NgioValueError("source must be a zarr array")

    if not isinstance(target, zarr.Array):
        raise NgioValueError("target must be a zarr array")

    if source.dtype != target.dtype:
        raise NgioValueError("source and target must have the same dtype")

    match mode:
        case "numpy":
            return _on_disk_numpy_zoom(source, target, order)
        case "dask":
            return _on_disk_dask_zoom(source, target, order)
        case "coarsen":
            return _on_disk_coarsen(
                source,
                target,
            )
        case _:
            raise NgioValueError("mode must be either 'dask', 'numpy' or 'coarsen'")


def _find_closest_arrays(
    processed: list[zarr.Array], to_be_processed: list[zarr.Array]
) -> tuple[np.intp, np.intp]:
    dist_matrix = np.zeros((len(processed), len(to_be_processed)))
    for i, arr_to_proc in enumerate(to_be_processed):
        for j, proc_arr in enumerate(processed):
            dist_matrix[j, i] = np.sqrt(
                np.sum(
                    [
                        (s1 - s2) ** 2
                        for s1, s2 in zip(
                            arr_to_proc.shape, proc_arr.shape, strict=False
                        )
                    ]
                )
            )

    indices = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
    assert len(indices) == 2, "Indices must be of length 2"
    return indices


def consolidate_pyramid(
    source: zarr.Array,
    targets: list[zarr.Array],
    order: InterpolationOrder = "linear",
    mode: Literal["dask", "numpy", "coarsen"] = "dask",
) -> None:
    """Consolidate the Zarr array."""
    processed = [source]
    to_be_processed = targets

    while to_be_processed:
        source_id, target_id = _find_closest_arrays(processed, to_be_processed)

        source_image = processed[source_id]
        target_image = to_be_processed.pop(target_id)

        on_disk_zoom(
            source=source_image,
            target=target_image,
            mode=mode,
            order=order,
        )
        processed.append(target_image)


def _maybe_int(value: float | int) -> float | int:
    """Convert a float to an int if it is an integer."""
    if isinstance(value, int):
        return value
    if value.is_integer():
        return int(value)
    return value


def init_empty_pyramid(
    store: StoreOrGroup,
    paths: list[str],
    ref_shape: Sequence[int],
    scaling_factors: Sequence[float],
    chunks: Sequence[int] | None = None,
    dtype: str = "uint16",
    mode: AccessModeLiteral = "a",
    dimension_separator: DIMENSION_SEPARATOR = "/",
    compressor="default",
) -> None:
    # Return the an Image object
    if chunks is not None and len(chunks) != len(ref_shape):
        raise NgioValueError(
            "The shape and chunks must have the same number of dimensions."
        )

    if chunks is not None:
        chunks = [min(c, s) for c, s in zip(chunks, ref_shape, strict=True)]

    if len(ref_shape) != len(scaling_factors):
        raise NgioValueError(
            "The shape and scaling factor must have the same number of dimensions."
        )

    # Ensure scaling factors are int if possible
    # To reduce the risk of floating point issues
    scaling_factors = [_maybe_int(s) for s in scaling_factors]

    root_group = open_group_wrapper(store, mode=mode)

    for path in paths:
        if any(s < 1 for s in ref_shape):
            raise NgioValueError(
                "Level shape must be at least 1 on all dimensions. "
                f"Calculated shape: {ref_shape} at level {path}."
            )
        new_arr = root_group.zeros(
            name=path,
            shape=ref_shape,
            dtype=dtype,
            chunks=chunks,
            dimension_separator=dimension_separator,
            overwrite=True,
            compressor=compressor,
        )

        _shape = [
            math.floor(s / sc) for s, sc in zip(ref_shape, scaling_factors, strict=True)
        ]
        ref_shape = _shape

        if chunks is None:
            chunks = new_arr.chunks
            assert chunks is not None
        chunks = [min(c, s) for c, s in zip(chunks, ref_shape, strict=True)]
    return None
