from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, model_validator

from ngio.common._zoom import (
    InterpolationOrder,
    _zoom_inputs_check,
    dask_zoom,
    numpy_zoom,
)
from ngio.utils import (
    NgioValueError,
)


def _on_disk_numpy_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    source_array = source[...]
    if not isinstance(source_array, np.ndarray):
        raise NgioValueError("source zarr array could not be read as a numpy array")
    target[...] = numpy_zoom(source_array, target_shape=target.shape, order=order)


def _on_disk_dask_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    source_array = da.from_zarr(source)
    target_array = dask_zoom(source_array, target_shape=target.shape, order=order)

    # This is a potential fix for Dask 2025.11
    # import dask.config
    # chunk_size_bytes = np.prod(target.chunks) * target_array.dtype.itemsize
    # current_chunk_size = dask.config.get("array.chunk-size")
    # Increase the chunk size to avoid dask potentially creating
    # corrupted chunks when writing chunks that are not multiple of the
    # target chunk size
    # dask.config.set({"array.chunk-size": f"{chunk_size_bytes}B"})
    target_array = target_array.rechunk(target.chunks)
    target_array = target_array.compute_chunk_sizes()
    target_array.to_zarr(target)
    # Restore previous chunk size
    # dask.config.set({"array.chunk-size": current_chunk_size})


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


################################################
#
# Builders for image pyramids
#
################################################

ChunksLike = tuple[int, ...] | Literal["auto"]
ShardsLike = tuple[int, ...] | Literal["auto"]


def shapes_from_scaling_factors(
    base_shape: tuple[int, ...],
    scaling_factors: tuple[float, ...],
    num_levels: int,
) -> list[tuple[int, ...]]:
    """Compute the shapes of each level in the pyramid from scaling factors.

    Args:
        base_shape (tuple[int, ...]): The shape of the base level.
        scaling_factors (tuple[float, ...]): The scaling factors between levels.
        num_levels (int): The number of levels in the pyramid.

    Returns:
        list[tuple[int, ...]]: The shapes of each level in the pyramid.
    """
    shapes = []
    current_shape = base_shape
    for _ in range(num_levels):
        shapes.append(current_shape)
        current_shape = tuple(
            max(1, int(s / f))
            for s, f in zip(current_shape, scaling_factors, strict=True)
        )
    return shapes


def _check_order(shapes: Sequence[tuple[int, ...]]):
    """Check if the shapes are in decreasing order."""
    num_pixels = [np.prod(shape) for shape in shapes]
    for i in range(1, len(num_pixels)):
        if num_pixels[i] >= num_pixels[i - 1]:
            raise NgioValueError("Shapes are not in decreasing order.")


class PyramidLevel(BaseModel):
    path: str
    shape: tuple[int, ...]
    scale: tuple[float, ...]
    chunks: ChunksLike = "auto"
    shards: ShardsLike | None = None

    @model_validator(mode="after")
    def _model_validation(self) -> "PyramidLevel":
        # Same length as shape
        if len(self.scale) != len(self.shape):
            raise NgioValueError(
                "Scale must have the same length as shape "
                f"({len(self.shape)}), got {len(self.scale)}"
            )
        if any(isinstance(s, float) and s < 0 for s in self.scale):
            raise NgioValueError("Scale values must be positive.")

        if isinstance(self.chunks, tuple):
            if len(self.chunks) != len(self.shape):
                raise NgioValueError(
                    "Chunks must have the same length as shape "
                    f"({len(self.shape)}), got {len(self.chunks)}"
                )
            normalized_chunks = []
            for dim_size, chunk_size in zip(self.shape, self.chunks, strict=True):
                normalized_chunks.append(min(dim_size, chunk_size))
            self.chunks = tuple(normalized_chunks)

        if isinstance(self.shards, tuple):
            if len(self.shards) != len(self.shape):
                raise NgioValueError(
                    "Shards must have the same length as shape "
                    f"({len(self.shape)}), got {len(self.shards)}"
                )
            normalized_shards = []
            for dim_size, shard_size in zip(self.shape, self.shards, strict=True):
                normalized_shards.append(min(dim_size, shard_size))
            self.shards = tuple(normalized_shards)
        return self


class ImagePyramidBuilder(BaseModel):
    levels: list[PyramidLevel]
    axes: tuple[str, ...]
    data_type: str = "uint16"
    dimension_separator: Literal[".", "/"] = "/"
    compressors: Any = "auto"
    zarr_format: Literal[2, 3] = 2
    other_array_kwargs: Mapping[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_scaling_factors(
        cls,
        levels_paths: tuple[str, ...],
        scaling_factors: tuple[float, ...],
        base_shape: tuple[int, ...],
        base_scale: tuple[float, ...],
        axes: tuple[str, ...],
        chunks: ChunksLike = "auto",
        shards: ShardsLike | None = None,
        data_type: str = "uint16",
        dimension_separator: Literal[".", "/"] = "/",
        compressors: Any = "auto",
        zarr_format: Literal[2, 3] = 2,
        other_array_kwargs: Mapping[str, Any] | None = None,
    ) -> "ImagePyramidBuilder":
        shapes = shapes_from_scaling_factors(
            base_shape=base_shape,
            scaling_factors=scaling_factors,
            num_levels=len(levels_paths),
        )
        return cls.from_shapes(
            shapes=shapes,
            base_scale=base_scale,
            axes=axes,
            levels_paths=levels_paths,
            chunks=chunks,
            shards=shards,
            data_type=data_type,
            dimension_separator=dimension_separator,
            compressors=compressors,
            zarr_format=zarr_format,
            other_array_kwargs=other_array_kwargs,
        )

    @classmethod
    def from_shapes(
        cls,
        shapes: Sequence[tuple[int, ...]],
        base_scale: tuple[float, ...],
        axes: tuple[str, ...],
        levels_paths: Sequence[str] | None = None,
        chunks: ChunksLike = "auto",
        shards: ShardsLike | None = None,
        data_type: str = "uint16",
        dimension_separator: Literal[".", "/"] = "/",
        compressors: Any = "auto",
        zarr_format: Literal[2, 3] = 2,
        other_array_kwargs: Mapping[str, Any] | None = None,
    ) -> "ImagePyramidBuilder":
        levels = []
        if levels_paths is None:
            levels_paths = tuple(str(i) for i in range(len(shapes)))
        _check_order(shapes)
        scale_ = base_scale
        for i, (path, shape) in enumerate(zip(levels_paths, shapes, strict=True)):
            levels.append(
                PyramidLevel(
                    path=path,
                    shape=shape,
                    scale=scale_,
                    chunks=chunks,
                    shards=shards,
                )
            )
            if i + 1 < len(shapes):
                # This only works for downsampling pyramids
                # The _check_order function ensures that
                # shapes are decreasing
                next_shape = shapes[i + 1]
                scaling_factor = tuple(
                    s1 / s2
                    for s1, s2 in zip(
                        shape,
                        next_shape,
                        strict=True,
                    )
                )
                scale_ = tuple(
                    s * f for s, f in zip(scale_, scaling_factor, strict=True)
                )
        other_array_kwargs = other_array_kwargs or {}
        return cls(
            levels=levels,
            axes=axes,
            data_type=data_type,
            dimension_separator=dimension_separator,
            compressors=compressors,
            zarr_format=zarr_format,
            other_array_kwargs=other_array_kwargs,
        )

    def to_zarr(self, group: zarr.Group) -> None:
        """Save the pyramid specification to a Zarr group.

        Args:
            group (zarr.Group): The Zarr group to save the pyramid specification to.
        """
        array_static_kwargs = {
            "dtype": self.data_type,
            "overwrite": True,
            "compressors": self.compressors,
            **self.other_array_kwargs,
        }

        if self.zarr_format == 2:
            array_static_kwargs["chunk_key_encoding"] = {
                "name": "v2",
                "separator": self.dimension_separator,
            }
        else:
            array_static_kwargs["chunk_key_encoding"] = {
                "name": "default",
                "separator": self.dimension_separator,
            }
            array_static_kwargs["dimension_names"] = self.axes
        for p_level in self.levels:
            group.create_array(
                name=p_level.path,
                shape=tuple(p_level.shape),
                chunks=p_level.chunks,
                shards=p_level.shards,
                **array_static_kwargs,
            )
