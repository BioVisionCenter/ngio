import itertools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, model_validator

from ngio.common._dask_io import store_dask
from ngio.common._zoom import (
    InterpolationOrder,
    _zoom_inputs_check,
    dask_zoom,
    numpy_zoom,
)
from ngio.config import get_config
from ngio.utils import (
    NgioFutureWarning,
    NgioValueError,
    deprecated_alias,
    stacklevel_of_first_caller,
)

#: How `consolidate` builds each level. `"auto"` takes the in-memory path only
#: where it is bit-identical to the chunked one; see `_resolve_auto_mode`.
ConsolidationMode: TypeAlias = Literal["dask", "numpy", "coarsen", "auto"]

#: The release in which `mode` starts defaulting to `"auto"`.
_DEFAULT_CHANGES_IN = "1.2"


def _read_numpy(source: zarr.Array) -> np.ndarray:
    source_array = source[...]
    if not isinstance(source_array, np.ndarray):
        raise NgioValueError("source zarr array could not be read as a numpy array")
    return source_array


def _coarsen_expr(
    source_array: da.Array,
    target: zarr.Array,
    order: InterpolationOrder = "linear",
    aggregation_function: Callable | None = None,
) -> da.Array:
    """Coarsen a dask array onto a target's shape, as an unevaluated expression.

    Args:
        source_array: The source array to coarsen.
        target: The array whose shape and dtype the result must match.
        order: Not really implemented for coarsening, kept for compatibility with
            the zoom function.
            order="linear" -> linear interpolation ~ np.mean
            order="nearest" -> nearest interpolation ~ np.max
        aggregation_function: The aggregation function to use.
    """
    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=None, target_shape=target.shape
    )

    if _target_shape != target.shape:
        raise NgioValueError(
            f"Coarsening would produce shape {_target_shape}, but the target "
            f"array has shape {target.shape}."
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
        factor = int(np.round(1 / s)) if s > 0 else 0
        if factor < 1:
            # Coarsening aggregates whole blocks, so it can only ever go down.
            # Consolidating from a middle level asks for the levels *above* the
            # source too, and those edges are upsamples. Without this the factor
            # is 0 and the failure surfaces from inside dask as a divide-by-zero
            # in `aligned_coarsen_chunks`, which says nothing about the cause.
            raise NgioValueError(
                f"Cannot coarsen axis {i} from size {source_array.shape[i]} to "
                f"{target.shape[i]}: coarsening only downsamples. Use "
                'mode="dask" or mode="numpy" to build a level larger than its '
                "source."
            )
        coarsening_setup[i] = factor

    return da.coarsen(
        aggregation_function, source_array, coarsening_setup, trim_excess=True
    )


def _zoom_expr(
    source_array: da.Array,
    target: zarr.Array,
    order: InterpolationOrder,
    mode: Literal["dask", "coarsen"],
    aggregation_function: Callable | None = None,
) -> da.Array:
    """One pyramid level as an unevaluated dask expression.

    The `astype` is not cosmetic. `da.coarsen(np.mean, ...)` promotes an integer
    source to float64; written to the store that float is cast back, so the next
    level down reads the cast value. An expression handed straight to the next
    level would instead propagate the float, and a mean pyramid built that way
    differs from a stored one in 28% of level 2 and 63% of level 3. Casting here
    is what the store would have done anyway, just done early -- which also drops
    the float64 intermediate and makes coarsening markedly lighter.
    """
    if mode == "coarsen":
        out = _coarsen_expr(source_array, target, order, aggregation_function)
    else:
        out = dask_zoom(source_array, target_shape=target.shape, order=order)

    if out.dtype != target.dtype:
        out = out.astype(target.dtype)
    return out


def _on_disk_numpy_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    target[...] = numpy_zoom(
        _read_numpy(source), target_shape=target.shape, order=order
    )


def _on_disk_dask_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    # No compute_chunk_sizes() here: it would execute the whole read -> zoom
    # graph purely to re-learn block shapes, throw the pixels away, and leave
    # the write to run the same graph again -- exactly double the chunk reads.
    # No rechunk either: store_dask rechunks onto the target's write unit,
    # which is `shards or chunks`. Rechunking to `target.chunks` here was worse
    # than doing nothing on a sharded target -- that is the shard's *inner*
    # chunk shape, so every block became a partial shard write.
    store_dask(_zoom_expr(da.from_zarr(source), target, order, "dask"), target)


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
    # See _on_disk_dask_zoom: store_dask owns the rechunk onto the write unit.
    store_dask(
        _zoom_expr(
            da.from_zarr(source), target, order, "coarsen", aggregation_function
        ),
        target,
    )


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
                order,
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

    # `dist_matrix` is 2-D by construction, so unravel_index yields a 2-tuple.
    row, column = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
    return row, column


def _consolidation_plan(
    source: zarr.Array, targets: list[zarr.Array]
) -> list[tuple[int, zarr.Array]]:
    """Order the pyramid as source -> target edges, without touching the store.

    The same greedy nearest-shape walk consolidation has always done -- same
    `_find_closest_arrays`, same sequence -- it just records the edges instead of
    executing them, so the caller can build one graph spanning all of them. For a
    well-formed pyramid the result is two chains rooted at the source: down to the
    coarsest level, then back up to the finest, each step from its immediate
    neighbour. Consolidating from a middle level therefore still upsamples the
    levels above it, as it always has.

    Recording the greedy result rather than replacing it with an explicit sort is
    deliberate. It costs nothing (see the O(L^3) note on `_find_closest_arrays`:
    L is 4-6) and it keeps the ordering provably unchanged even for a hand-built
    pyramid whose shapes are not monotone.

    Returns:
        `(parent, target)` pairs, where `parent` indexes `[source, *targets in
        emission order]`. Every parent is emitted before any of its children.
    """
    nodes = [source]
    remaining = list(targets)
    plan: list[tuple[int, zarr.Array]] = []

    while remaining:
        source_id, target_id = _find_closest_arrays(nodes, remaining)
        target = remaining.pop(int(target_id))
        plan.append((int(source_id), target))
        nodes.append(target)
    return plan


def _is_integral_downsample(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> bool:
    """Whether the blockwise and whole-array zooms agree exactly on this edge.

    `dask_zoom` snaps its block grid to the scaling ratio, so with a 2-tap kernel
    and an exact integer factor no output sample needs a neighbour from another
    block and the two paths land on the same pixels. Let the ratio slip off an
    integer -- 2.004, which is what a 501-pixel axis gives -- and they disagree
    on 99.9% of the level; upsample instead and they disagree on 1.6%.
    """
    return all(
        t >= 1 and s >= t and s % t == 0
        for s, t in zip(source_shape, target_shape, strict=True)
    )


def _resolve_auto_mode(
    source: zarr.Array,
    plan: list[tuple[int, zarr.Array]],
    order: InterpolationOrder,
) -> Literal["numpy", "dask"]:
    """Choose the in-memory path only where it is bit-identical to the chunked one.

    Size alone would be the wrong test. `numpy` and `dask` are two different
    algorithms, not one algorithm at two block sizes, and a threshold that
    silently switched between two different answers would be worse than making
    the caller pick. So `"auto"` takes `numpy` only inside the envelope where
    they provably agree, and quietly falls back to `dask` everywhere else.
    """
    if source.nbytes > get_config().consolidation.numpy_max_bytes:
        return "dask"

    if order == "cubic":
        # A cubic kernel reads four samples per output, so a blockwise zoom with
        # no halo is wrong at every block boundary -- the paths differ on 10-16%
        # of pixels even on a plain power-of-two pyramid.
        return "dask"

    nodes = [source]
    for parent, target in plan:
        if not _is_integral_downsample(nodes[parent].shape, target.shape):
            return "dask"
        nodes.append(target)
    return "numpy"


def _warn_default_will_change() -> None:
    """Announce the coming default, only where it will actually change something.

    Fired from the mode resolution rather than from `Image.consolidate`, because
    only here is the plan known -- and the warning is worth nothing to someone
    whose pyramid `"auto"` would decline anyway. Deduplication is left to the
    `warnings` module, which suppresses repeats per call site. A once-per-process
    flag would be worse: under `filterwarnings = ["error"]` only the first caller
    raises, so which test fails would depend on collection order.
    """
    warnings.warn(
        "Pyramid consolidation still builds every level through dask by "
        f"default. In ngio={_DEFAULT_CHANGES_IN} the default for `mode` changes "
        'from `"dask"` to `"auto"`, which builds a small pyramid in memory '
        "instead -- 3-5x faster, at a peak of roughly 1.6x the source level. "
        'This image is inside that envelope. Pass `mode="auto"` to opt in now, '
        'or `mode="dask"` to keep the current behaviour and silence this.',
        NgioFutureWarning,
        stacklevel=stacklevel_of_first_caller(),
    )


def _resolve_mode(
    source: zarr.Array,
    plan: list[tuple[int, zarr.Array]],
    order: InterpolationOrder,
    mode: ConsolidationMode | None,
) -> Literal["dask", "numpy", "coarsen"]:
    if mode == "auto":
        return _resolve_auto_mode(source, plan, order)

    if mode is None:
        # An empty plan consolidates nothing, so the coming default cannot
        # change anything for this caller and the notice would be pure noise.
        if plan and _resolve_auto_mode(source, plan, order) == "numpy":
            _warn_default_will_change()
        return "dask"

    return mode


def _consolidate_on_disk(
    plan: list[tuple[int, zarr.Array]],
    nodes: list[zarr.Array],
    order: InterpolationOrder,
    mode: Literal["dask", "coarsen"],
) -> None:
    """One independent read -> zoom -> write per level, each its own compute.

    Level i+1 is written and then read back from the store to build level i+2,
    which costs 1.33x the pyramid's size in reads to produce 0.33x in writes.
    Fusing the levels into one graph removes that re-read, and was tried: it
    bought ~1% of wall clock (the re-read was already overlapped with parallel
    work, and the larger graph cost back what the IO saved), while roughly
    doubling peak memory and multiplying an already-untenable task count.

    The task count is the reason this stays serial. A dask graph costs ~2.4 KB
    and one task per chunk, so at 256x256 chunks a 100 GB image is ~820k tasks
    and ~1.7 GB of graph *before a byte is read* -- and fusing every level into
    one graph makes that 1.23M tasks and ~3 GB. Dask's own guidance is to stay
    under ~100k tasks, which this crosses at 8 GB either way. Keeping the levels
    separate at least bounds the graph by the largest single level rather than
    the whole pyramid. The real fix is not to build a graph per chunk at all.
    """
    for parent, target in plan:
        on_disk_zoom(source=nodes[parent], target=target, order=order, mode=mode)


def _consolidate_numpy(
    source: zarr.Array,
    plan: list[tuple[int, zarr.Array]],
    order: InterpolationOrder,
) -> None:
    levels = {0: _read_numpy(source)}

    for position, (parent, target) in enumerate(plan):
        out = numpy_zoom(levels[parent], target_shape=target.shape, order=order)
        target[...] = out
        levels[position + 1] = out

        # Release levels no later edge reads from. This mode already carries the
        # most memory of the three -- holding every level to the end would make
        # the worst case worse for no reason.
        still_needed = {p for p, _ in plan[position + 1 :]}
        for index in [i for i in levels if i not in still_needed]:
            del levels[index]


def consolidate_pyramid(
    source: zarr.Array,
    targets: list[zarr.Array],
    order: InterpolationOrder = "linear",
    mode: ConsolidationMode | None = None,
) -> None:
    """Consolidate the Zarr array.

    Args:
        source: The level to build the others from.
        targets: Every other level in the multiscale, above the source as well as
            below it -- levels above are upsampled, as they always have been.
        order: The interpolation order.
        mode: `"dask"`, `"numpy"` or `"coarsen"` to pick the path outright, or
            `"auto"` to take the in-memory path wherever it is bit-identical to
            the chunked one. `None` means the caller did not choose: it behaves
            as `"dask"` today and warns where `"auto"` would have differed.
    """
    for target in targets:
        if source.dtype != target.dtype:
            raise NgioValueError("source and target must have the same dtype")

    plan = _consolidation_plan(source, targets)

    match resolved := _resolve_mode(source, plan, order, mode):
        case "numpy":
            _consolidate_numpy(source, plan, order)
        case "dask" | "coarsen":
            _consolidate_on_disk(plan, [source, *(t for _, t in plan)], order, resolved)
        case _:
            raise NgioValueError(
                "mode must be either 'dask', 'numpy', 'coarsen' or 'auto'"
            )


################################################
#
# Builders for image pyramids
#
################################################

ChunksLike = tuple[int, ...] | Literal["auto"]
ShardsLike = tuple[int, ...] | Literal["auto"]


def compute_shapes_from_scaling_factors(
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
            max(1, math.floor(s / f))
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
    translation: tuple[float, ...]
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

        if len(self.translation) != len(self.shape):
            raise NgioValueError(
                "Translation must have the same length as shape "
                f"({len(self.shape)}), got {len(self.translation)}"
            )

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


def compute_scales_from_shapes(
    shapes: Sequence[tuple[int, ...]],
    base_scale: tuple[float, ...],
) -> list[tuple[float, ...]]:
    scales = [base_scale]
    scale_ = base_scale
    for current_shape, next_shape in itertools.pairwise(shapes):
        # This only works for downsampling pyramids
        # The _check_order function (called before) ensures that the
        # shapes are decreasing
        _scaling_factor = tuple(
            s1 / s2
            for s1, s2 in zip(
                current_shape,
                next_shape,
                strict=True,
            )
        )
        scale_ = tuple(s * f for s, f in zip(scale_, _scaling_factor, strict=True))
        scales.append(scale_)
    return scales


def _compute_translations_from_shapes(
    scales: Sequence[tuple[float, ...]],
    base_translation: Sequence[float] | None,
) -> list[tuple[float, ...]]:
    translations = []
    if base_translation is None:
        n_dim = len(scales[0])
        base_translation = tuple(0.0 for _ in range(n_dim))
    else:
        base_translation = tuple(base_translation)

    translation_ = base_translation
    for _ in scales:
        # TBD: How to update translation
        # For now, we keep it constant but we should probably change it
        # to reflect the shift introduced by downsampling
        # translation_ = translation_ + _scaling_factor
        translations.append(translation_)
    return translations


def _compute_scales_from_factors(
    base_scale: tuple[float, ...], scaling_factors: tuple[float, ...], num_levels: int
) -> list[tuple[float, ...]]:
    precision_scales = []
    current_scale = base_scale
    for _ in range(num_levels):
        precision_scales.append(current_scale)
        current_scale = tuple(
            s * f for s, f in zip(current_scale, scaling_factors, strict=True)
        )
    return precision_scales


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
    @deprecated_alias(levels_paths="level_paths")
    def from_scaling_factors(
        cls,
        level_paths: tuple[str, ...],
        scaling_factors: tuple[float, ...],
        base_shape: tuple[int, ...],
        base_scale: tuple[float, ...],
        axes: tuple[str, ...],
        base_translation: Sequence[float] | None = None,
        chunks: ChunksLike = "auto",
        shards: ShardsLike | None = None,
        data_type: str = "uint16",
        dimension_separator: Literal[".", "/"] = "/",
        compressors: Any = "auto",
        zarr_format: Literal[2, 3] = 2,
        other_array_kwargs: Mapping[str, Any] | None = None,
        precision_scale: bool = True,
    ) -> "ImagePyramidBuilder":
        # Since shapes needs to be rounded to integers, we compute them here
        # and then pass them to from_shapes
        # This ensures that the shapes and scaling factors are consistent
        # and avoids accumulation of rounding errors
        shapes = compute_shapes_from_scaling_factors(
            base_shape=base_shape,
            scaling_factors=scaling_factors,
            num_levels=len(level_paths),
        )

        if precision_scale:
            # Compute precise scales from shapes
            # Since shapes are rounded to integers, the scaling factors
            # may not be exactly the same as the input scaling factors
            # Thus, we compute the scales from the shapes to ensure consistency
            base_scale_ = compute_scales_from_shapes(
                shapes=shapes,
                base_scale=base_scale,
            )
        else:
            base_scale_ = _compute_scales_from_factors(
                base_scale=base_scale,
                scaling_factors=scaling_factors,
                num_levels=len(level_paths),
            )

        return cls.from_shapes(
            shapes=shapes,
            base_scale=base_scale_,
            axes=axes,
            base_translation=base_translation,
            level_paths=level_paths,
            chunks=chunks,
            shards=shards,
            data_type=data_type,
            dimension_separator=dimension_separator,
            compressors=compressors,
            zarr_format=zarr_format,
            other_array_kwargs=other_array_kwargs,
        )

    @classmethod
    @deprecated_alias(levels_paths="level_paths")
    def from_shapes(
        cls,
        shapes: Sequence[tuple[int, ...]],
        base_scale: tuple[float, ...] | list[tuple[float, ...]],
        axes: tuple[str, ...],
        base_translation: Sequence[float] | None = None,
        level_paths: Sequence[str] | None = None,
        chunks: ChunksLike = "auto",
        shards: ShardsLike | None = None,
        data_type: str = "uint16",
        dimension_separator: Literal[".", "/"] = "/",
        compressors: Any = "auto",
        zarr_format: Literal[2, 3] = 2,
        other_array_kwargs: Mapping[str, Any] | None = None,
    ) -> "ImagePyramidBuilder":
        levels = []
        if level_paths is None:
            level_paths = tuple(str(i) for i in range(len(shapes)))

        _check_order(shapes)
        if isinstance(base_scale, tuple) and all(
            isinstance(s, float) for s in base_scale
        ):
            scales = compute_scales_from_shapes(shapes, base_scale)
        elif isinstance(base_scale, list):
            scales = base_scale
            if len(scales) != len(shapes):
                raise NgioValueError(
                    "Scales must have the same length as shapes "
                    f"({len(shapes)}), got {len(scales)}"
                )
        else:
            raise NgioValueError(
                "base_scale must be either a tuple of floats or a list of tuples "
                " of floats."
            )

        translations = _compute_translations_from_shapes(scales, base_translation)
        for level_path, shape, scale, translation in zip(
            level_paths,
            shapes,
            scales,
            translations,
            strict=True,
        ):
            level = PyramidLevel(
                path=level_path,
                shape=shape,
                scale=scale,
                translation=translation,
                chunks=chunks,
                shards=shards,
            )
            levels.append(level)
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
        # Heterogeneous by construction, and `other_array_kwargs` lets callers
        # pass through any `create_array` parameter, so the values cannot be
        # narrowed to a useful union.
        array_static_kwargs: dict[str, Any] = {
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
