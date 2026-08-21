import itertools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import dask.array as da
import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, model_validator

from ngio.common._dask_io import RegionType, store_dask
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
)
from ngio.utils._warnings import stacklevel_of_first_caller

ConsolidationMode: TypeAlias = Literal["dask", "numpy", "coarsen", "auto"]
"""How `consolidate` builds each level. `"auto"` takes the in-memory path only
where it is bit-identical to the chunked one; see `_resolve_auto_mode`."""

RegionsLike: TypeAlias = Sequence[tuple[RegionType, ...]]
"""Where the source level changed, for `consolidate(regions=...)`: one on-disk
index tuple per touched region, in the source array's axis order — exactly
what a setter's `slicing_ops.normalized_slicing_tuple` produces."""

#: A region as per-axis `[start, stop)` pixel bounds — the internal canonical
#: form every `RegionsLike` input is normalized to.
PixelRegion: TypeAlias = tuple[tuple[int, int], ...]

#: The release in which `mode` starts defaulting to `"auto"`.
_DEFAULT_CHANGES_IN = "1.2"


def _read_numpy(source: zarr.Array) -> np.ndarray:
    source_array = source[...]
    if not isinstance(source_array, np.ndarray):
        raise NgioValueError("source zarr array could not be read as a numpy array")
    return source_array


def _coarsen_expr(
    source_array: da.Array,
    target_shape: tuple[int, ...],
    order: InterpolationOrder = "linear",
    aggregation_function: Callable | None = None,
) -> da.Array:
    """Coarsen a dask array onto a target shape, as an unevaluated expression.

    Args:
        source_array: The source array to coarsen.
        target_shape: The shape the result must match.
        order: Not really implemented for coarsening, kept for compatibility with
            the zoom function.
            order="linear" -> linear interpolation ~ np.mean
            order="nearest" -> nearest interpolation ~ np.max
        aggregation_function: The aggregation function to use.
    """
    _scale, _target_shape = _zoom_inputs_check(
        source_array=source_array, scale=None, target_shape=target_shape
    )

    if _target_shape != target_shape:
        raise NgioValueError(
            f"Coarsening would produce shape {_target_shape}, but the target "
            f"array has shape {target_shape}."
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
                f"{target_shape[i]}: coarsening only downsamples. Use "
                'mode="dask" or mode="numpy" to build a level larger than its '
                "source."
            )
        coarsening_setup[i] = factor

    return da.coarsen(
        aggregation_function, source_array, coarsening_setup, trim_excess=True
    )


def _zoom_expr(
    source_array: da.Array,
    target_shape: tuple[int, ...],
    target_dtype: np.dtype,
    order: InterpolationOrder,
    mode: Literal["dask", "coarsen"],
    aggregation_function: Callable | None = None,
) -> da.Array:
    """One pyramid level (or a region of one) as an unevaluated dask expression.

    The `astype` is not cosmetic. `da.coarsen(np.mean, ...)` promotes an integer
    source to float64; written to the store that float is cast back, so the next
    level down reads the cast value. An expression handed straight to the next
    level would instead propagate the float, and a mean pyramid built that way
    differs from a stored one in 28% of level 2 and 63% of level 3. Casting here
    is what the store would have done anyway, just done early — which also drops
    the float64 intermediate and makes coarsening markedly lighter.
    """
    if mode == "coarsen":
        out = _coarsen_expr(source_array, target_shape, order, aggregation_function)
    else:
        out = dask_zoom(source_array, target_shape=target_shape, order=order)

    if out.dtype != target_dtype:
        out = out.astype(target_dtype)
    return out


def _on_disk_numpy_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    target[...] = numpy_zoom(
        _read_numpy(source), target_shape=target.shape, order=order
    )


def _read_source_dask(source: zarr.Array) -> da.Array:
    """The source level as a dask array, fetched at its write unit.

    On a sharded source, blocks sized to the inner chunks make every dask
    task a *partial-shard* read — a shard-index fetch plus a range request
    per inner chunk. Reading whole shards costs one coarse fetch each.

    The immediate rechunk back to the inner chunks is load-bearing, not a
    nicety: `dask_zoom` derives its block grid from the array's chunksize and
    zooms block-locally, so handing it shard-sized blocks would change output
    pixels outside the integral-downsample/nearest-linear envelope
    (`test_consolidation_matches_a_per_level_reference` pins this). Shard →
    inner-chunk rechunk is pure views — no extra IO. Unsharded sources take
    the plain path, expression unchanged.
    """
    if source.shards is None:
        return da.from_zarr(source)
    return da.from_zarr(source, chunks=source.shards).rechunk(source.chunks)


def _on_disk_dask_zoom(
    source: zarr.Array,
    target: zarr.Array,
    order: InterpolationOrder,
) -> None:
    # No compute_chunk_sizes() here: it would execute the whole read -> zoom
    # graph purely to re-learn block shapes, throw the pixels away, and leave
    # the write to run the same graph again — exactly double the chunk reads.
    # No rechunk onto the target either: store_dask rechunks onto the target's
    # write unit, which is `shards or chunks`. Rechunking to `target.chunks`
    # here was worse than doing nothing on a sharded target — that is the
    # shard's *inner* chunk shape, so every block became a partial shard write.
    store_dask(
        _zoom_expr(
            _read_source_dask(source), target.shape, target.dtype, order, "dask"
        ),
        target,
    )


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
            _read_source_dask(source),
            target.shape,
            target.dtype,
            order,
            "coarsen",
            aggregation_function,
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

    The same greedy nearest-shape walk consolidation has always done — same
    `_find_closest_arrays`, same sequence — it just records the edges instead of
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
    integer — 2.004, which is what a 501-pixel axis gives — and they disagree
    on 99.9% of the level; upsample instead and they disagree on 1.6%.
    """
    return all(
        t >= 1 and s >= t and s % t == 0
        for s, t in zip(source_shape, target_shape, strict=True)
    )


def _normalize_regions(
    regions: RegionsLike, shape: tuple[int, ...]
) -> list[PixelRegion]:
    """Turn caller-facing region tuples into clamped `[start, stop)` bounds.

    An `int` selects one index, a `list[int]` its bounding box — over-covering
    a fancy selection is safe here, since recomputing an untouched pixel just
    rewrites the value it already had. Empty selections drop out.
    """
    out: list[PixelRegion] = []
    for region in regions:
        if len(region) != len(shape):
            raise NgioValueError(
                f"Region {region} has {len(region)} axes, but the source "
                f"array has {len(shape)}."
            )
        bounds = []
        for selection, dim in zip(region, shape, strict=True):
            if isinstance(selection, slice):
                start, stop, step = selection.indices(dim)
                if step != 1:
                    raise NgioValueError(
                        f"Region {region} uses a stepped slice; regions must "
                        "be contiguous."
                    )
            elif isinstance(selection, int):
                start = selection + dim if selection < 0 else selection
                stop = start + 1
            elif isinstance(selection, list):
                if not selection:
                    start, stop = 0, 0
                else:
                    start, stop = min(selection), max(selection) + 1
            else:
                raise NgioValueError(
                    f"Region {region} contains {selection!r}; each axis must "
                    "be a slice, an int, or a list of ints."
                )
            bounds.append((max(0, start), min(dim, stop)))
        if all(stop > start for start, stop in bounds):
            out.append(tuple(bounds))
    return out


def _regions_touch(a: PixelRegion, b: PixelRegion) -> bool:
    """Whether two half-open boxes overlap or share a face (adjacency counts)."""
    return all(
        a_start <= b_stop and b_start <= a_stop
        for (a_start, a_stop), (b_start, b_stop) in zip(a, b, strict=True)
    )


def _merge_pass(regions: list[PixelRegion]) -> list[PixelRegion]:
    """One sweep of union-find over touching regions, components to boxes."""
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Sweep on the most discriminating axis, as `check_if_regions_overlap`
    # does: candidates for a touch test are only the regions still "active"
    # (not yet ended) when a region starts, which is near-linear for the
    # scattered boxes setters and ROI tables produce.
    axis = max(
        range(len(regions[0])),
        key=lambda ax: len({region[ax][0] for region in regions}),
    )
    order = sorted(range(len(regions)), key=lambda i: regions[i][axis][0])
    active: list[int] = []
    for i in order:
        start = regions[i][axis][0]
        active = [j for j in active if regions[j][axis][1] >= start]
        for j in active:
            if _regions_touch(regions[i], regions[j]):
                parent[find(i)] = find(j)
        active.append(i)

    boxes: dict[int, PixelRegion] = {}
    for i, region in enumerate(regions):
        root = find(i)
        if root not in boxes:
            boxes[root] = region
        else:
            boxes[root] = tuple(
                (min(a_start, b_start), max(a_stop, b_stop))
                for (a_start, a_stop), (b_start, b_stop) in zip(
                    boxes[root], region, strict=True
                )
            )
    return sorted(boxes.values())


def _merge_regions(regions: list[PixelRegion]) -> list[PixelRegion]:
    """Merge touching regions into disjoint bounding boxes.

    Collapsing a component to its box can create new touches, so passes repeat
    until the count is stable. Boxes over-cover an L-shaped component, which is
    safe — the recomputed extra pixels land on the values they already had --
    and the coverage bail-out in `_plan_partial` caps how far that can degrade.
    """
    while len(regions) > 1:
        merged = _merge_pass(regions)
        if len(merged) == len(regions):
            return merged
        regions = merged
    return regions


def _coverage(regions: list[PixelRegion], shape: tuple[int, ...]) -> float:
    """Fraction of the array the (disjoint, merged) regions cover."""
    total = math.prod(shape)
    if total == 0:
        return 0.0
    covered = sum(
        math.prod(stop - start for start, stop in region) for region in regions
    )
    return covered / total


def _propagate_regions(
    parent_regions: list[PixelRegion],
    factors: tuple[int, ...],
    target: zarr.Array,
) -> list[PixelRegion]:
    """Map a parent level's written regions onto a target level, snapped.

    The affected target pixels are `[start // f, ceil(stop / f))` per axis;
    snapping outward to the target's write unit (`shards or chunks`) makes
    every write a whole-unit write with a single writer, and — because the
    source window is then `region * f`, an exact multiple — keeps the
    blockwise zoom inside the envelope where it matches the whole-array one.
    """
    units = target.shards or target.chunks
    propagated = []
    for region in parent_regions:
        propagated.append(
            tuple(
                (
                    (start // f // unit) * unit,
                    min(dim, math.ceil(math.ceil(stop / f) / unit) * unit),
                )
                for (start, stop), f, unit, dim in zip(
                    region, factors, units, target.shape, strict=True
                )
            )
        )
    return _merge_regions(propagated)


def _plan_partial(
    source: zarr.Array,
    plan: list[tuple[int, zarr.Array]],
    merged: list[PixelRegion],
    order: InterpolationOrder,
) -> dict[int, list[PixelRegion]] | None:
    """Regions to rebuild per pyramid node, or `None` for a full rebuild.

    `merged` is the source level's touched regions, already normalized and
    merged (`consolidate_pyramid` owns that step, and the empty case: no
    touched regions consolidate nothing rather than falling back here).

    The region path is only taken where it is bytes-identical to a full
    rebuild: every edge an integral downsample and `order` in `{"nearest",
    "linear"}` — the same envelope `"auto"` trusts, for the same reason
    (`_is_integral_downsample`). A source coverage above
    `ConsolidationConfig.partial_max_coverage` also declines: past that,
    region bookkeeping costs more than the rebuild it saves.

    Keys index `[source, *targets in plan order]`, matching the plan's parent
    indices.
    """
    if order == "cubic":
        return None
    if (
        _coverage(merged, source.shape)
        > get_config().consolidation.partial_max_coverage
    ):
        return None

    node_regions: dict[int, list[PixelRegion]] = {0: merged}
    node_shapes: list[tuple[int, ...]] = [source.shape]
    for position, (parent, target) in enumerate(plan):
        if not _is_integral_downsample(node_shapes[parent], target.shape):
            return None
        factors = tuple(
            s // t for s, t in zip(node_shapes[parent], target.shape, strict=True)
        )
        node_regions[position + 1] = _propagate_regions(
            node_regions[parent], factors, target
        )
        node_shapes.append(target.shape)
    return node_regions


def _consolidate_regions(
    plan: list[tuple[int, zarr.Array]],
    nodes: list[zarr.Array],
    node_regions: dict[int, list[PixelRegion]],
    order: InterpolationOrder,
    mode: Literal["dask", "numpy", "coarsen"],
) -> None:
    """Rebuild only the planned regions, edge by edge in plan order.

    Parents come before children in the plan, so every source window reads
    pixels that are already consistent: inside the parent's rebuilt region
    they were just written, outside it they were valid before the writes --
    the precondition of a partial consolidation. That is also why the numpy
    mode reads each parent window from *disk* rather than chaining levels in
    memory: outward snapping at the child can reach past what the parent
    rebuilt, and only the store holds those pixels.
    """
    for position, (parent, target) in enumerate(plan):
        parent_array = nodes[parent]
        factors = tuple(
            s // t for s, t in zip(parent_array.shape, target.shape, strict=True)
        )
        for region in node_regions[position + 1]:
            source_slices = tuple(
                slice(start * f, stop * f)
                for (start, stop), f in zip(region, factors, strict=True)
            )
            target_slices = tuple(slice(start, stop) for start, stop in region)
            region_shape = tuple(stop - start for start, stop in region)
            if mode == "numpy":
                patch = parent_array[source_slices]
                if not isinstance(patch, np.ndarray):
                    raise NgioValueError(
                        "source zarr array could not be read as a numpy array"
                    )
                target[target_slices] = numpy_zoom(
                    patch, target_shape=region_shape, order=order
                )
            else:
                store_dask(
                    _zoom_expr(
                        _read_source_dask(parent_array)[source_slices],
                        region_shape,
                        target.dtype,
                        order,
                        mode,
                    ),
                    target,
                    region=target_slices,
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
        # no halo is wrong at every block boundary — the paths differ on 10-16%
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
    only here is the plan known — and the warning is worth nothing to someone
    whose pyramid `"auto"` would decline anyway. Deduplication is left to the
    `warnings` module, which suppresses repeats per call site. A once-per-process
    flag would be worse: under `filterwarnings = ["error"]` only the first caller
    raises, so which test fails would depend on collection order.
    """
    warnings.warn(
        "Pyramid consolidation still builds every level through dask by "
        f"default. In ngio={_DEFAULT_CHANGES_IN} the default for `mode` changes "
        'from `"dask"` to `"auto"`, which builds a small pyramid in memory '
        "instead — 3-5x faster, at a peak of roughly 1.6x the source level. "
        'This image is inside that envelope. Pass `mode="auto"` (called '
        "`consolidation_mode` on creation helpers and iterators) to opt in "
        'now, or `mode="dask"` to keep the current behaviour and silence this.',
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
    and ~1.7 GB of graph *before a byte is read* — and fusing every level into
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
        # most memory of the three — holding every level to the end would make
        # the worst case worse for no reason.
        still_needed = {p for p, _ in plan[position + 1 :]}
        for index in [i for i in levels if i not in still_needed]:
            del levels[index]


def consolidate_pyramid(
    source: zarr.Array,
    targets: list[zarr.Array],
    order: InterpolationOrder = "linear",
    mode: ConsolidationMode | None = None,
    regions: RegionsLike | None = None,
) -> None:
    """Consolidate the Zarr array.

    Args:
        source: The level to build the others from.
        targets: Every other level in the multiscale, above the source as well as
            below it — levels above are upsampled, as they always have been.
        order: The interpolation order.
        mode: `"dask"`, `"numpy"` or `"coarsen"` to pick the path outright, or
            `"auto"` to take the in-memory path wherever it is bit-identical to
            the chunked one. `None` means the caller did not choose: it behaves
            as `"dask"` today and warns where `"auto"` would have differed.
        regions: Where the source level changed, as on-disk index tuples in
            the source's axis order (what a setter's
            `slicing_ops.normalized_slicing_tuple` produces). Only the
            pyramid regions derived from them are rebuilt, bit-identical to
            a full rebuild; empty regions rebuild nothing. Where the region
            path is not exact (a non-integral downsample edge,
            `order="cubic"`), or past
            `ConsolidationConfig.partial_max_coverage` of the source, the
            whole pyramid is rebuilt instead. `None` rebuilds everything.
    """
    for target in targets:
        if source.dtype != target.dtype:
            raise NgioValueError("source and target must have the same dtype")

    plan = _consolidation_plan(source, targets)

    merged: list[PixelRegion] | None = None
    if regions is not None:
        merged = _merge_regions(_normalize_regions(regions, source.shape))
        if not merged:
            # Nothing was touched, so nothing derives from it. Returning
            # before the mode resolution also keeps the `mode=None` future
            # warning quiet — same reasoning as the empty-plan case there.
            return

    resolved = _resolve_mode(source, plan, order, mode)

    if merged is not None:
        node_regions = _plan_partial(source, plan, merged, order)
        if node_regions is not None:
            nodes = [source, *(target for _, target in plan)]
            _consolidate_regions(plan, nodes, node_regions, order, resolved)
            return

    match resolved:
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
            if self.chunks == "auto":
                # zarr infers "auto" chunks from the full array shape and then
                # requires shard % chunk == 0 — it does not auto-chunk within
                # the shard — so this combination fails deep in zarr, after
                # the group metadata is on disk.
                raise NgioValueError(
                    f"An explicit shard shape {self.shards} needs an explicit "
                    "chunk shape: with chunks='auto' zarr infers chunks from "
                    "the array shape and then rejects a shard that is not a "
                    "whole multiple of them. Pass `chunks=` as well, or use "
                    "shards='auto'."
                )
            if len(self.shards) != len(self.shape):
                raise NgioValueError(
                    "Shards must have the same length as shape "
                    f"({len(self.shape)}), got {len(self.shards)}"
                )
            normalized_shards = []
            for axis, (dim_size, shard_size) in enumerate(
                zip(self.shape, self.shards, strict=True)
            ):
                shard_size = min(dim_size, shard_size)
                if isinstance(self.chunks, tuple):
                    # zarr requires the shard extent to be a whole multiple of
                    # the chunk extent, so a clip that lands between multiples
                    # rounds down to one — never below a single chunk.
                    chunk_size = self.chunks[axis]
                    shard_size = max(chunk_size, shard_size - shard_size % chunk_size)
                normalized_shards.append(shard_size)
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

    @model_validator(mode="after")
    def _require_v3_for_shards(self) -> "ImagePyramidBuilder":
        # Every array creation funnels through this builder, and it is
        # constructed before anything touches the store — so this is the one
        # place a v2+shards mistake can fail *before* a half-built container
        # (group + OME metadata, no arrays) lands on disk.
        if self.zarr_format != 2:
            return self
        for level in self.levels:
            if level.shards == "auto":
                # The only valid v2 meaning of "let ngio decide": no sharding.
                level.shards = None
            elif level.shards is not None:
                raise NgioValueError(
                    "Sharding requires zarr format 3 (OME-Zarr >= 0.5); got "
                    f"shards={level.shards!r} for level {level.path!r}. Pass "
                    'shards=None, or create with ngff_version="0.5".'
                )
        return self

    @classmethod
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
