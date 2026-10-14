"""The one way ngio writes a dask array into an existing zarr array."""

from itertools import pairwise
from typing import TypeAlias

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.utils import parse_bytes

from ngio.config import get_config
from ngio.utils import NgioValueError

# Structurally `io_pipes._ops_slices.SlicingType`, restated rather than imported:
# that module is a consumer of this one.
RegionType: TypeAlias = slice | list[int] | int


def write_unit(zarr_array: zarr.Array) -> tuple[int, ...]:
    """The smallest region zarr can write without reading first.

    The shard when the array is sharded, the chunk otherwise. Writing anything
    less than a whole unit forces zarr to read, decode, merge, re-encode and
    rewrite it.
    """
    return zarr_array.shards or zarr_array.chunks


def _resident_unit(zarr_array: zarr.Array) -> tuple[int, ...]:
    """The write unit clipped to the array: the largest unit that can exist.

    Never below 1, so a zero-length axis (an empty table index, say) still has
    a grid; there is nothing to write on it, and `da.store` accepts that.
    """
    return tuple(
        max(1, min(extent, size))
        for extent, size in zip(write_unit(zarr_array), zarr_array.shape, strict=True)
    )


def write_unit_bytes(zarr_array: zarr.Array) -> int:
    """Bytes of one write unit *as it can exist in this array*.

    A chunk or shard extent may exceed the array extent it spans — chunks
    `(1, 10, 2160, 2560)` over a single-z image is an ordinary OME-Zarr shape.
    Only one, partial, unit can exist on such an axis, so the declared extent
    overstates what a budget has to hold: 105 MiB against a realisable 10.5 MiB
    on that geometry. The block grid is built from the same clipped unit, so a
    budget raised to the declared one would be raised past anything the write
    needs.
    """
    return int(np.prod(_resident_unit(zarr_array))) * zarr_array.dtype.itemsize


def block_budget(zarr_array: zarr.Array) -> int:
    """How many bytes one write block may assemble in memory.

    Capped from above by `DaskConfig.write_block_max_bytes` (bounding peak
    memory), floored — last, so no cap can undercut it — at one whole write
    unit, the smallest block that can be written without sharing a unit.
    """
    budget = parse_bytes(dask.config.get("array.chunk-size"))
    cap = get_config().dask.write_block_max_bytes
    if cap is not None:
        budget = min(budget, cap)
    return max(budget, write_unit_bytes(zarr_array))


def write_block(zarr_array: zarr.Array) -> tuple[int, ...]:
    """The block one dask task writes: whole write units, sized under the budget.

    Dask's `normalize_chunks("auto", previous_chunks=...)` is only a byte-size
    and aspect-ratio heuristic; it never promised to land on unit boundaries,
    and on an array that is not a multiple of its unit it does not (it sizes
    against the *median* of the clipped unit tuple, then balances the axis).
    So it is asked on a shape rounded up to whole units, where every previous
    chunk is a full unit and its aggregation branch returns unit multiples —
    and its answer is used only if it is one. Otherwise one unit per block,
    which is always safe and still one task per unit.

    The unit is clipped to the array first: an overhanging chunk is one
    partial unit on that axis, and the declared extent would push dask into
    its shrinking branch.
    """
    unit = _resident_unit(zarr_array)
    padded = tuple(
        max(u, -(-s // u) * u) for s, u in zip(zarr_array.shape, unit, strict=True)
    )
    auto = normalize_chunks(
        "auto",
        shape=padded,
        limit=block_budget(zarr_array),
        dtype=zarr_array.dtype,
        previous_chunks=unit,
    )
    block = tuple(int(sizes[0]) for sizes in auto)
    if any(b % u for b, u in zip(block, unit, strict=True)):
        return unit
    return block


def _normalize_region(
    region: tuple[RegionType, ...] | None, shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """`region` as slices with concrete, in-range bounds; the whole array for `None`."""
    if region is None:
        return tuple(slice(0, size) for size in shape)
    slices = []
    for index, size in zip(region, shape, strict=True):
        if not isinstance(index, slice):
            # Ints and index lists are rewritten into slices by the slice
            # setters before they reach this module.
            raise NgioValueError(
                f"store_dask expects a tuple of slices as region, got {index!r}."
            )
        if index.step not in (None, 1):
            # The single-writer-per-unit argument assumes the region is a
            # contiguous cut of the unit grid; a stepped slice is not.
            raise NgioValueError(
                f"store_dask does not support stepped slices, got {index}."
            )
        start, stop, _ = index.indices(size)
        slices.append(slice(start, max(start, stop)))
    return tuple(slices)


def write_grid(
    zarr_array: zarr.Array, region: tuple[RegionType, ...] | None
) -> tuple[tuple[int, ...], ...]:
    """Block sizes for `region` whose interior edges all lie on write-unit edges.

    The region is cut at every multiple of `write_block` that falls strictly
    inside it, in *array* coordinates. Each unit is then touched by exactly one
    block, which is what makes a lock-free `da.store` safe: a unit with one
    writer cannot lose an update. A region that starts or ends inside a unit
    gives that unit a partial block, still a single writer -- its
    read-modify-write is unavoidable but not a hazard.
    """
    block = write_block(zarr_array)
    grid = []
    for index, b in zip(
        _normalize_region(region, zarr_array.shape), block, strict=True
    ):
        start, stop = index.start, index.stop
        first_inner = -(-start // b) * b
        if first_inner == start:
            first_inner += b
        edges = [start, *range(first_inner, stop, b), stop]
        grid.append(tuple(e1 - e0 for e0, e1 in pairwise(edges)))
    return tuple(grid)


def _require_write_unit_alignment(
    grid: tuple[tuple[int, ...], ...],
    region: tuple[slice, ...],
    zarr_array: zarr.Array,
) -> None:
    """Refuse a block grid that would give a write unit two writers.

    Checked in array coordinates: region start plus the cumulative block sizes.
    `write_grid` holds this by construction, so this is the invariant the
    write path is built on, kept cheap and run on every write.
    """
    for axis, (sizes, index, step) in enumerate(
        zip(grid, region, write_unit(zarr_array), strict=True)
    ):
        edge = start = index.start
        for size in sizes[:-1]:
            edge += size
            if edge % step:
                raise NgioValueError(
                    f"Refusing a parallel dask write: along axis {axis} the "
                    f"block grid {sizes} starting at {start} splits a write "
                    f"unit (size {step}) between two writers, which can "
                    "silently lose updates. Please report this to ngio."
                )


def store_dask(
    patch: da.Array,
    zarr_array: zarr.Array,
    region: tuple[RegionType, ...] | None = None,
) -> None:
    """Write a dask array into an existing zarr array, aligned to its write unit.

    The write unit of a zarr array is `shards or chunks`, never `chunks` alone:
    zarr can only skip the read-modify-write when a write covers a *whole* unit.
    `da.store` issues one `zarr.Array.__setitem__` per dask block, so on a
    sharded target a block covering one of the shard's inner chunks makes zarr
    read, decode, merge, re-encode and rewrite the entire shard — and two such
    blocks racing on one shard silently lose an update.

    The patch is therefore rechunked onto a grid ngio builds from whole write
    units (`write_grid`), cut to the region on the array's own unit boundaries,
    so every unit has exactly one writer. That is both the speed fix and the
    reason no lock is needed: the contention is structurally absent rather
    than serialised away. Dask's own chunk heuristic only decides how many
    units go into one block (`write_block`); it is never trusted for alignment.

    A region that does not cover whole units still costs a read-modify-write on
    its boundary units — unavoidable, and safe, since each has a single writer.

    Args:
        patch: The data to write.
        zarr_array: The array to write into. Must already exist.
        region: Where to write, as a tuple of slices, or `None` for the whole
            array.
    """
    slices = _normalize_region(region, zarr_array.shape)
    grid = write_grid(zarr_array, slices)
    _require_write_unit_alignment(grid, slices, zarr_array)
    da.store(
        patch.rechunk(grid),
        zarr_array,  # ty: ignore[invalid-argument-type]  # dask's stub omits zarr
        regions=[slices],
        lock=False,
    )
