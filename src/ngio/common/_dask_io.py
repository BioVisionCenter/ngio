"""The one way ngio writes a dask array into an existing zarr array."""

import warnings
from typing import TypeAlias

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.core import PerformanceWarning, normalize_chunks
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


def write_unit_bytes(zarr_array: zarr.Array) -> int:
    """Bytes of one write unit *as it can exist in this array*.

    A chunk or shard extent may exceed the array extent it spans — chunks
    `(1, 10, 2160, 2560)` over a single-z image is an ordinary OME-Zarr shape.
    Only one, partial, unit can exist on such an axis, so the declared extent
    overstates what a budget has to hold: 105 MiB against a realisable 10.5 MiB
    on that geometry.

    Clipping is not a refinement, it is what the consumer already does.
    `normalize_chunks` clips `previous_chunks` to the shape before it sizes a
    block, so the grid the write actually lands on is built from the clipped
    unit — and a budget raised to the declared one raises it past anything the
    write needs.
    """
    unit = write_unit(zarr_array)
    resident = [
        min(extent, size) for extent, size in zip(unit, zarr_array.shape, strict=True)
    ]
    return int(np.prod(resident)) * zarr_array.dtype.itemsize


def block_budget(zarr_array: zarr.Array) -> int:
    """The `array.chunk-size` to build one write block under, in bytes.

    `to_zarr` sizes its blocks with `normalize_chunks("auto", previous_chunks=
    unit)`, which grows a block in whole multiples of the write unit until it
    approaches this budget. Bounded on both sides, and the order matters --
    read it outward from `write_unit_bytes`:

    - **Floor, applied last.** One whole unit must fit. Below that,
      `normalize_chunks` cannot reach a multiple of the unit and falls back to
      blocks that *straddle* one -- which `to_zarr` then writes with
      `lock=False`, i.e. several writers on one shard and a lost update. A
      256 MiB shard against the 128 MiB default is the case. Because the floor
      is applied last it cannot be undercut by the ceiling, so the safety
      property holds at any configured cap.
    - **Ceiling.** Dask's default is 128 MiB against a unit that is typically a
      few hundred KiB, so it packs ~1000 units into one resident block for
      nothing. Peak memory is roughly blocks-in-flight times block size; at the
      8 MiB default this is a 75% cut on a 4 GB pyramid (565 -> 140 MB) for
      +0.37% task count and no measurable wall clock. See `DaskConfig`.

    A cap above the unit is therefore inert by construction: a coarse geometry
    -- a 105 MiB shard, say -- gets exactly one unit per block whatever the cap
    says, which is already the smallest block that can be written safely.
    """
    budget = parse_bytes(dask.config.get("array.chunk-size"))
    cap = get_config().dask.write_block_max_bytes
    if cap is not None:
        budget = min(budget, cap)
    return max(budget, write_unit_bytes(zarr_array))


def _require_write_unit_alignment(zarr_array: zarr.Array) -> None:
    """Refuse a whole-array write whose block grid would double-write a unit.

    Replicates the grid `da.to_zarr` rechunks to (`normalize_chunks` under the
    active `array.chunk-size` budget, so call this inside the budget scope)
    and requires every *interior* block boundary to land on a write-unit
    boundary — the condition that gives each chunk or shard exactly one
    writer under `lock=False`. `block_budget`'s floor makes this hold today;
    the check is the tripwire if a dask upgrade ever changes the grid, since
    the write would otherwise lose updates silently.
    """
    unit = write_unit(zarr_array)
    grid = normalize_chunks(
        "auto",
        shape=zarr_array.shape,
        dtype=zarr_array.dtype,
        previous_chunks=unit,
    )
    for axis, (sizes, step) in enumerate(zip(grid, unit, strict=True)):
        boundary = 0
        for size in sizes[:-1]:
            boundary += size
            if boundary % step:
                raise NgioValueError(
                    f"Refusing a parallel dask write: along axis {axis} the "
                    f"block grid {sizes} splits a write unit (size {step}) "
                    "between two writers, which can silently lose updates. "
                    "This indicates a dask chunk-normalization change; please "
                    "report it to ngio."
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

    `da.to_zarr` rechunks the input onto the target's write-unit grid first, cut
    to the region on that same grid, so every unit is touched by exactly one
    block. That is both the speed fix and the reason no lock is needed: the
    contention is structurally absent rather than serialised away. The one case
    where it does not hold is a write unit larger than dask's `array.chunk-size`
    budget, which `block_budget` removes by raising the budget for the call --
    the same function that caps it from above, so blocks stay bounded.

    A region that does not cover whole units still costs a read-modify-write on
    its boundary units — unavoidable, and safe, since each has a single writer.

    Serial callers only: the budget is applied through a process-global
    `dask.config` scope, so two concurrent `store_dask` calls can run under
    each other's budget. The dask iterator verbs enforce this by rejecting
    parallel mappers.

    Args:
        patch: The data to write.
        zarr_array: The array to write into. Must already exist.
        region: Where to write, or `None` for the whole array.
    """
    if region is not None:
        for index in region:
            if isinstance(index, slice) and index.step not in (None, 1):
                # The single-writer-per-unit argument above assumes the region
                # is a contiguous cut of the unit grid; a stepped slice is not.
                raise NgioValueError(
                    f"store_dask does not support stepped slices, got {index}."
                )

    # Both bounds, and why each exists, live in `block_budget`.
    budget = block_budget(zarr_array)

    with dask.config.set({"array.chunk-size": budget}), warnings.catch_warnings():
        if region is None:
            _require_write_unit_alignment(zarr_array)
        # Dask's rechunk warning is muted because it is wrong in both
        # directions here. With a region, it reports that the region does not
        # cover whole units — a ROI narrower than a chunk, say. That
        # read-modify-write is real and unavoidable, but it is not a hazard:
        # the region is cut on the array's own block grid, so each boundary
        # unit has exactly one writer, and no `array.chunk-size` makes a
        # sub-chunk ROI cover a whole chunk.
        # Whole-array: it fires on a single whole-axis block whose
        # extent is not a unit multiple, which has no interior boundary and
        # is single-writer-safe; the genuine hazard it exists for is caught
        # precisely, and loudly, by `_require_write_unit_alignment` above.
        # Silencing per write, not the condition: `filterwarnings("error")`
        # downstream would otherwise make ordinary writes raise. Matched on
        # the message, so dask's other `PerformanceWarning`s still surface.
        warnings.filterwarnings(
            "ignore",
            message="The input Dask array will be rechunked",
            category=PerformanceWarning,
        )
        da.to_zarr(patch, zarr_array, region=region)
