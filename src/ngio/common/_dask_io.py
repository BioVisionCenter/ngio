"""The one way ngio writes a dask array into an existing zarr array."""

import warnings
from typing import TypeAlias

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.core import PerformanceWarning
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
        # What remains after the guard is dask reporting that the *region* does
        # not cover whole units -- a ROI narrower than a chunk, say. That read-
        # modify-write is real and unavoidable, but it is not a hazard: the
        # region is cut on the array's own block grid, so each boundary unit has
        # exactly one writer (`test_dask_write_race.py` pins that). The warning's
        # own remedy does not apply either, since no `array.chunk-size` makes a
        # sub-chunk ROI cover a whole chunk. Silencing an unactionable warning
        # per write, not the condition: `filterwarnings("error")` downstream
        # would otherwise make ordinary ROI writes raise. Matched on the
        # message, so dask's other `PerformanceWarning`s still surface.
        warnings.filterwarnings(
            "ignore",
            message="The input Dask array will be rechunked",
            category=PerformanceWarning,
        )
        da.to_zarr(patch, zarr_array, region=region)
