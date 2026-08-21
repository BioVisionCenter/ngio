"""Deterministic repro for the `da.store` chunk write race.

A region write whose start is not chunk-aligned puts two dask blocks inside the
same zarr chunk, and `da.store` gives each its own read-modify-write of it — so
one update is lost, on a schedule nobody controls. `GatedStore` below holds the
first reader of the contested chunk until a second one arrives, which turns that
race from luck into a certainty.

ngio writes through `da.to_zarr`, which rechunks the patch onto the target's
write-unit grid before storing, so the contention this fixture constructs cannot
exist: the chunk has exactly one writer. The gate therefore never gets its second
arrival and times out, and *that is the pass condition*. A second arrival means
two blocks are read-modify-writing one chunk again — the alignment regressed.
"""

import asyncio
import contextlib
from typing import Any

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
import zarr.storage

from ngio.io_pipes._ops_slices import (
    SlicingOps,
    set_slice_as_dask,
    set_slice_as_numpy,
)

# Long enough that a genuinely concurrent second reader always arrives inside
# it, short enough not to slow the suite down when nothing does. Paid in full on
# every run now that the aligned write never produces one — one second, for a
# test that fails the moment the write path stops aligning.
_GATE_TIMEOUT_S = 1.0

# The single chunk both blocks partially cover, in zarr v3 default chunk key
# encoding.
_CONTESTED_KEY = "c/0"


class GatedStore(zarr.storage.WrapperStore):
    """Hold the first read of the contested chunk until a second one arrives."""

    def __init__(self, store: zarr.storage.MemoryStore) -> None:
        super().__init__(store)
        self.armed = False
        self.arrivals = 0
        # Created lazily: it must belong to zarr's I/O event loop.
        self._second_reader: asyncio.Event | None = None

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        if self.armed and key == _CONTESTED_KEY:
            if self._second_reader is None:
                self._second_reader = asyncio.Event()
            self.arrivals += 1
            if self.arrivals == 1:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._second_reader.wait(), timeout=_GATE_TIMEOUT_S
                    )
            else:
                self._second_reader.set()
        return await super().get(key, prototype, byte_range)


@pytest.fixture
def slicing_ops() -> SlicingOps:
    # Chunk 0 spans [0, 16) and chunk 1 spans [16, 32). Writing [4, 20) with
    # 8-element dask blocks puts block 0 at [4, 12) and block 1 at [12, 20):
    # both partially cover chunk 0. That is the *input* blocking; an aligned
    # write path merges the two before touching the store.
    return SlicingOps(
        on_disk_axes=("x",),
        on_disk_shape=(64,),
        on_disk_chunks=(16,),
        slicing_tuple=(slice(4, 20),),
    )


def test_misaligned_dask_write_matches_numpy(slicing_ops: SlicingOps) -> None:
    baseline = np.arange(1000, 1064, dtype="uint16")
    patch = np.arange(1, 17, dtype="uint16")

    reference = zarr.create_array(
        store=zarr.storage.MemoryStore(), shape=(64,), chunks=(16,), dtype="uint16"
    )
    reference[:] = baseline
    set_slice_as_numpy(zarr_array=reference, patch=patch, slicing_ops=slicing_ops)

    gated = GatedStore(zarr.storage.MemoryStore())
    array = zarr.create_array(store=gated, shape=(64,), chunks=(16,), dtype="uint16")
    array[:] = baseline

    gated.armed = True
    with dask.config.set(scheduler="threads", num_workers=4):
        set_slice_as_dask(
            zarr_array=array,
            patch=da.from_array(patch, chunks=(8,)),
            slicing_ops=slicing_ops,
        )
    gated.armed = False

    # Exactly one reader: the single block that owns the chunk after the
    # alignment rechunk. Two or more means the chunk is being read-modify-written
    # by concurrent blocks again. Zero means the chunk key drifted and the gate
    # never fired, which would make the equality below vacuous.
    assert gated.arrivals == 1
    np.testing.assert_array_equal(array[:], reference[:])
