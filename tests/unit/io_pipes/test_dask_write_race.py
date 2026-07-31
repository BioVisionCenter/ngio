"""Deterministic repro for the `da.store(lock=False)` chunk write race.

A misaligned dask region write makes two dask blocks read-modify-write the
same zarr chunk. The `GatedStore` below forces both reads of the contested
chunk to complete before either write, so the lost update manifests on every
run — no sleeps, no scheduling luck.
"""

import asyncio
import contextlib

import dask
import dask.array as da
import numpy as np
import zarr
import zarr.storage

from ngio.io_pipes._ops_slices import (
    SlicingOps,
    set_slice_as_dask,
    set_slice_as_numpy,
)


class GatedStore(zarr.storage.WrapperStore):
    """Force two concurrent RMWs of one chunk to both read before either writes."""

    def __init__(self, store: zarr.storage.MemoryStore) -> None:
        super().__init__(store)
        self.armed = False
        self.arrivals = 0
        self._event: asyncio.Event | None = None  # created lazily on zarr's I/O loop

    async def get(self, key, prototype, byte_range=None):
        if self.armed and key == "c/0":
            if self._event is None:
                self._event = asyncio.Event()
            self.arrivals += 1
            if self.arrivals == 1:
                # Wait for the second reader; under a properly locked da.store
                # it can never arrive while we hold the chunk, so time out.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._event.wait(), timeout=1.0)
            else:
                self._event.set()
        return await super().get(key, prototype, byte_range)


def test_misaligned_dask_write_matches_numpy() -> None:
    baseline = np.arange(1000, 1064, dtype="uint16")
    patch = np.arange(1, 17, dtype="uint16")
    slicing_ops = SlicingOps(
        on_disk_axes=("x",),
        on_disk_shape=(64,),
        on_disk_chunks=(16,),
        slicing_tuple=(slice(4, 20),),
    )

    ref_arr = zarr.create_array(
        store=zarr.storage.MemoryStore(), shape=(64,), chunks=(16,), dtype="uint16"
    )
    ref_arr[:] = baseline
    set_slice_as_numpy(zarr_array=ref_arr, patch=patch, slicing_ops=slicing_ops)

    gated = GatedStore(zarr.storage.MemoryStore())
    arr = zarr.create_array(store=gated, shape=(64,), chunks=(16,), dtype="uint16")
    arr[:] = baseline

    # Block 0 covers [4:12] (chunk 0 partial); block 1 covers [12:20]
    # (chunk 0 partial and chunk 1 partial) -> both RMW chunk "c/0".
    gated.armed = True
    with dask.config.set(scheduler="threads", num_workers=4):
        set_slice_as_dask(
            zarr_array=arr,
            patch=da.from_array(patch, chunks=(8,)),
            slicing_ops=slicing_ops,
        )
    gated.armed = False

    # Both pre- and post-fix the contested chunk is read twice; anything less
    # means the chunk key drifted and the gate went vacuous.
    assert gated.arrivals >= 2
    np.testing.assert_array_equal(arr[:], ref_arr[:])
