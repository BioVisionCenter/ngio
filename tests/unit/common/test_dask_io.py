from collections import Counter
from itertools import product

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
import zarr.storage
from dask.array.core import _get_zarr_write_chunks
from dask.utils import parse_bytes

from ngio.common._dask_io import (
    block_budget,
    store_dask,
    write_block,
    write_grid,
    write_unit,
    write_unit_bytes,
)
from ngio.config import DaskConfig, get_config
from ngio.utils import NgioValueError


def _array(tmp_path, name, **kwargs):
    return zarr.create_array(tmp_path / name, dtype="uint16", **kwargs)


def _memory_array(**kwargs):
    kwargs.setdefault("dtype", "uint16")
    return zarr.create_array(store=zarr.storage.MemoryStore(), **kwargs)


def _with_cap(monkeypatch, cap):
    """Pin `write_block_max_bytes` for one test. `None` removes the ceiling."""
    monkeypatch.setattr(get_config(), "dask", DaskConfig(write_block_max_bytes=cap))


def _resident(zarr_array):
    return tuple(
        min(u, s) for u, s in zip(write_unit(zarr_array), zarr_array.shape, strict=True)
    )


def _assert_grid_is_unit_aligned(grid, region, zarr_array):
    """Every interior block edge lands on a write-unit edge, in array coordinates.

    That is the whole safety argument for writing with `lock=False`: a unit with
    one writer cannot lose an update, however many workers run.
    """
    shape = zarr_array.shape
    if region is None:
        region = tuple(slice(0, s) for s in shape)
    for axis, (sizes, index, size, step) in enumerate(
        zip(grid, region, shape, write_unit(zarr_array), strict=True)
    ):
        start, stop, _ = index.indices(size)
        assert sum(sizes) == stop - start, f"axis {axis}: {sizes} != {start}:{stop}"
        assert all(s > 0 for s in sizes), f"axis {axis}: empty block in {sizes}"
        edge = start
        for s in sizes[:-1]:
            edge += s
            assert edge % step == 0, (
                f"axis {axis}: edge {edge} of grid {sizes} from {start} "
                f"splits a unit of {step}"
            )


class CountingStore(zarr.storage.WrapperStore):
    """Tally chunk-data reads per key.

    A read that returns data is a read-modify-write about to happen; two of them
    on one key from concurrent blocks is the lost-update race. Probes of keys
    that do not exist yet (`None`) are not counted: zarr's complete-shard write
    probed existing shards once up to 3.1.6 and stopped in 3.2.1, and that is
    upstream drift, not a property of ngio's grid.

    Only the async `get` is hooked: zarr's default pipeline reads through it,
    and the straddling-region test pins an exact non-zero count, so reads
    moving to another surface would fail there rather than pass vacuously.
    """

    def __init__(self, store):
        super().__init__(store)
        self.reads: Counter[str] = Counter()

    async def get(self, key, prototype, byte_range=None):
        buffer = await super().get(key, prototype, byte_range)
        if key.startswith("c/") and buffer is not None:
            self.reads[key] += 1
        return buffer


# ---------------------------------------------------------------------------
# write unit
# ---------------------------------------------------------------------------


def test_write_unit_is_the_shard_when_sharded(tmp_path):
    sharded = _array(
        tmp_path, "s.zarr", shape=(8, 64, 64), chunks=(1, 16, 16), shards=(4, 64, 64)
    )
    plain = _array(tmp_path, "p.zarr", shape=(8, 64, 64), chunks=(1, 16, 16))

    assert write_unit(sharded) == (4, 64, 64)
    assert write_unit(plain) == (1, 16, 16)
    # dask's own notion of the write unit; if they ever disagree, the two
    # libraries describe the store differently and the docs here are wrong.
    assert write_unit(sharded) == _get_zarr_write_chunks(sharded)
    assert write_unit(plain) == _get_zarr_write_chunks(plain)


# shape/chunks lifted from a real OME-Zarr array. The z extent of the chunk
# (10) overhangs the z extent of the array (1), so the declared chunk is
# 105.47 MiB while the largest chunk that can exist here is 10.55 MiB.
_OVERHANGING = {"shape": (3, 1, 19440, 20480), "chunks": (1, 10, 2160, 2560)}


def test_write_unit_bytes_clips_an_overhanging_chunk(tmp_path):
    array = _array(tmp_path, "real.zarr", **_OVERHANGING)

    declared = int(np.prod(array.chunks)) * array.dtype.itemsize
    assert declared == 110_592_000  # 105.47 MiB, and unreachable
    assert write_unit_bytes(array) == 11_059_200  # 10.55 MiB, (1, 1, 2160, 2560)
    # The grid itself is unchanged: only the byte figure was ever overstated.
    assert write_unit(array) == _OVERHANGING["chunks"]


@pytest.mark.parametrize("shards", [None, (4, 64, 64)])
def test_write_unit_bytes_is_the_plain_product_when_nothing_overhangs(tmp_path, shards):
    """No clipping where none is due -- the guard must not shrink an honest unit."""
    array = _array(
        tmp_path,
        f"fit-{shards}.zarr",
        shape=(8, 64, 64),
        chunks=(1, 16, 16),
        shards=shards,
    )
    unit = write_unit(array)
    assert write_unit_bytes(array) == int(np.prod(unit)) * array.dtype.itemsize


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_the_floor_wins_over_the_cap(tmp_path, monkeypatch):
    """A unit larger than the cap gets one unit per block, not a split one.

    The lost-update guard is `max(..., unit_bytes)` and it is applied last, so
    no configured cap can subdivide a write unit.
    """
    array = _array(
        tmp_path,
        "shard.zarr",
        shape=(4, 512, 512),
        chunks=(1, 128, 128),
        shards=(2, 512, 512),
    )
    unit_bytes = write_unit_bytes(array)
    assert unit_bytes == 1024 * 1024  # 1 MiB shard, deliberately above the caps below

    _with_cap(monkeypatch, unit_bytes // 8)
    assert block_budget(array) == unit_bytes
    assert write_block(array) == write_unit(array)

    # And the write still lands correctly rather than merely being budgeted.
    patch = np.arange(4 * 512 * 512, dtype="uint16").reshape(4, 512, 512)
    store_dask(da.from_array(patch, chunks=(1, 128, 128)), array)
    np.testing.assert_array_equal(array[...], patch)


def test_the_cap_never_raises_a_lower_budget_you_set(tmp_path, monkeypatch):
    """A ceiling, not a target: an `array.chunk-size` below the cap is kept."""
    array = _array(tmp_path, "low.zarr", shape=(64, 512, 512), chunks=(1, 256, 256))
    _with_cap(monkeypatch, 64 * 2**20)

    with dask.config.set({"array.chunk-size": "4MiB"}):
        assert block_budget(array) == 4 * 2**20


def test_no_cap_reproduces_the_dask_default(tmp_path, monkeypatch):
    array = _array(tmp_path, "none.zarr", shape=(64, 512, 512), chunks=(1, 256, 256))
    _with_cap(monkeypatch, None)

    assert block_budget(array) == parse_bytes(dask.config.get("array.chunk-size"))


def test_an_overhanging_chunk_does_not_inflate_the_block(tmp_path, monkeypatch):
    """The budget must not be raised for capacity no write can use.

    Measuring the declared extent would take the budget past dask's own default
    -- 1,055 MiB here -- and the block would grow to spend it. The ceiling is
    removed so this is about the floor alone.
    """
    _with_cap(monkeypatch, None)
    array = _array(
        tmp_path,
        "overhang.zarr",
        shape=(3, 1, 19440, 20480),
        chunks=(1, 100, 2160, 2560),
    )
    dask_default = parse_bytes(dask.config.get("array.chunk-size"))
    assert int(np.prod(array.chunks)) * array.dtype.itemsize > dask_default, (
        "fixture must overhang far enough to clear dask's default"
    )

    block = write_block(array)
    assert int(np.prod(block)) * array.dtype.itemsize <= dask_default
    for axis, (b, u) in enumerate(zip(block, _resident(array), strict=True)):
        assert b % u == 0, f"axis {axis}: block {b} is not whole units of {u}"


def test_the_cap_bounds_the_block(tmp_path, monkeypatch):
    """A block is capped, and is still a whole number of write units.

    Both halves matter: the cap is what bounds memory, the multiple is what
    keeps one writer per unit.
    """
    array = _array(tmp_path, "capped.zarr", shape=(64, 512, 512), chunks=(1, 256, 256))
    unit = write_unit(array)
    unit_bytes = write_unit_bytes(array)

    for cap in (unit_bytes, 2 * unit_bytes, 8 * 2**20, 64 * 2**20):
        _with_cap(monkeypatch, cap)
        block = write_block(array)
        assert int(np.prod(block)) * array.dtype.itemsize <= cap, f"cap {cap}"
        for axis, (extent, u) in enumerate(zip(block, unit, strict=True)):
            assert extent % u == 0, f"cap {cap}, axis {axis}: {extent} straddles {u}"


# ---------------------------------------------------------------------------
# block and grid
# ---------------------------------------------------------------------------


def test_small_units_are_glued_into_one_block():
    """Dask's heuristic must still engage: several units per block, not one.

    Alignment never depends on it (a non-multiple answer falls back to one
    unit per block), but a fallback here would silently turn every small-chunk
    write into one task per chunk. Pinned so a dask change fails CI instead.
    """
    array = _memory_array(shape=(1, 2, 10, 3000, 3000), chunks=(1, 1, 1, 256, 256))
    unit = write_unit(array)
    block = write_block(array)

    assert all(b % u == 0 for b, u in zip(block, unit, strict=True))
    assert int(np.prod(block)) // int(np.prod(unit)) >= 4, block
    assert int(np.prod(block)) * array.dtype.itemsize <= block_budget(array)


@pytest.mark.parametrize("z", [1, 3])
def test_an_overhanging_unit_is_one_block_on_that_axis(z):
    """Chunk z=10 over fewer planes: one block on z, heuristic still engaged."""
    array = _memory_array(shape=(1, 2, z, 4320, 3840), chunks=(1, 1, 10, 2160, 2560))
    block = write_block(array)

    assert block[2] == z
    assert block[3] % 2160 == 0 and block[4] % 2560 == 0
    grid = write_grid(array, None)
    assert grid[2] == (z,)
    _assert_grid_is_unit_aligned(grid, None, array)


# The geometry from the ome-zarr-converters-tools report: units at or above
# the 8 MiB default cap, on arrays that are not a multiple of the unit. Dask's
# own grid here is (1860, 1212) -- two writers on one chunk.
_HAZARD_UNITS = [
    # (chunks, shards, dtype) -> unit bytes >= 8 MiB
    ((1, 1, 1, 2048, 2048), None, "uint16"),
    ((1, 1, 1, 1024, 1024), None, "float64"),
    ((1, 1, 1, 1024, 1024), (1, 1, 1, 4096, 4096), "uint16"),
]


def _hazard_cases():
    for chunks, shards, dtype in _HAZARD_UNITS:
        unit = shards or chunks
        for c, fy, fx in product((1, 2, 4), range(8, 17, 2), range(8, 17, 2)):
            shape = (1, c, 1, unit[3] * fy // 8, unit[4] * fx // 8)
            yield pytest.param(
                shape, chunks, shards, dtype, id=f"{dtype}-{shape}-{unit[3:]}"
            )


@pytest.mark.parametrize("shape, chunks, shards, dtype", list(_hazard_cases()))
def test_the_grid_never_splits_a_unit(shape, chunks, shards, dtype):
    array = _memory_array(shape=shape, chunks=chunks, shards=shards, dtype=dtype)
    unit = write_unit(array)
    uy, ux = unit[3], unit[4]
    regions = [
        None,
        # starts and ends inside a unit on y and x
        (
            slice(0, 1),
            slice(0, shape[1]),
            slice(0, 1),
            slice(uy // 2, shape[3]),
            slice(ux // 4, shape[4] - 7),
        ),
        # a whole-unit field, offset by one unit
        (
            slice(0, 1),
            slice(0, shape[1]),
            slice(0, 1),
            slice(0, uy),
            slice(0, min(ux, shape[4])),
        ),
    ]
    block = write_block(array)
    assert all(b % u == 0 for b, u in zip(block, _resident(array), strict=True))
    budget = block_budget(array)
    assert int(np.prod(block)) * array.dtype.itemsize <= budget or block == _resident(
        array
    )
    for region in regions:
        _assert_grid_is_unit_aligned(write_grid(array, region), region, array)


def test_the_reported_shapes_get_one_unit_per_block():
    """At unit == budget, the block is the unit and the grid is cut on it."""
    for shape in [(1, 2, 1, 3072, 3072), (1, 2, 1, 3072, 2048), (1, 2, 3, 3072, 3072)]:
        array = _memory_array(shape=shape, chunks=(1, 1, 1, 2048, 2048))
        assert write_block(array) == (1, 1, 1, 2048, 2048)
        grid = write_grid(array, None)
        assert grid[3] == (2048, 1024) and grid[4][0] == 2048


def test_the_grid_rejects_stepped_slices_and_non_slices():
    array = _memory_array(shape=(8, 8), chunks=(4, 4))
    with pytest.raises(NgioValueError, match="stepped"):
        write_grid(array, (slice(0, 8, 2), slice(0, 8)))
    with pytest.raises(NgioValueError, match="tuple of slices"):
        write_grid(array, (3, slice(0, 8)))


def test_a_zero_length_axis_is_an_empty_write():
    """An empty table index copied through the zarr fallback has shape `(0,)`."""
    array = _memory_array(shape=(0, 4), chunks=(1, 4))
    assert write_grid(array, None) == ((0,), (4,))
    store_dask(da.zeros((0, 4), chunks=(1, 4), dtype="uint16"), array)

    array = _memory_array(shape=(8, 4), chunks=(4, 4))
    array[...] = 3
    region = (slice(3, 3), slice(0, 4))
    assert write_grid(array, region) == ((0,), (4,))
    store_dask(da.zeros((0, 4), chunks=(1, 4), dtype="uint16"), array, region)
    assert (array[...] == 3).all()


def test_store_dask_leaves_the_dask_config_alone():
    """The budget is passed to dask as an argument, never as a global scope."""
    array = _memory_array(shape=(4, 32, 32), chunks=(1, 32, 32))
    before = dask.config.get("array.chunk-size")
    with dask.config.set({"array.chunk-size": "1MiB"}):
        store_dask(da.zeros((4, 32, 32), chunks=(1, 32, 32), dtype="uint16"), array)
        assert dask.config.get("array.chunk-size") == "1MiB"
    assert dask.config.get("array.chunk-size") == before


# ---------------------------------------------------------------------------
# writes: data
# ---------------------------------------------------------------------------


def test_unit_larger_than_the_dask_budget_still_aligns(tmp_path):
    """A write unit above `array.chunk-size` must not produce straddling blocks."""
    array = _array(
        tmp_path,
        "big.zarr",
        shape=(4, 512, 512),
        chunks=(1, 128, 128),
        shards=(2, 512, 512),
    )
    unit_bytes = int(np.prod(write_unit(array))) * array.dtype.itemsize
    patch = da.from_array(np.ones((4, 512, 512), dtype="uint16"), chunks=(1, 128, 128))

    # A budget deliberately below one unit: the configuration the floor exists for.
    with dask.config.set({"array.chunk-size": unit_bytes // 4}):
        assert write_block(array) == write_unit(array)
        store_dask(patch, array)

    np.testing.assert_array_equal(array[...], np.ones((4, 512, 512), dtype="uint16"))


def test_overhanging_chunk_writes_correctly(tmp_path):
    """End to end on the overhanging shape, small enough to actually move the bytes."""
    shape, chunks = (4, 1, 64, 64), (1, 8, 32, 32)
    baseline = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape)

    reference = _array(tmp_path, "ref.zarr", shape=shape, chunks=chunks)
    reference[...] = baseline

    array = _array(tmp_path, "got.zarr", shape=shape, chunks=chunks)
    store_dask(da.from_array(baseline, chunks=(1, 1, 16, 16)), array)

    np.testing.assert_array_equal(array[...], reference[...])


def test_a_capped_write_still_matches_numpy(tmp_path, monkeypatch):
    """Batching is an allocation decision; it may not touch the bytes written."""
    baseline = np.arange(8 * 128 * 128, dtype="uint16").reshape(8, 128, 128)
    reference = _array(tmp_path, "ref2.zarr", shape=(8, 128, 128), chunks=(1, 64, 64))
    reference[...] = baseline

    array = _array(tmp_path, "got2.zarr", shape=(8, 128, 128), chunks=(1, 64, 64))
    _with_cap(monkeypatch, write_unit_bytes(array))  # one unit per block, the extreme
    store_dask(da.from_array(baseline, chunks=(1, 32, 32)), array)

    np.testing.assert_array_equal(array[...], reference[...])


def test_region_write_matches_numpy(tmp_path):
    """A region that covers no whole unit is still written correctly."""
    baseline = np.arange(4 * 32 * 32, dtype="uint16").reshape(4, 32, 32)
    region = (slice(1, 3), slice(5, 21), slice(5, 21))
    patch = np.full((2, 16, 16), 7, dtype="uint16")

    reference = _array(tmp_path, "ref.zarr", shape=(4, 32, 32), chunks=(2, 32, 32))
    reference[...] = baseline
    reference[region] = patch

    array = _array(tmp_path, "got.zarr", shape=(4, 32, 32), chunks=(2, 32, 32))
    array[...] = baseline
    store_dask(da.from_array(patch, chunks=(1, 8, 8)), array, region=region)

    np.testing.assert_array_equal(array[...], reference[...])


@pytest.mark.parametrize(
    "region",
    [
        None,
        (slice(0, 1), slice(0, 2), slice(0, 1), slice(0, 3072), slice(0, 3072)),
        (slice(0, 1), slice(0, 2), slice(0, 1), slice(1024, 2048), slice(1024, 2048)),
    ],
    ids=["whole-array", "whole-region", "field"],
)
def test_the_reported_geometry_round_trips(region):
    """The converter's case: 8 MiB chunks on a 1.5-unit image, dask patch.

    Before the fix, dask's grid put two writers on each chunk: 2.2M wrong
    pixels on the whole region, 0.7M on the field, and the whole-array write
    raised from the tripwire. Threaded scheduler so the race has every chance.
    """
    shape = (1, 2, 1, 3072, 3072)
    array = _memory_array(shape=shape, chunks=(1, 1, 1, 2048, 2048))
    rng = np.random.default_rng(0)
    full = rng.integers(0, 1000, shape, dtype="uint16")
    expected = full if region is None else full[region]
    patch = da.from_array(expected, chunks=(1, 1, 1, 1024, 1024))

    with dask.config.set(scheduler="threads", num_workers=4):
        store_dask(patch, array, region=region)

    got = array[...] if region is None else array[region]
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# writes: store reads -- the guarantee, not the symptom
# ---------------------------------------------------------------------------
#
# Zero chunk reads means no read-modify-write, so there is nothing a race could
# lose, on any machine at any worker count. One read per boundary unit means
# the unit has exactly one writer. A clean pixel comparison proves neither.


@pytest.mark.parametrize("shards", [None, (1, 64, 64)], ids=["chunks", "shards"])
def test_a_unit_aligned_write_reads_each_unit_at_most_once(shards):
    store = CountingStore(zarr.storage.MemoryStore())
    array = zarr.create_array(
        store=store,
        shape=(4, 128, 128),
        chunks=(1, 32, 32) if shards else (1, 64, 64),
        shards=shards,
        dtype="uint16",
    )
    array[...] = 1  # existing content, so a read-modify-write would have to read
    store.reads.clear()

    store_dask(da.ones((4, 128, 128), chunks=(1, 64, 64), dtype="uint16"), array)
    assert all(n == 1 for n in store.reads.values()), dict(store.reads)
    if shards is None:
        # A complete chunk write never reads. (A complete *shard* write still
        # probed the existing shard once on zarr <= 3.1.6; see CountingStore.)
        assert not store.reads

    store.reads.clear()
    region = (slice(1, 3), slice(64, 128), slice(0, 128))
    store_dask(da.ones((2, 64, 128), chunks=(1, 64, 64), dtype="uint16"), array, region)
    assert all(n == 1 for n in store.reads.values()), dict(store.reads)
    if shards is None:
        assert not store.reads


@pytest.mark.parametrize("shards", [None, (1, 64, 64)], ids=["chunks", "shards"])
def test_a_straddling_region_reads_each_boundary_unit_exactly_once(shards):
    """A region that starts inside a unit: the case `rechunk(unit)` alone corrupts.

    Rechunking to the unit lands the blocks on a grid offset from the array's,
    so every block straddles two units. The grid is cut in array coordinates
    instead, so the two partial y-units are each read once, by their one block.
    """
    store = CountingStore(zarr.storage.MemoryStore())
    array = zarr.create_array(
        store=store,
        shape=(4, 128, 128),
        chunks=(1, 32, 32) if shards else (1, 64, 64),
        shards=shards,
        dtype="uint16",
    )
    baseline = np.arange(4 * 128 * 128, dtype="uint16").reshape(4, 128, 128)
    array[...] = baseline
    store.reads.clear()

    region = (slice(1, 3), slice(40, 100), slice(0, 128))
    patch = np.full((2, 60, 128), 7, dtype="uint16")
    with dask.config.set(scheduler="threads", num_workers=4):
        store_dask(da.from_array(patch, chunks=(1, 20, 32)), array, region=region)

    # z in {1, 2} x y-units {0, 1} x x-units {0, 1}: eight boundary units.
    assert len(store.reads) == 8, dict(store.reads)
    assert all(n == 1 for n in store.reads.values()), dict(store.reads)
    expected = baseline.copy()
    expected[region] = patch
    np.testing.assert_array_equal(array[...], expected)
