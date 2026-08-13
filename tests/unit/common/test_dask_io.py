import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from dask.array.core import _get_zarr_write_chunks, normalize_chunks
from dask.utils import parse_bytes

from ngio.common._dask_io import store_dask, write_unit, write_unit_bytes


def _array(tmp_path, name, **kwargs):
    return zarr.create_array(tmp_path / name, dtype="uint16", **kwargs)


def _block(zarr_array):
    """The block shape `to_zarr` would rechunk onto, under the current config."""
    return tuple(
        c[0]
        for c in normalize_chunks(
            "auto",
            shape=zarr_array.shape,
            dtype=zarr_array.dtype,
            previous_chunks=_get_zarr_write_chunks(zarr_array),
        )
    )


def test_write_unit_is_the_shard_when_sharded(tmp_path):
    sharded = _array(
        tmp_path, "s.zarr", shape=(8, 64, 64), chunks=(1, 16, 16), shards=(4, 64, 64)
    )
    plain = _array(tmp_path, "p.zarr", shape=(8, 64, 64), chunks=(1, 16, 16))

    assert write_unit(sharded) == (4, 64, 64)
    assert write_unit(plain) == (1, 16, 16)
    # The rule ngio relies on is dask's own; if they ever disagree, the write
    # path is aligning to a grid dask is not.
    assert write_unit(sharded) == _get_zarr_write_chunks(sharded)
    assert write_unit(plain) == _get_zarr_write_chunks(plain)


@pytest.mark.parametrize("shards", [None, (4, 64, 64)])
def test_blocks_never_share_a_write_unit(tmp_path, shards):
    """No two dask blocks may land in one write unit.

    That is the whole safety argument for writing with `lock=False`: a unit with
    one writer cannot lose an update, however many workers run. Asserted on the
    block grid rather than on the data, because a data check only fails when the
    race happens to be lost on that run.
    """
    array = _array(
        tmp_path, "a.zarr", shape=(8, 64, 64), chunks=(1, 16, 16), shards=shards
    )
    unit = write_unit(array)
    patch = da.from_array(
        np.arange(8 * 64 * 64, dtype="uint16").reshape(8, 64, 64), chunks=(1, 16, 16)
    )
    store_dask(patch, array)

    blocks = _get_zarr_write_chunks(array)
    for axis, (block, extent) in enumerate(zip(blocks, unit, strict=True)):
        assert block % extent == 0, f"axis {axis}: block {block} straddles {extent}"


def test_unit_larger_than_the_dask_budget_still_aligns(tmp_path):
    """A write unit above `array.chunk-size` must not produce straddling blocks.

    Without the guard in `store_dask`, `normalize_chunks("auto", ...)` cannot
    reach one whole unit inside the budget and falls back to blocks that cross
    unit boundaries -- which `to_zarr` then writes with `lock=False`, i.e. two
    writers on one shard.
    """
    array = _array(
        tmp_path,
        "big.zarr",
        shape=(4, 512, 512),
        chunks=(1, 128, 128),
        shards=(2, 512, 512),
    )
    unit_bytes = int(np.prod(write_unit(array))) * array.dtype.itemsize
    patch = da.from_array(np.ones((4, 512, 512), dtype="uint16"), chunks=(1, 128, 128))

    # A budget deliberately below one unit: the configuration the guard exists for.
    with dask.config.set({"array.chunk-size": unit_bytes // 4}):
        store_dask(patch, array)

    np.testing.assert_array_equal(array[...], np.ones((4, 512, 512), dtype="uint16"))


def test_the_dask_budget_is_restored(tmp_path):
    array = _array(tmp_path, "r.zarr", shape=(4, 32, 32), chunks=(1, 32, 32))
    before = dask.config.get("array.chunk-size")
    store_dask(da.zeros((4, 32, 32), chunks=(1, 32, 32), dtype="uint16"), array)
    assert dask.config.get("array.chunk-size") == before


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


def test_an_overhanging_chunk_does_not_inflate_the_budget(tmp_path, monkeypatch):
    """The budget must not be raised for capacity no write can use.

    Measuring the declared extent takes the budget past dask's own default once
    the overhang is large enough -- 1,055 MiB here -- and `normalize_chunks`
    spends every byte of it, turning an 84 MiB block into a 791 MiB one. No
    write is performed: `to_zarr` is stubbed so the assertion is about the
    budget `store_dask` establishes, not about moving 2 GB.
    """
    array = _array(
        tmp_path,
        "overhang.zarr",
        shape=(3, 1, 19440, 20480),
        chunks=(1, 100, 2160, 2560),
    )
    assert int(np.prod(array.chunks)) * array.dtype.itemsize > parse_bytes(
        dask.config.get("array.chunk-size")
    ), "fixture must overhang far enough to clear dask's default"

    untouched = _block(array)  # what dask picks when nothing raises the budget
    seen = {}

    def _stub(arr, z, region=None, **kwargs):
        seen["budget"] = parse_bytes(dask.config.get("array.chunk-size"))
        seen["block"] = _block(z)

    monkeypatch.setattr(da, "to_zarr", _stub)
    store_dask(da.zeros(array.shape, chunks=array.chunks, dtype="uint16"), array)

    assert seen["budget"] == parse_bytes(dask.config.get("array.chunk-size"))
    assert seen["block"] == untouched
    assert int(np.prod(seen["block"])) * array.dtype.itemsize <= parse_bytes("128MiB")


def test_overhanging_chunk_writes_correctly(tmp_path):
    """End to end on the same shape, small enough to actually move the bytes."""
    shape, chunks = (4, 1, 64, 64), (1, 8, 32, 32)
    baseline = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape)

    reference = _array(tmp_path, "ref.zarr", shape=shape, chunks=chunks)
    reference[...] = baseline

    array = _array(tmp_path, "got.zarr", shape=shape, chunks=chunks)
    store_dask(da.from_array(baseline, chunks=(1, 1, 16, 16)), array)

    np.testing.assert_array_equal(array[...], reference[...])
    # Still one writer per unit, measured against the unit that can exist.
    resident = [min(u, s) for u, s in zip(write_unit(array), shape, strict=True)]
    for axis, (block, extent) in enumerate(zip(_block(array), resident, strict=True)):
        assert block % extent == 0, f"axis {axis}: block {block} straddles {extent}"


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
