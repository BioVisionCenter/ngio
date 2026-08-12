import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from dask.array.core import _get_zarr_write_chunks

from ngio.common._dask_io import store_dask, write_unit


def _array(tmp_path, name, **kwargs):
    return zarr.create_array(tmp_path / name, dtype="uint16", **kwargs)


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
    patch = da.from_array(
        np.ones((4, 512, 512), dtype="uint16"), chunks=(1, 128, 128)
    )

    # A budget deliberately below one unit: the configuration the guard exists for.
    with dask.config.set({"array.chunk-size": unit_bytes // 4}):
        store_dask(patch, array)

    np.testing.assert_array_equal(array[...], np.ones((4, 512, 512), dtype="uint16"))


def test_the_dask_budget_is_restored(tmp_path):
    array = _array(tmp_path, "r.zarr", shape=(4, 32, 32), chunks=(1, 32, 32))
    before = dask.config.get("array.chunk-size")
    store_dask(da.zeros((4, 32, 32), chunks=(1, 32, 32), dtype="uint16"), array)
    assert dask.config.get("array.chunk-size") == before


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
