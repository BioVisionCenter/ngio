"""Op counting stays intact across zarr 3.3's second IO surface.

zarr 3.3 added a coalescing `get_ranges` and a synchronous store surface
(`get_sync`/`set_sync`/`delete_sync`). `WrapperStore` forwards all of them
straight to the wrapped store, so a read that switches surfaces would tally as
zero without the hooks in `CountingNgioStore` — a phantom improvement in the
baseline diff, which is precisely what the gate exists to prevent.

Nothing in `scenarios.py` reaches either surface today: ngio's fixtures are
unsharded, and `BatchedCodecPipeline` is still zarr's default. These tests
therefore drive both deliberately, so the hooks do not rot unnoticed until the
day zarr flips a default.
"""

import pytest
import zarr
from tests.performance._counting import count, counting_store, zero_fill
from zarr.abc.store import Store

from ngio import open_ome_zarr_container
from ngio.images._create_synt_container import create_synthetic_ome_zarr

pytestmark = pytest.mark.skipif(
    not hasattr(Store, "get_ranges"),
    reason="get_ranges and the sync store surface arrived in zarr 3.3",
)

_BATCHED = "zarr.core.codec_pipeline.BatchedCodecPipeline"
_FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"


def _image(shards=None):
    store = counting_store({})
    create_synthetic_ome_zarr(
        store,
        shape=(1, 64, 64),
        levels=1,
        chunks=(1, 16, 16),
        shards=shards,
        ngff_version="0.5",
    )
    return open_ome_zarr_container(store).get_image()


def _read_tally(image, extent=32):
    with count() as counters:
        image.get_array(x=slice(0, extent), y=slice(0, extent))
    return zero_fill(counters)


def test_sharded_read_counts_get_ranges():
    """A partial read of a sharded array routes through `Store.get_ranges`.

    The read must stay *inside* one shard: asking for a whole shard makes zarr
    fetch it in one piece and never reach the range coalescer.
    """
    tally = _read_tally(_image(shards=(1, 32, 32)), extent=16)

    assert tally["get_ranges"] > 0
    # The coalescer runs over `self.get`, so the merged fetches are counted
    # there rather than being folded into the batch counter.
    assert tally["get.chunk"] > 0
    assert tally["get.partial"] > 0
    assert tally["bytes.read"] > 0


def test_sync_pipeline_tallies_the_same_as_the_async_one():
    """Chunk IO lands in the same counters whichever surface zarr picks.

    `FusedCodecPipeline` reads through `get_sync` instead of `get`. Asserting
    equality rather than merely "non-zero" is the point: it pins the sync hooks
    to the *same* counter keys as their async twins, so switching pipelines can
    never silently move counts between keys.
    """
    with zarr.config.set({"codec_pipeline.path": _BATCHED}):
        batched = _read_tally(_image())
    with zarr.config.set({"codec_pipeline.path": _FUSED}):
        fused = _read_tally(_image())

    assert fused == batched
    assert fused["get.chunk"] > 0  # guards against both being vacuously empty


def test_sync_pipeline_counts_writes():
    """`set_sync` writes tally as `set.chunk` / `bytes.written`, like `set`."""
    with zarr.config.set({"codec_pipeline.path": _FUSED}):
        image = _image()
        array = image.get_array(x=slice(0, 32), y=slice(0, 32))
        with count() as counters:
            image.set_array(array, x=slice(0, 32), y=slice(0, 32))
        tally = zero_fill(counters)

    assert tally["set.chunk"] > 0
    assert tally["bytes.written"] > 0
