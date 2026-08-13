"""Relational invariants over the committed baselines.

The exact-count gate can be regenerated wholesale with `--update-baseline`,
which makes every number look like ordinary churn in review. The properties
that actually matter are *relationships between* scenarios — cached costs less
than uncached, a sharded write reads nothing back, `auto` resolution is free —
and these assert them directly on the committed files, so a regeneration that
destroys one fails here instead of merging as numeric noise.
"""

import pytest
from tests.performance._baseline import load_baseline
from tests.performance.conftest import STORE_KINDS


@pytest.fixture(params=STORE_KINDS)
def scenarios(request):
    data = load_baseline(request.param)
    if data is None:
        pytest.skip(f"no committed baseline for the {request.param!r} store")
    return data["scenarios"]


def test_cached_is_strictly_cheaper(scenarios):
    """`cache=True` must cost strictly fewer metadata reads than `cache=False`.

    These pairs were byte-for-byte identical while the cache flag was inert;
    their inequality is the recorded form of the fix.
    """
    for cached, uncached in [
        ("open_container_cached", "open_container"),
        ("dimensions_x10_cached", "dimensions_x10"),
        ("plate_images_paths_cached", "plate_images_paths"),
    ]:
        assert scenarios[cached]["get.meta"] < scenarios[uncached]["get.meta"], (
            f"{cached} no longer beats {uncached}"
        )


def test_sharded_write_reads_no_chunk_data_back(scenarios):
    """No chunk data read back means no read-modify-write — a safety property.

    A partial-shard write is not merely slow: two of them racing on one shard
    lose an update. The unsharded control performs no chunk reads at all; the
    sharded write may only *probe* shards that do not exist yet, so its byte
    count must not exceed the control's (metadata-only) bytes.
    """
    assert scenarios["write_dask"]["get.chunk"] == 0
    assert (
        scenarios["write_sharded_dask"]["bytes.read"]
        <= scenarios["write_dask"]["bytes.read"]
    )


def test_auto_consolidation_resolution_is_free(scenarios):
    """`mode="auto"` resolves to numpy on this geometry and must match it.

    The resolution inspects already-open arrays and the config, so it is not
    allowed to cost a single store operation over the mode it picks.
    """
    assert scenarios["consolidate_auto"] == scenarios["consolidate_numpy"]


def test_channel_selection_metadata_is_flat_in_call_count(scenarios):
    """Repeating a channel-selecting read must not repeat the metadata cost.

    The channel metadata is cached against the metadata generation; only the
    chunk reads may scale with the call count. A per-ROI loop with a
    `channel_selection` is the ordinary shape of this.
    """
    x1 = scenarios["read_channel_selection"]
    x3 = scenarios["read_channel_selection_x3"]
    assert x3["get.meta"] == x1["get.meta"]
    assert x3["get.chunk"] == 3 * x1["get.chunk"]
