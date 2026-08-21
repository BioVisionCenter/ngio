import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr

from ngio.common._pyramid import (
    InterpolationOrder,
    PyramidLevel,
    _consolidation_plan,
    _find_closest_arrays,
    _resolve_auto_mode,
    consolidate_pyramid,
    on_disk_zoom,
)
from ngio.common._zoom import numpy_zoom
from ngio.config import ConsolidationConfig, get_config
from ngio.utils import NgioFutureWarning, NgioValueError


@pytest.mark.parametrize(
    "order, mode",
    [
        ("nearest", "dask"),
        ("linear", "dask"),
        ("nearest", "numpy"),
        ("linear", "numpy"),
        ("nearest", "coarsen"),
        ("linear", "coarsen"),
    ],
)
def test_on_disk_zooms(
    tmp_path: Path, order: InterpolationOrder, mode: Literal["dask", "numpy", "coarsen"]
):
    """Every mode halves y and x, and the values are the ones the mode promises.

    This used to assert nothing at all -- it only checked that the call did not
    raise, which is how `mode="coarsen"` shipped for a while silently averaging
    label IDs. The ratio here is an exact 2, so the blockwise and whole-array
    zooms agree and a single in-memory reference pins all three modes.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(16, 128, 128), dtype="uint8")

    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(16, 128, 128), dtype="uint8"
    )
    source_array[...] = data
    target_array = zarr.create_array(
        tmp_path / "target.zarr", shape=(16, 64, 64), dtype="uint8"
    )

    on_disk_zoom(source_array, target_array, order=order, mode=mode)

    blocked = data.reshape(16, 64, 2, 64, 2)
    if mode == "coarsen":
        aggregated = blocked.max(axis=(2, 4)) if order == "nearest" else None
        expected = aggregated if aggregated is not None else blocked.mean(axis=(2, 4))
        expected = expected.astype("uint8")
    else:
        expected = numpy_zoom(data, target_shape=(16, 64, 64), order=order)

    np.testing.assert_array_equal(target_array[...], expected)


def test_coarsen_nearest_takes_the_max(tmp_path: Path):
    """`order="nearest"` must take the max over each block, never the mean.

    `Label.consolidate()` passes `order="nearest"` precisely so a label pyramid
    keeps IDs that were really segmented. Averaging invents ones that were not —
    the mean of labels 3 and 7 is 5 — and truncates on an integer dtype, so this
    asserts the blockwise max exactly rather than only that the call returns.
    """
    labels = np.array(
        [
            [3, 3, 0, 0],
            [3, 7, 0, 9],
            [4, 4, 8, 8],
            [4, 4, 8, 2],
        ],
        dtype="uint16",
    )
    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(1, 4, 4), dtype="uint16"
    )
    source_array[...] = labels[np.newaxis]

    target_array = zarr.create_array(
        tmp_path / "target.zarr", shape=(1, 2, 2), dtype="uint16"
    )
    on_disk_zoom(source_array, target_array, order="nearest", mode="coarsen")

    # Blockwise max. The mean would give [[4, 2], [4, 6]] — three of the four
    # cells wrong, and 6 is a label that appears nowhere in the source.
    np.testing.assert_array_equal(
        target_array[...], np.array([[[7, 9], [4, 8]]], dtype="uint16")
    )
    assert set(np.unique(target_array[...])) <= set(np.unique(labels))


def _reference_consolidate(
    source: zarr.Array,
    targets: list[zarr.Array],
    order: InterpolationOrder,
    mode: Literal["dask", "numpy", "coarsen"],
) -> None:
    """One independent read-zoom-write per level, spelled out.

    `consolidate_pyramid` does this today, so the comparison below looks
    tautological. It is not, and it is the reason this test exists: chaining the
    levels through memory instead -- one dask graph, one compute, no store round
    trip between levels -- was implemented and reverted, and it does *not*
    produce these bytes unless two things are done exactly right.

    `fast_zoom` runs per block with no halo, so a level's boundary artifacts
    depend on the block grid it is handed. Reading a level back gives it the
    on-disk chunks; carrying a dask expression forward gives it
    `block_output_shape`, which halves every level -- order="cubic" then drifts
    on 13% of pixels at level 2 and order="linear" on 99.7% once the scaling
    ratio stops being dyadic. Separately, `da.coarsen(np.mean, ...)` promotes to
    float64, and a stored level has been cast back by the store while a carried
    expression has not.

    So this guards a future re-attempt: anything that stops writing each level
    to the store before reading it for the next has to reproduce these bytes.
    """
    processed = [source]
    to_be_processed = list(targets)
    while to_be_processed:
        source_id, target_id = _find_closest_arrays(processed, to_be_processed)
        target = to_be_processed.pop(target_id)
        on_disk_zoom(source=processed[source_id], target=target, mode=mode, order=order)
        processed.append(target)


def _pyramid_levels(
    root: Path,
    tag: str,
    shapes: Sequence[tuple[int, ...]],
    chunks: tuple[int, ...],
    dtype: str,
    shards: tuple[int, ...] | None,
) -> list[zarr.Array]:
    return [
        zarr.create_array(
            root / f"{tag}_{i}.zarr",
            shape=shape,
            chunks=tuple(min(s, c) for s, c in zip(shape, chunks, strict=True)),
            shards=(
                None
                if shards is None
                else tuple(min(s, v) for s, v in zip(shape, shards, strict=True))
            ),
            dtype=dtype,
        )
        for i, shape in enumerate(shapes)
    ]


# `odd` is the load-bearing one. On a dyadic pyramid a fused chain drifts from a
# stored one only at order="cubic"; once the scaling ratio stops being an exact 2
# the drift reaches 99.7% of pixels at order="linear". A matrix without it passes
# whether or not the inter-level rechunk is present.
_GEOMETRIES = {
    "pow2": (
        [(1, 4, 512, 512), (1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)],
        (1, 1, 128, 128),
        None,
    ),
    "odd": (
        [(1, 4, 501, 501), (1, 4, 250, 250), (1, 4, 125, 125), (1, 4, 62, 62)],
        (1, 1, 128, 128),
        None,
    ),
    "sharded": (
        [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64), (1, 4, 32, 32)],
        (1, 1, 32, 32),
        (1, 1, 128, 128),
    ),
}


@pytest.mark.parametrize("geometry", sorted(_GEOMETRIES))
@pytest.mark.parametrize(
    "mode, order",
    [
        ("dask", "nearest"),
        ("dask", "linear"),
        ("dask", "cubic"),
        ("numpy", "nearest"),
        ("numpy", "linear"),
        ("numpy", "cubic"),
        ("coarsen", "nearest"),
        ("coarsen", "linear"),
    ],
)
@pytest.mark.parametrize("source_level", [0, 1])
def test_consolidation_matches_a_per_level_reference(
    tmp_path: Path,
    geometry: str,
    mode: Literal["dask", "numpy", "coarsen"],
    order: InterpolationOrder,
    source_level: int,
):
    """Multi-level consolidation matches level-at-a-time consolidation exactly.

    See `_reference_consolidate` for why this is not tautological. The cells that
    matter are `cubic` on any geometry and `linear` on `odd`: those are the ones
    that move the moment a level stops being written and re-read between steps.

    `source_level=1` covers consolidating from a middle level, where the plan
    branches both down and *up* and the levels above get upsampled.
    """
    if mode == "coarsen" and source_level != 0:
        pytest.skip(
            "coarsen cannot upsample; covered by test_coarsen_refuses_to_upsample"
        )

    shapes, chunks, shards = _GEOMETRIES[geometry]
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=shapes[source_level], dtype="uint16")

    results = {}
    for tag, consolidate in (
        ("reference", _reference_consolidate),
        ("actual", consolidate_pyramid),
    ):
        levels = _pyramid_levels(tmp_path, tag, shapes, chunks, "uint16", shards)
        levels[source_level][...] = data
        targets = [level for i, level in enumerate(levels) if i != source_level]
        consolidate(levels[source_level], targets, order=order, mode=mode)
        results[tag] = [level[...] for level in levels]

    for i, (reference, actual) in enumerate(
        zip(results["reference"], results["actual"], strict=True)
    ):
        np.testing.assert_array_equal(actual, reference, err_msg=f"level {i} differs")


@pytest.mark.parametrize(
    "shapes, order, expected, reason",
    [
        ([(1, 4, 256, 256), (1, 4, 128, 128)], "linear", "numpy", "exact ratio 2"),
        ([(1, 4, 300, 300), (1, 4, 100, 100)], "nearest", "numpy", "exact ratio 3"),
        ([(1, 4, 501, 501), (1, 4, 250, 250)], "linear", "dask", "ratio 2.004"),
        ([(1, 4, 256, 256), (1, 4, 128, 128)], "cubic", "dask", "4-tap kernel"),
        ([(1, 4, 128, 128), (1, 4, 256, 256)], "linear", "dask", "upsample"),
    ],
)
def test_auto_takes_numpy_only_where_the_paths_agree(
    tmp_path: Path,
    shapes: list[tuple[int, ...]],
    order: InterpolationOrder,
    expected: str,
    reason: str,
):
    """`auto` is a speed knob, never a silent change of answer.

    `numpy` zooms the whole array, `dask` zooms per block with no halo. They
    coincide only for an integral-ratio downsample with a 2-tap kernel; outside
    that they are two different algorithms, and picking between them on a size
    threshold would make the output depend on how big the image happened to be.
    """
    levels = _pyramid_levels(tmp_path, "auto", shapes, (1, 1, 128, 128), "uint16", None)
    plan = _consolidation_plan(levels[0], levels[1:])

    assert _resolve_auto_mode(levels[0], plan, order) == expected, reason


def test_auto_declines_when_the_source_exceeds_the_budget(tmp_path: Path):
    """The size test is against the source level, and `0` disables the path."""
    shapes = [(1, 4, 256, 256), (1, 4, 128, 128)]
    levels = _pyramid_levels(
        tmp_path, "budget", shapes, (1, 1, 128, 128), "uint16", None
    )
    plan = _consolidation_plan(levels[0], levels[1:])

    assert _resolve_auto_mode(levels[0], plan, "linear") == "numpy"

    config = get_config()
    original = config.consolidation
    try:
        config.consolidation = ConsolidationConfig(numpy_max_bytes=0)
        assert _resolve_auto_mode(levels[0], plan, "linear") == "dask"
    finally:
        config.consolidation = original


@pytest.mark.parametrize(
    "shapes",
    [
        # Inside the envelope: `auto` takes numpy, and must still agree.
        [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)],
        # Outside it: `auto` declines and must behave exactly like `dask`. This
        # is the branch where the resolved mode has to reach `_consolidate_dask`
        # -- passing the caller's `"auto"` through instead only worked because
        # `_zoom_expr` treats everything that is not `"coarsen"` as dask.
        [(1, 4, 501, 501), (1, 4, 250, 250), (1, 4, 125, 125)],
    ],
    ids=["inside", "outside"],
)
def test_auto_matches_dask(tmp_path: Path, shapes: list[tuple[int, ...]]):
    """`auto` never changes the result, whichever way it resolves."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=shapes[0], dtype="uint16")

    results = {}
    for mode in ("dask", "auto"):
        levels = _pyramid_levels(
            tmp_path, mode, shapes, (1, 1, 128, 128), "uint16", None
        )
        levels[0][...] = data
        consolidate_pyramid(levels[0], levels[1:], order="linear", mode=mode)
        results[mode] = [level[...] for level in levels]

    for i, (chunked, auto) in enumerate(
        zip(results["dask"], results["auto"], strict=True)
    ):
        np.testing.assert_array_equal(auto, chunked, err_msg=f"level {i} differs")


def test_consolidate_warns_only_where_the_default_will_change(tmp_path: Path):
    """The notice fires where 1.2 will act differently, and nowhere else.

    Not on an explicit `mode`, and not on a pyramid `auto` would decline anyway
    -- telling someone a default is changing when it would not change their
    result is how a deprecation notice gets filtered out and ignored.
    """
    inside = _pyramid_levels(
        tmp_path,
        "inside",
        [(1, 4, 256, 256), (1, 4, 128, 128)],
        (1, 1, 128, 128),
        "uint16",
        None,
    )
    outside = _pyramid_levels(
        tmp_path,
        "outside",
        [(1, 4, 501, 501), (1, 4, 250, 250)],
        (1, 1, 128, 128),
        "uint16",
        None,
    )

    with pytest.warns(NgioFutureWarning, match="mode"):
        consolidate_pyramid(inside[0], inside[1:], order="linear")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        consolidate_pyramid(inside[0], inside[1:], order="linear", mode="dask")
        consolidate_pyramid(inside[0], inside[1:], order="linear", mode="auto")
        consolidate_pyramid(outside[0], outside[1:], order="linear")
        # Nothing to consolidate: the coming default cannot change anything.
        consolidate_pyramid(inside[0], [], order="linear")


@pytest.mark.parametrize(
    "shape, chunks, shards, expected_shards",
    [
        # Clip lands between chunk multiples: round down to one.
        ((1, 15), (1, 10), (1, 20), (1, 10)),
        ((1, 100), (1, 10), (1, 25), (1, 20)),
        # A shard below one chunk is bumped up to it.
        ((1, 100), (1, 10), (1, 5), (1, 10)),
        # Already valid: untouched.
        ((1, 100), (1, 10), (1, 40), (1, 40)),
    ],
)
def test_pyramid_level_clips_shards_to_a_chunk_multiple(
    tmp_path: Path, shape, chunks, shards, expected_shards
):
    """zarr requires shard % chunk == 0, so the clip must preserve it.

    Clipping chunks and shards to the shape independently could produce e.g.
    chunks (1, 10) with shards (1, 15), which `create_array` rejects.
    """
    level = PyramidLevel(
        path="0",
        shape=shape,
        scale=(1.0, 1.0),
        translation=(0.0, 0.0),
        chunks=chunks,
        shards=shards,
    )
    assert level.shards == expected_shards

    zarr.create_array(
        tmp_path / "level.zarr",
        shape=level.shape,
        chunks=level.chunks,
        shards=level.shards,
        dtype="uint16",
    )


def test_pyramid_level_refuses_auto_chunks_with_explicit_shards():
    """zarr infers "auto" chunks from the full shape, then rejects the shard."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="explicit chunk shape"):
        PyramidLevel(
            path="0",
            shape=(64, 64),
            scale=(1.0, 1.0),
            translation=(0.0, 0.0),
            chunks="auto",
            shards=(4, 16),
        )


def test_coarsen_refuses_to_upsample(tmp_path: Path):
    """Coarsening upward names the cause instead of dividing by zero inside dask.

    Reachable in practice: consolidating from a middle level asks for the levels
    above the source as well, and those edges are upsamples. The factor comes out
    0 and dask's `aligned_coarsen_chunks` raises a bare divide-by-zero that says
    nothing about coarsening, the level, or the alternative.
    """
    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(1, 64, 64), dtype="uint16"
    )
    target_array = zarr.create_array(
        tmp_path / "target.zarr", shape=(1, 128, 128), dtype="uint16"
    )

    with pytest.raises(NgioValueError, match="only downsamples"):
        on_disk_zoom(source_array, target_array, order="linear", mode="coarsen")


@pytest.mark.parametrize("mode", ["dask", "coarsen"])
def test_on_disk_zoom_sharded_matches_unsharded(
    tmp_path: Path, mode: Literal["dask", "coarsen"]
):
    """A zoom onto a sharded target matches the same zoom onto an unsharded one.

    Writes are atomic per shard object, so a block covering only part of a shard
    makes zarr read-modify-write the whole thing — and several such blocks race.
    Both dask paths hand the zoomed array to `store_dask`, which rechunks onto
    the target's write unit (`shards or chunks`) so each shard has exactly one
    writer. The unsharded layout is race-free by alignment, so it is the
    reference the sharded result has to match.
    """
    rng = np.random.default_rng(0)
    source_array = zarr.create_array(
        tmp_path / "source.zarr", shape=(16, 128, 128), dtype="uint8"
    )
    source_array[...] = rng.integers(0, 255, size=(16, 128, 128), dtype="uint8")

    results = {}
    for name, shards in (("unsharded", None), ("sharded", (8, 32, 32))):
        target_array = zarr.create_array(
            tmp_path / f"target_{mode}_{name}.zarr",
            shape=(16, 64, 64),
            chunks=(4, 16, 16),
            shards=shards,
            dtype="uint8",
        )
        on_disk_zoom(source_array, target_array, order="nearest", mode=mode)
        results[name] = target_array[...]

    np.testing.assert_array_equal(results["sharded"], results["unsharded"])
