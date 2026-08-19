"""Region-scoped consolidation: byte parity with a full rebuild, and fallbacks.

The region path is only ever allowed to be a *faster spelling* of the full
rebuild, never a different answer — every parity test here writes patches into
two identical pyramids, rebuilds one fully and one by regions, and requires
every level to match exactly. Anything outside the exact envelope must fall
back to the full rebuild, silently.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import zarr

from ngio.common._pyramid import (
    PixelRegion,
    _consolidation_plan,
    _coverage,
    _merge_regions,
    _normalize_regions,
    _plan_partial,
    consolidate_pyramid,
)
from ngio.config import ConsolidationConfig, get_config
from ngio.utils import NgioValueError


def _pyramid_levels(
    root: Path,
    tag: str,
    shapes: Sequence[tuple[int, ...]],
    chunks: tuple[int, ...],
    dtype: str = "uint16",
    shards: tuple[int, ...] | None = None,
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


_GEOMETRIES = {
    # 4D, two dyadic levels below the source, chunk borders inside the regions.
    "czyx": ([(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)], (1, 1, 128, 128)),
    # 2D, and a last chunk clipped by the array bound (192 = 128 + 64).
    "yx_clipped": ([(256, 192), (128, 96), (64, 48)], (128, 128)),
}

# One region straddling a chunk border, one hugging the far (clipped) corner.
_REGION_SPECS = [
    {"y": slice(100, 160), "x": slice(10, 120)},
    {"y": slice(200, 256), "x": slice(150, 192)},
]


def _regions_for(shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    regions = []
    for spec in _REGION_SPECS:
        leading = (slice(None),) * (len(shape) - 2)
        y = slice(spec["y"].start, min(spec["y"].stop, shape[-2]))
        x = slice(spec["x"].start, min(spec["x"].stop, shape[-1]))
        regions.append((*leading, y, x))
    return regions


@pytest.mark.parametrize("geometry", list(_GEOMETRIES))
@pytest.mark.parametrize("shards", [None, (1, 2, 128, 128)])
@pytest.mark.parametrize("order", ["nearest", "linear"])
@pytest.mark.parametrize("mode", ["dask", "numpy", "coarsen", "auto"])
def test_region_consolidate_is_byte_identical_to_full(
    tmp_path: Path, geometry: str, shards, order, mode
):
    shapes, chunks = _GEOMETRIES[geometry]
    if shards is not None:
        if geometry != "czyx":
            pytest.skip("sharding is exercised on the 4D geometry")
        shards = tuple(shards)

    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=shapes[0], dtype=np.uint16)

    full = _pyramid_levels(tmp_path, "full", shapes, chunks, shards=shards)
    partial = _pyramid_levels(tmp_path, "partial", shapes, chunks, shards=shards)
    for levels in (full, partial):
        levels[0][...] = data
        consolidate_pyramid(levels[0], levels[1:], order=order, mode=mode)

    for region in _regions_for(shapes[0]):
        patch_shape = tuple(
            len(range(*sel.indices(dim)))
            for sel, dim in zip(region, shapes[0], strict=True)
        )
        patch = rng.integers(0, 1000, size=patch_shape, dtype=np.uint16)
        full[0][region] = patch
        partial[0][region] = patch

    consolidate_pyramid(full[0], full[1:], order=order, mode=mode)
    consolidate_pyramid(
        partial[0],
        partial[1:],
        order=order,
        mode=mode,
        regions=_regions_for(shapes[0]),
    )

    for level, (reference, rebuilt) in enumerate(
        zip(full[1:], partial[1:], strict=True)
    ):
        np.testing.assert_array_equal(
            reference[...], rebuilt[...], err_msg=f"level {level + 1} differs"
        )


def test_region_consolidate_skips_untouched_chunks(tmp_path: Path):
    """The point of the feature: an untouched target chunk is never rewritten.

    Chunks are 64px so level 1 holds several write units — with chunks the size
    of the level, snapping would (correctly) cover everything and prove nothing.
    """
    shapes, _ = _GEOMETRIES["czyx"]
    levels = _pyramid_levels(tmp_path, "skip", shapes, (1, 1, 64, 64))
    rng = np.random.default_rng(0)
    levels[0][...] = rng.integers(0, 1000, size=shapes[0], dtype=np.uint16)
    consolidate_pyramid(levels[0], levels[1:], order="linear", mode="dask")

    # Poison an untouched region of level 1: a full rebuild would repair it,
    # a region-scoped one must not touch it.
    poison_region = (slice(None), slice(None), slice(64, 128), slice(64, 128))
    poison = np.full((1, 4, 64, 64), 4321, dtype=np.uint16)
    levels[1][poison_region] = poison

    touched = (slice(None), slice(None), slice(0, 32), slice(0, 32))
    levels[0][touched] = rng.integers(0, 1000, size=(1, 4, 32, 32), dtype=np.uint16)
    consolidate_pyramid(
        levels[0], levels[1:], order="linear", mode="dask", regions=[touched]
    )

    np.testing.assert_array_equal(levels[1][poison_region], poison)


def _merged(regions, shape):
    """Slice-form regions to the merged `PixelRegion`s `_plan_partial` takes."""
    return _merge_regions(_normalize_regions(regions, shape))


class TestPlanPartialFallbacks:
    """`_plan_partial` must return `None` everywhere the region path is inexact."""

    def _plan(self, tmp_path, shapes, chunks=(1, 1, 128, 128)):
        levels = _pyramid_levels(tmp_path, "fb", shapes, chunks)
        return levels, _consolidation_plan(levels[0], levels[1:])

    def test_non_integral_pyramid(self, tmp_path: Path):
        levels, plan = self._plan(
            tmp_path, [(1, 4, 501, 501), (1, 4, 250, 250), (1, 4, 125, 125)]
        )
        region = (slice(None), slice(None), slice(0, 64), slice(0, 64))
        merged = _merged([region], levels[0].shape)
        assert _plan_partial(levels[0], plan, merged, "linear") is None

    def test_cubic_order(self, tmp_path: Path):
        levels, plan = self._plan(
            tmp_path, [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)]
        )
        region = (slice(None), slice(None), slice(0, 64), slice(0, 64))
        merged = _merged([region], levels[0].shape)
        assert _plan_partial(levels[0], plan, merged, "cubic") is None

    def test_mid_level_source_upsamples(self, tmp_path: Path):
        levels, _ = self._plan(
            tmp_path, [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)]
        )
        # Consolidating from level 1 asks for level 0 too — an upsample edge.
        plan = _consolidation_plan(levels[1], [levels[0], levels[2]])
        region = (slice(None), slice(None), slice(0, 32), slice(0, 32))
        merged = _merged([region], levels[1].shape)
        assert _plan_partial(levels[1], plan, merged, "linear") is None

    def test_coverage_above_threshold(self, tmp_path: Path):
        levels, plan = self._plan(
            tmp_path, [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)]
        )
        whole = (slice(None),) * 4
        merged = _merged([whole], levels[0].shape)
        assert _plan_partial(levels[0], plan, merged, "linear") is None

    def test_empty_regions_consolidate_nothing(self, tmp_path: Path):
        """Empty regions mean "nothing was touched" — a no-op, not a rebuild."""
        levels, _ = self._plan(
            tmp_path, [(1, 4, 256, 256), (1, 4, 128, 128), (1, 4, 64, 64)]
        )
        poison = np.full(levels[1].shape, 4321, dtype=np.uint16)
        levels[1][...] = poison

        consolidate_pyramid(levels[0], levels[1:], order="linear", regions=[])
        degenerate = (slice(None), slice(None), slice(10, 10), slice(0, 64))
        consolidate_pyramid(levels[0], levels[1:], order="linear", regions=[degenerate])

        # A full rebuild (or any consolidation at all) would repair the poison.
        np.testing.assert_array_equal(levels[1][...], poison)

    def test_fallback_end_to_end_matches_full(self, tmp_path: Path):
        """Outside the envelope, `regions=` behaves exactly like a full rebuild."""
        shapes = [(1, 4, 501, 501), (1, 4, 250, 250), (1, 4, 125, 125)]
        rng = np.random.default_rng(7)
        data = rng.integers(0, 1000, size=shapes[0], dtype=np.uint16)

        full = _pyramid_levels(tmp_path, "e2e_full", shapes, (1, 1, 128, 128))
        partial = _pyramid_levels(tmp_path, "e2e_partial", shapes, (1, 1, 128, 128))
        region = (slice(None), slice(None), slice(0, 100), slice(0, 100))
        for levels in (full, partial):
            levels[0][...] = data

        consolidate_pyramid(full[0], full[1:], order="linear", mode="dask")
        consolidate_pyramid(
            partial[0], partial[1:], order="linear", mode="dask", regions=[region]
        )
        for reference, rebuilt in zip(full[1:], partial[1:], strict=True):
            np.testing.assert_array_equal(reference[...], rebuilt[...])


def test_coverage_threshold_is_configurable(tmp_path: Path):
    levels = _pyramid_levels(
        tmp_path, "cov", [(1, 4, 256, 256), (1, 4, 128, 128)], (1, 1, 128, 128)
    )
    plan = _consolidation_plan(levels[0], levels[1:])
    region = (slice(None), slice(None), slice(0, 32), slice(0, 32))

    merged = _merged([region], levels[0].shape)
    config = get_config()
    original = config.consolidation
    try:
        config.consolidation = ConsolidationConfig(partial_max_coverage=0.0)
        assert _plan_partial(levels[0], plan, merged, "linear") is None
        config.consolidation = ConsolidationConfig(partial_max_coverage=1.0)
        assert _plan_partial(levels[0], plan, merged, "linear") is not None
    finally:
        config.consolidation = original


class TestMergeRegions:
    def test_overlapping_merge(self):
        merged = _merge_regions([((0, 10), (0, 10)), ((5, 20), (5, 15))])
        assert merged == [((0, 20), (0, 15))]

    def test_adjacent_merge(self):
        merged = _merge_regions([((0, 10), (0, 10)), ((10, 20), (0, 10))])
        assert merged == [((0, 20), (0, 10))]

    def test_diagonal_stays_separate(self):
        regions: list[PixelRegion] = [((0, 10), (0, 10)), ((20, 30), (20, 30))]
        assert _merge_regions(regions) == sorted(regions)

    def test_corner_touch_merges(self):
        # Sharing only a corner still counts as touching — the bounding box
        # over-covers, which is safe by the recompute-is-identity argument.
        merged = _merge_regions([((0, 10), (0, 10)), ((10, 20), (10, 20))])
        assert merged == [((0, 20), (0, 20))]

    def test_bbox_collapse_repeats_until_stable(self):
        # A and B merge into a box that only then touches C.
        merged = _merge_regions(
            [((0, 10), (0, 30)), ((0, 30), (0, 10)), ((12, 20), (12, 20))]
        )
        assert merged == [((0, 30), (0, 30))]

    def test_coverage_of_disjoint_regions(self):
        regions: list[PixelRegion] = [((0, 10), (0, 10)), ((20, 30), (20, 30))]
        assert _coverage(regions, (100, 100)) == pytest.approx(0.02)


class TestNormalizeRegions:
    def test_selection_forms(self):
        normalized = _normalize_regions(
            [(slice(0, 10), 5, [2, 7, 4])], shape=(20, 20, 20)
        )
        assert normalized == [((0, 10), (5, 6), (2, 8))]

    def test_clamping(self):
        # Python slice semantics: a negative start counts from the end, an
        # out-of-range stop clamps to the bound.
        assert _normalize_regions([(slice(-5, 100),)], shape=(20,)) == [((15, 20),)]
        assert _normalize_regions([(slice(0, 100),)], shape=(20,)) == [((0, 20),)]

    def test_negative_int(self):
        assert _normalize_regions([(-1,)], shape=(20,)) == [((19, 20),)]

    def test_empty_selection_drops_out(self):
        assert _normalize_regions([(slice(5, 5),)], shape=(20,)) == []
        assert _normalize_regions([([],)], shape=(20,)) == []

    def test_stepped_slice_raises(self):
        with pytest.raises(NgioValueError, match="stepped"):
            _normalize_regions([(slice(0, 10, 2),)], shape=(20,))

    def test_wrong_ndim_raises(self):
        with pytest.raises(NgioValueError, match="axes"):
            _normalize_regions([(slice(0, 10),)], shape=(20, 20))
