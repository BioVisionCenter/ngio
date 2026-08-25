"""`by_grid` tail policies and `by_blocks` partitions."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import SegmentationIterator
from ngio.iterators._rois_utils import _axis_intervals, _axis_partition
from ngio.utils import NgioValueError


def _iterator(shape=(100, 100), chunks="auto"):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros(shape, dtype="uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=chunks,
    )
    label = ome_zarr.derive_label("seg")
    return SegmentationIterator(ome_zarr.get_image(), label, axes_order="yx")


def _x_intervals(iterator):
    """The (start, length) tiles along x, read back from the world-space ROIs."""
    out = []
    for roi in iterator.rois:
        box = roi.to_pixel(pixel_size=iterator.ref_image.pixel_size)
        out.append((int(box["x"].start), int(box["x"].length)))
    return sorted(set(out))


def test_tail_clip_is_the_default():
    tiles = _iterator().by_grid(size_x=32)
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 32), (96, 4)]


def test_tail_balance_resplits_the_last_two():
    """A 4 px overhang becomes two 18 px tiles, never a thin slice."""
    tiles = _iterator().by_grid(size_x=32, tail="balance")
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 18), (82, 18)]


def test_tail_balance_with_odd_remainder():
    tiles = _iterator(shape=(100, 65)).by_grid(size_x=32, tail="balance")
    assert _x_intervals(tiles) == [(0, 32), (32, 17), (49, 16)]


def test_tail_balance_needs_adjacent_tiles():
    with pytest.raises(NgioValueError, match="balance"):
        _iterator().by_grid(size_x=32, stride_x=24, tail="balance")


def test_tail_shift_keeps_full_tiles():
    """The last tile moves back to stay full-size, overlapping its neighbour."""
    tiles = _iterator().by_grid(size_x=32, tail="shift")
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 32), (68, 32)]


def test_tail_shift_degrades_to_clip_when_the_tile_is_too_big():
    assert _axis_intervals("x", 20, 32, 32, "shift") == [(0, 20)]


def test_tail_shift_deduplicates_collapsed_starts():
    # 64 @ 32: the grid divides exactly, so shift changes nothing and the
    # would-be duplicate is dropped.
    assert _axis_intervals("x", 64, 32, 32, "shift") == [(0, 32), (32, 32)]


def test_a_zero_sized_axis_is_not_blamed_on_the_tail_policy():
    from ngio.utils import NgioValueError

    for tail in ("clip", "drop", "shift", "balance"):
        with pytest.raises(NgioValueError, match="size 0"):
            _axis_intervals("x", 0, 32, 32, tail)


def test_tail_drop_discards_the_partial_tile():
    tiles = _iterator().by_grid(size_x=32, tail="drop")
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 32)]


def test_by_blocks_partitions_near_equally():
    tiles = _iterator().by_blocks(num_x=3)
    assert _x_intervals(tiles) == [(0, 34), (34, 33), (67, 33)]


def test_by_blocks_validates():
    with pytest.raises(NgioValueError, match=">= 1"):
        _iterator().by_blocks(num_x=0)
    with pytest.raises(NgioValueError, match="empty"):
        _axis_partition("x", 3, 5)


def test_by_blocks_composes_with_the_grid_machinery():
    """Blocks along both axes cross like any grid, with unique names."""
    tiles = _iterator().by_blocks(num_x=2, num_y=2)
    assert len(tiles.rois) == 4
    names = [roi.name for roi in tiles.rois]
    assert len(set(names)) == len(names)


def test_by_write_units_matches_the_old_write_grid():
    """Parity pin: by_write_units tiles exactly as by_write_units() did."""
    iterator = _iterator(shape=(64, 64), chunks=(32, 32))
    tiled = iterator.by_write_units()
    assert len(tiled.rois) == 4
    assert not tiled.check_if_write_units_overlap()


def test_stitch_works_over_a_balance_tiled_grid():
    """Balanced tiles have distinct origins, so the stitch placements rank fine."""
    data = np.zeros((100, 100), dtype="uint8")
    data[20:28, 58:70] = 255  # crosses the balanced boundary at x=64
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=data,
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        consolidation_mode="dask",
    )
    label = ome_zarr.derive_label("seg")
    iterator = (
        SegmentationIterator(
            ome_zarr.get_image(),
            label,
            axes_order="yx",
            consolidation_mode="dask",
        )
        .with_stitch()
        .by_grid(size_x=32, size_y=32, tail="balance")
        .with_halo(x=8, y=8)
    )
    iterator.map(lambda patch: (patch > 128).astype("uint32"))

    written = label.get_as_numpy()
    assert sorted(int(v) for v in set(written.ravel()) - {0}) == [1]
