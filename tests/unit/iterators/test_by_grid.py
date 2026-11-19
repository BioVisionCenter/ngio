"""`by_grid` tail policies and `by_blocks` partitions."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.common import Roi
from ngio.iterators import SegmentationIterator
from ngio.iterators._rois_utils import _axis_intervals, _axis_partition
from ngio.utils import NgioValueError


def _iterator(shape=(100, 100), chunks="auto", pixelsize=1.0):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros(shape, dtype="uint8"),
        pixelsize=pixelsize,
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
    iterator = _iterator()
    tiles = iterator.by_grid(size_x=32)
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 32), (96, 4)]
    assert tiles.rois == iterator.by_grid(size_x=32, anchor="roi").rois


def test_tail_balance_resplits_the_last_two():
    """A 4 px overhang becomes two 18 px tiles, never a thin slice."""
    tiles = _iterator().by_grid(size_x=32, tail="balance")
    assert _x_intervals(tiles) == [(0, 32), (32, 32), (64, 18), (82, 18)]


def test_tail_balance_with_odd_remainder():
    tiles = _iterator(shape=(100, 65)).by_grid(size_x=32, tail="balance")
    assert _x_intervals(tiles) == [(0, 32), (32, 17), (49, 16)]


@pytest.mark.parametrize("anchor", ["image", "roi"])
def test_tail_balance_needs_adjacent_tiles(anchor):
    with pytest.raises(NgioValueError, match="balance"):
        _iterator().by_grid(size_x=32, stride_x=24, tail="balance", anchor=anchor)


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


def test_roi_anchor_shifts_full_tiles_within_each_field():
    regions = [
        Roi.from_values(name=f"fov_{y}_{x}", slices={"y": (y, 500), "x": (x, 500)})
        for y in (0, 500)
        for x in (0, 500)
    ]
    iterator = _iterator(shape=(1000, 1000)).product(regions)
    grid = {"size_x": 100, "size_y": 100, "stride_x": 90, "stride_y": 90}

    image_tiles = iterator.by_grid(**grid)
    assert image_tiles.rois == iterator.by_grid(**grid, anchor="image").rois
    assert len(image_tiles.rois) == 169

    tiles = iterator.by_grid(**grid, anchor="roi", tail="shift")
    assert len(tiles.rois) == 144
    for region in regions:
        prefix = f"{region.name}:"
        field_tiles = [roi for roi in tiles.rois if roi.get_name().startswith(prefix)]
        assert len(field_tiles) == 36
        for axis in ("y", "x"):
            assert {(roi[axis].start, roi[axis].length) for roi in field_tiles} == {
                (region[axis].start + start, 100)
                for start in (0, 90, 180, 270, 360, 400)
            }


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("clip", [(27, 32), (59, 32), (91, 1)]),
        ("balance", [(27, 32), (59, 17), (76, 16)]),
        ("shift", [(27, 32), (59, 32), (60, 32)]),
        ("drop", [(27, 32), (59, 32)]),
    ],
)
def test_roi_anchor_applies_tail_at_region_edge(tail, expected):
    # The parent has no upper y bound, and its physical x bounds map to [27, 92).
    region = Roi.from_values(name=None, slices={"y": (10.0, None), "x": (13.6, 32.2)})
    iterator = _iterator(pixelsize=0.5).product([region])
    tiles = iterator.by_grid(size_x=32, anchor="roi", tail=tail)
    assert _x_intervals(tiles) == expected
    assert {(roi["y"].start, roi["y"].length) for roi in tiles.rois} == {(10.0, 40.0)}


@pytest.mark.parametrize("names", [("a", "b"), (None, None)])
def test_roi_anchor_keeps_names_unique_for_overlapping_regions(names):
    regions = [
        Roi.from_values(name=name, slices={"y": (16, extent), "x": (16, extent)})
        for name, extent in zip(names, (48, 64), strict=True)
    ]
    tiles = (
        _iterator()
        .product(regions)
        .by_grid(size_y=32, size_x=32, anchor="roi", base_name="patch")
    )
    assert len({roi.name for roi in tiles.rois}) == len(tiles.rois) == 8
    if names[0] is not None:
        assert tiles.rois[0].name == "a:patch_t0_z0_y16_x16"
        assert tiles.rois[4].name == "b:patch_t0_z0_y16_x16"


@pytest.mark.parametrize("method", ["by_chunks", "by_write_units"])
def test_storage_grid_stays_image_anchored_after_product(method):
    region = Roi.from_values(name=None, slices={"y": (9, 47), "x": (11, 45)})
    iterator = _iterator(shape=(64, 64), chunks=(32, 32)).product([region])
    tiles = getattr(iterator, method)()
    assert _x_intervals(tiles) == [(11, 21), (32, 24)]
    assert not tiles.check_if_write_units_overlap()


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
