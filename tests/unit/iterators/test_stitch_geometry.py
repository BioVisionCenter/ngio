"""Tests for the tile geometry behind stitching, and for union-find."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.common._union_find import UnionFind
from ngio.iterators._stitch_geometry import (
    forward_bands,
    neighbours_along,
    shared_band,
    tile_placements,
)
from ngio.utils import NgioValueError


def _image(shape=(100, 100)):
    return create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros(shape, dtype="uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=(32, 32),
        consolidation_mode="dask",
    ).get_image()


def _grid_rois(image, size=32):
    from ngio.iterators._rois_utils import grid

    return grid(
        rois=[image.build_image_roi_table(name=None).rois()[0]],
        ref_image=image,
        size_y=size,
        size_x=size,
    )


def test_grid_index_survives_a_clipped_last_tile():
    """A 100px axis tiled at 32 ends with start=96, length=4.

    `96 // 4` is 24; the answer is 3. Ranking the distinct origins gets it right
    without needing to know the tile size at all.
    """
    image = _image(shape=(100, 100))
    rois = _grid_rois(image, size=32)
    placements = tile_placements(rois, image, axes=("y", "x"))

    starts = sorted({p.origin[1] for p in placements})
    assert starts == [0, 32, 64, 96]
    last = next(p for p in placements if p.origin == (96, 96))
    assert last.grid == (3, 3)
    assert last.stop == (100, 100), "the final tile is clipped, not full-size"
    # Every cell is distinct, which is what the id offsets rely on.
    assert len({p.grid for p in placements}) == len(placements)


def test_neighbours_are_face_adjacent_only():
    image = _image(shape=(64, 64))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    assert len(placements) == 4

    pairs = neighbours_along(placements, axis=1)
    assert {(a.grid, b.grid) for a, b in pairs} == {((0, 0), (0, 1)), ((1, 0), (1, 1))}
    # Diagonal cells never pair: they share a corner, not a face.
    for a, b in pairs:
        assert a.grid[0] == b.grid[0]


def test_shared_band_is_inside_the_upper_tile():
    image = _image(shape=(64, 64))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    lower, upper = neighbours_along(placements, axis=1)[0]

    band = shared_band(lower, upper, axis=1, halo=4, axes=("y", "x"))
    assert band is not None
    # The band starts where the upper tile starts and is `halo` wide.
    assert band["x"] == (32, 36)
    assert band["y"] == (0, 32)


def test_shared_band_is_none_without_a_halo():
    image = _image(shape=(64, 64))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    lower, upper = neighbours_along(placements, axis=1)[0]
    assert shared_band(lower, upper, axis=1, halo=0, axes=("y", "x")) is None


def test_shared_band_clips_to_the_upper_tile():
    """A halo wider than the neighbour cannot read past it."""
    image = _image(shape=(100, 100))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    lower = next(p for p in placements if p.grid == (0, 2))
    upper = next(p for p in placements if p.grid == (0, 3))

    band = shared_band(lower, upper, axis=1, halo=8, axes=("y", "x"))
    assert band is not None
    assert band["x"] == (96, 100), "the clipped last tile is only 4px wide"


def test_forward_bands_never_collide():
    """One band per tile per axis, each inside the next tile's core."""
    image = _image(shape=(100, 100))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    bands = forward_bands(placements, axis=1, halo=8, axes=("y", "x"))

    # Far-edge tiles contribute nothing; every other tile contributes one band.
    assert all(placements[i].grid[1] < 3 for i in bands)

    seen: set[tuple[int, int]] = set()
    for band in bands.values():
        pixels = {
            (y, x)
            for y in range(*band["y"])
            for x in range(*band["x"])
        }
        assert not (pixels & seen), "two tiles wrote the same scratch pixel"
        seen |= pixels


def test_forward_bands_stay_inside_the_clipped_last_tile():
    """A halo wider than the neighbour is clipped, never spilling past it."""
    image = _image(shape=(100, 100))
    placements = tile_placements(_grid_rois(image, size=32), image, axes=("y", "x"))
    bands = forward_bands(placements, axis=1, halo=16, axes=("y", "x"))

    third_column = next(
        bands[p.index] for p in placements if p.grid == (0, 2) and p.index in bands
    )
    assert third_column["x"] == (96, 100)


def test_union_find_groups_transitively():
    uf = UnionFind[int]()
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(10, 11)

    resolved = uf.resolve()
    assert resolved[1] == resolved[2] == resolved[3] == 1
    assert resolved[10] == resolved[11] == 10
    assert resolved[1] != resolved[10]


def test_union_find_representative_is_deterministic():
    """Order of unions must not change the answer."""
    forward = UnionFind[int]()
    for a, b in ((3, 1), (2, 3), (5, 2)):
        forward.union(a, b)

    backward = UnionFind[int]()
    for a, b in ((5, 2), (2, 3), (3, 1)):
        backward.union(a, b)

    assert forward.resolve() == backward.resolve()
    assert set(forward.resolve().values()) == {1}


def test_union_find_is_idempotent():
    uf = UnionFind[int]()
    uf.union(1, 2)
    first = uf.resolve()
    uf.union(1, 2)
    uf.union(2, 1)
    assert uf.resolve() == first
