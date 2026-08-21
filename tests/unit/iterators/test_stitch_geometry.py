"""Pixel-space tile extents, pair sweeps, and the union-find they feed."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.common import Roi
from ngio.common._union_find import UnionFind
from ngio.iterators._stitch_geometry import (
    TileExtent,
    intersection_box,
    sweep_pairs,
    tile_extents,
    touching_unstitched_axes,
)
from ngio.utils import NgioValueError


def _label_image(shape=(100, 100), chunks=(32, 32)):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros(shape, dtype="uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=chunks,
        consolidation_mode="dask",
    )
    return ome_zarr.derive_label("seg")


def _roi(name, y, x):
    return Roi.from_values(slices={"y": y, "x": x}, name=name)


def test_tile_extents_cover_core_and_grown():
    label = _label_image()
    rois = [_roi("a", (0, 32), (0, 32)), _roi("b", (0, 32), (32, 32))]

    def read_roi(roi):  # a fake 4px halo, clipped at the image border
        roi_px = roi.to_pixel(pixel_size=label.pixel_size)
        grown = {}
        for axis in ("y", "x"):
            roi_slice = roi_px.get(axis)
            start = max(0, int(roi_slice.start) - 4)
            stop = min(100, int(roi_slice.start + roi_slice.length) + 4)
            grown[axis] = (start, stop - start)
        return Roi.from_values(slices=grown, name=roi.name, space="pixel")

    extents = tile_extents(rois, label, read_roi, ("y", "x"))
    assert extents[0] == TileExtent(
        index=0, core=((0, 32), (0, 32)), grown=((0, 36), (0, 36))
    )
    assert extents[1].grown == ((0, 36), (28, 68))


def test_tile_extents_unpinned_axis_spans_fully():
    label = _label_image()
    rois = [Roi.from_values(slices={"x": (0, 50), "y": (0, 100)}, name="a")]
    extents = tile_extents(rois, label, lambda roi: roi, ("y", "x"))
    assert extents[0].core == ((0, 100), (0, 50))


def test_tile_extents_refuses_an_empty_region():
    label = _label_image()
    rois = [_roi("a", (10, 0), (0, 32))]
    with pytest.raises(NgioValueError, match="empty"):
        tile_extents(rois, label, lambda roi: roi, ("y", "x"))


def test_intersection_box():
    assert intersection_box(((0, 10), (0, 10)), ((5, 15), (5, 15))) == (
        (5, 10),
        (5, 10),
    )
    assert intersection_box(((0, 10), (0, 10)), ((10, 20), (0, 10))) is None


def test_sweep_pairs_face_and_corner_neighbours():
    """A 2x2 grid of grown boxes: 4 face pairs and 2 corner pairs."""
    boxes = [
        ((0, 36), (0, 36)),
        ((0, 36), (28, 64)),
        ((28, 64), (0, 36)),
        ((28, 64), (28, 64)),
    ]
    assert sweep_pairs(boxes) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_sweep_pairs_disjoint_and_single():
    assert sweep_pairs([((0, 10), (0, 10)), ((20, 30), (20, 30))]) == []
    assert sweep_pairs([((0, 10), (0, 10))]) == []
    # Touching (half-open) is not overlap.
    assert sweep_pairs([((0, 10), (0, 10)), ((0, 10), (10, 20))]) == []


def test_sweep_pairs_matches_brute_force_on_a_ragged_layout():
    rng = np.random.default_rng(7)
    boxes = []
    for _ in range(60):
        y0, x0 = rng.integers(0, 80, size=2)
        boxes.append(
            (
                (int(y0), int(y0 + rng.integers(5, 30))),
                (int(x0), int(x0 + rng.integers(5, 30))),
            )
        )
    brute = sorted(
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if intersection_box(boxes[i], boxes[j]) is not None
    )
    assert sweep_pairs(boxes) == brute


def test_touching_unstitched_axes():
    extents = [
        TileExtent(index=0, core=((0, 1), (0, 64)), grown=((0, 1), (0, 64))),
        TileExtent(index=1, core=((1, 2), (0, 64)), grown=((1, 2), (0, 64))),
    ]
    # z (axis 0) has no halo and the cores touch face to face there.
    assert touching_unstitched_axes(extents, [0]) == [0]
    # Overlapping (not just touching) pairs are already stitched: no warning.
    overlapping = [
        TileExtent(index=0, core=((0, 2), (0, 64)), grown=((0, 2), (0, 64))),
        TileExtent(index=1, core=((1, 3), (0, 64)), grown=((1, 3), (0, 64))),
    ]
    assert touching_unstitched_axes(overlapping, [0]) == []


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
