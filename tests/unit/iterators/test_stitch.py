"""End-to-end stitching: objects split by a tile boundary become one id."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import SegmentationIterator, ThreadedMapper
from ngio.iterators._stitch import StitchConfig
from ngio.utils import NgioValueError


def _connected_components(patch: np.ndarray) -> np.ndarray:
    """A tiny 2D labeller, so the test does not depend on scipy."""
    out = np.zeros(patch.shape, dtype="uint32")
    next_id = 0
    for start in zip(*np.nonzero(patch > 0), strict=True):
        if out[start]:
            continue
        next_id += 1
        stack = [start]
        out[start] = next_id
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < patch.shape[0]
                    and 0 <= nx < patch.shape[1]
                    and patch[ny, nx] > 0
                    and not out[ny, nx]
                ):
                    out[ny, nx] = next_id
                    stack.append((ny, nx))
    return out


def _setup(image_data: np.ndarray, tile: int = 32):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=image_data,
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=(tile, tile),
        consolidation_mode="dask",
    )
    return ome_zarr, ome_zarr.get_image(), ome_zarr.derive_label("seg")


def _run_with(image_data, func, stitch, *, halo=4, tile=32, mapper=None):
    ome_zarr, image, label = _setup(image_data, tile=tile)
    iterator = SegmentationIterator(
        image, label, axes_order="yx", consolidation_mode="dask", stitch=stitch
    ).grid(size_y=tile, size_x=tile)
    if halo:
        iterator = iterator.with_halo(y=halo, x=halo)
    iterator.map(func, mapper=mapper)
    return ome_zarr.get_label("seg").get_as_numpy()


def _run(image_data, *, stitch, halo=4, tile=32, mapper=None):
    return _run_with(
        image_data, _connected_components, stitch, halo=halo, tile=tile, mapper=mapper
    )


def _one_object_across_the_seam(size=64, tile=32):
    """A bar spanning the vertical tile boundary at x=tile."""
    data = np.zeros((size, size), dtype="uint8")
    data[20:28, tile - 10 : tile + 10] = 255
    return data


def _two_objects_touching_at_the_seam(size=64, tile=32):
    """Two bars that meet exactly at x=tile but are distinct objects.

    Adjacency across the cut cannot tell this from a single split object;
    overlap can, because each tile predicts *both* bars in its halo and the
    correspondence is one-to-one.
    """
    data = np.zeros((size, size), dtype="uint8")
    data[20:28, tile - 10 : tile] = 255
    data[30:38, tile : tile + 10] = 255
    return data


def _object_ids(written: np.ndarray) -> set[int]:
    return {int(value) for value in np.unique(written) if value}


def test_split_object_becomes_one_id():
    data = _one_object_across_the_seam()
    stitched = _run(data, stitch=True)
    assert len(_object_ids(stitched)) == 1

    # The pixels are all still labelled, just with one id.
    np.testing.assert_array_equal(stitched > 0, data > 0)


def test_ids_collide_across_tiles_without_stitching():
    """The control, and the problem stitching exists to solve.

    Every tile numbers its objects from 1, so without stitching two objects in
    two different tiles are both `1` and nothing downstream can tell them apart.
    (This is also why "one id" is not on its own evidence that a split object
    was joined — unstitched halves collide on `1` too.)
    """
    data = np.zeros((64, 64), dtype="uint8")
    data[8:16, 8:16] = 255  # tile (0, 0)
    data[40:48, 40:48] = 255  # tile (1, 1)

    assert _object_ids(_run(data, stitch=False)) == {1}, "two objects, one id"
    assert len(_object_ids(_run(data, stitch=True))) == 2


def test_touching_objects_stay_distinct():
    """The case adjacency gets wrong, and the reason for the overlap criterion.

    Note this holds *regardless of the threshold*: the two tiles pair their
    objects one-to-one, so the wrong pairs score zero rather than merely low.
    The criterion is what saves this, not `iou_threshold` — see
    `test_threshold_rejects_weak_agreement` for what the threshold is for.
    """
    data = _two_objects_touching_at_the_seam()
    stitched = _run(data, stitch=True)
    assert len(_object_ids(stitched)) == 2, "two distinct objects were merged"


def _eroding_components(patch: np.ndarray) -> np.ndarray:
    """Label, then shave the object's last column *of this patch*.

    Two tiles see different windows, so they shave different pixels — which is
    how a deterministic test can produce genuine disagreement between
    neighbours. A real segmenter disagrees for its own reasons.
    """
    out = _connected_components(patch)
    for label in np.unique(out):
        if not label:
            continue
        columns = np.nonzero((out == label).any(axis=0))[0]
        out[:, columns.max()] = np.where(
            out[:, columns.max()] == label, 0, out[:, columns.max()]
        )
    return out


def test_threshold_rejects_weak_agreement():
    """Neighbours that only partly agree about an object are left unmerged."""
    data = _one_object_across_the_seam()

    lenient = _run(data, stitch=StitchConfig(iou_threshold=0.3), halo=4)
    strict = _run(data, stitch=StitchConfig(iou_threshold=0.99), halo=4)

    # Identical predictions agree perfectly, so even the strict run merges.
    assert len(_object_ids(lenient)) == 1
    assert len(_object_ids(strict)) == 1

    # With neighbours that disagree near the seam, the threshold starts to bite.
    disagreeing = _run_with(data, _eroding_components, StitchConfig(iou_threshold=0.99))
    assert len(_object_ids(disagreeing)) == 2, "weak agreement should not merge"


def _five_objects(size=64, tile=32):
    """Two objects straddling a seam, three wholly inside their own tile."""
    data = np.zeros((size, size), dtype="uint8")
    data[4:10, tile - 6 : tile + 8] = 255
    data[50:58, tile - 6 : tile + 8] = 255
    data[14:20, 4:10] = 255
    data[14:20, tile + 8 : tile + 16] = 255
    data[40:46, 4:10] = 255
    return data


def test_ids_are_compacted():
    data = _one_object_across_the_seam()
    assert _object_ids(_run(data, stitch=True)) == {1}


def test_compacted_ids_are_dense_and_sequential():
    """Five objects, two of them merged across a seam, numbered 1..5 with no gaps."""
    stitched = _run(_five_objects(), stitch=True)
    assert sorted(_object_ids(stitched)) == [1, 2, 3, 4, 5]


def test_uncompacted_ids_keep_their_blocks():
    """Without compaction the ids stay sparse, one block per tile."""
    stitched = _run(_five_objects(), stitch=StitchConfig(compact=False))
    ids = _object_ids(stitched)
    assert len(ids) == 5
    assert min(ids) > 1, "block offsets are preserved"


def test_ids_are_left_sparse_when_compaction_is_off():
    data = _one_object_across_the_seam()
    stitched = _run(data, stitch=StitchConfig(compact=False))
    ids = _object_ids(stitched)
    assert len(ids) == 1
    assert max(ids) > 1, "uncompacted ids keep their block offsets"


def test_stitching_is_parallel_safe():
    data = _one_object_across_the_seam()
    serial = _run(data, stitch=True)
    parallel = _run(data, stitch=True, mapper=ThreadedMapper(4))
    np.testing.assert_array_equal(serial, parallel)


def test_scratch_arrays_are_removed():
    _, image, label = _setup(_one_object_across_the_seam())
    iterator = (
        SegmentationIterator(
            image, label, axes_order="yx", consolidation_mode="dask", stitch=True
        )
        .grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )
    iterator.map(_connected_components)

    assert "_ngio_stitch" not in list(label._group_handler.group.keys())


def test_scratch_can_live_in_a_separate_store():
    """The bands are small, so a MemoryStore is often the right home for them."""
    scratch = MemoryStore()
    data = _one_object_across_the_seam()
    stitched = _run(data, stitch=StitchConfig(scratch_store=scratch))

    assert len(_object_ids(stitched)) == 1
    assert list(zarr.open_group(scratch).keys()), "bands were banked in the store"


def test_a_supplied_scratch_store_keeps_the_label_clean():
    """Nothing is written inside the OME-Zarr label group."""
    _, image, label = _setup(_one_object_across_the_seam())
    iterator = (
        SegmentationIterator(
            image,
            label,
            axes_order="yx",
            consolidation_mode="dask",
            stitch=StitchConfig(scratch_store=MemoryStore()),
        )
        .grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )
    iterator.map(_connected_components)

    assert "_ngio_stitch" not in list(label._group_handler.group.keys())


def test_memory_scratch_refuses_to_cross_a_process_boundary():
    """A MemoryStore pickles by value, so the bands would be lost silently."""
    import pickle

    _, image, label = _setup(_one_object_across_the_seam())
    iterator = (
        SegmentationIterator(
            image,
            label,
            axes_order="yx",
            consolidation_mode="dask",
            stitch=StitchConfig(scratch_store=MemoryStore()),
        )
        .grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )
    setter = iterator.build_numpy_setter(iterator.rois[0])

    with pytest.raises(NgioValueError, match="in-memory store cannot be used"):
        pickle.dumps(setter)


def test_stitching_requires_a_halo():
    _, image, label = _setup(_one_object_across_the_seam())
    iterator = SegmentationIterator(
        image, label, axes_order="yx", consolidation_mode="dask", stitch=True
    ).grid(size_y=32, size_x=32)

    with pytest.raises(NgioValueError, match="needs a halo"):
        iterator.map(_connected_components)


def test_config_rejects_a_zero_threshold():
    with pytest.raises(NgioValueError, match="iou_threshold"):
        StitchConfig(iou_threshold=0.0)
    with pytest.raises(NgioValueError, match="block_size"):
        StitchConfig(block_size=0)


def test_stitch_survives_a_clipped_last_tile():
    """A 100px image tiled at 32 ends with a 4px tile; ids must still be unique."""
    data = np.zeros((100, 100), dtype="uint8")
    data[20:28, 22:42] = 255
    stitched = _run(data, stitch=True, tile=32)
    assert len(_object_ids(stitched)) == 1


def test_relabel_sequential_without_a_stitch():
    """Dense ids no longer require stitching: the label can renumber itself."""
    ome_zarr, _, label = _setup(np.zeros((64, 64), dtype="uint8"))
    sparse = np.zeros((64, 64), dtype=label.zarr_array.dtype)
    sparse[4:10, 4:10] = 5000
    sparse[40:46, 40:46] = 9000
    label.set_array(sparse)

    assert label.relabel_sequential(consolidation_mode="dask") == 2
    assert _object_ids(ome_zarr.get_label("seg").get_as_numpy()) == {1, 2}


def test_stitch_refuses_the_dask_path():
    """The dask setters do not offset ids or bank bands; raising beats corrupting."""
    _, image, label = _setup(_one_object_across_the_seam())
    iterator = (
        SegmentationIterator(
            image, label, axes_order="yx", consolidation_mode="dask", stitch=True
        )
        .grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )

    with pytest.raises(NgioValueError, match="numpy path"):
        iterator.build_dask_setter(iterator.rois[0])


def test_stitch_refuses_tiles_split_on_an_unhaloed_axis():
    """z-tiled tiles with a yx-only halo collapse onto one grid cell: refuse."""
    data = np.zeros((2, 64, 64), dtype="uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=data,
        pixelsize=1.0,
        axes_names="zyx",
        levels=1,
        chunks=(1, 32, 32),
        consolidation_mode="dask",
    )
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("seg")
    iterator = (
        SegmentationIterator(
            image, label, axes_order="zyx", consolidation_mode="dask", stitch=True
        )
        .grid(size_z=1, size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )

    with pytest.raises(NgioValueError, match="same stitch grid cell"):
        iterator.map(lambda patch: np.zeros_like(patch, dtype="uint32"))


def test_stitch_refuses_duplicate_tile_names():
    from ngio import Roi

    _, image, label = _setup(_one_object_across_the_seam())
    iterator = SegmentationIterator(
        image, label, axes_order="yx", consolidation_mode="dask", stitch=True
    )
    iterator._set_rois(
        [
            Roi.from_values(name="dup", slices={"y": (0, 64), "x": (0, 32)}),
            Roi.from_values(name="dup", slices={"y": (0, 64), "x": (32, 32)}),
        ]
    )
    iterator = iterator.with_halo(y=4, x=4)

    with pytest.raises(NgioValueError, match="share the name"):
        iterator.map(_connected_components)


def test_failed_map_removes_the_scratch():
    """A failed run cannot be resolved; the scratch must not linger."""
    _, image, label = _setup(_one_object_across_the_seam())
    iterator = (
        SegmentationIterator(
            image, label, axes_order="yx", consolidation_mode="dask", stitch=True
        )
        .grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )

    def exploding(patch: np.ndarray) -> np.ndarray:
        raise RuntimeError("segmenter died")

    with pytest.raises(RuntimeError, match="segmenter died"):
        iterator.map(exploding)

    assert "_ngio_stitch" not in list(label._group_handler.group.keys())
