"""Tests for the label array operations behind stitching."""

import numpy as np
import pytest
import zarr

from ngio.common._label_ops import (
    check_offset_fits,
    chunk_selections,
    offset_labels,
    overlap_iou,
    relabel,
    relabel_sequential,
)
from ngio.utils import NgioValueError


def test_offset_leaves_background_alone():
    array = np.array([[0, 1], [2, 0]], dtype="uint32")
    np.testing.assert_array_equal(offset_labels(array, 1000), [[0, 1001], [1002, 0]])


def test_offset_preserves_dtype():
    assert offset_labels(np.ones((2, 2), dtype="uint16"), 5).dtype == np.uint16


def test_overflow_is_refused():
    array = np.full((2, 2), 10, dtype="uint16")
    check_offset_fits(array, 100, block_size=100)
    with pytest.raises(NgioValueError, match="past what uint16 can hold"):
        check_offset_fits(array, 70_000, block_size=10_000)


def test_float_labels_are_refused():
    with pytest.raises(NgioValueError, match="integer-typed"):
        check_offset_fits(np.zeros((2, 2), dtype="float32"), 1, block_size=1)


def test_overlap_scores_identical_labellings_at_one():
    left = np.array([[1, 1], [0, 2]], dtype="uint32")
    right = np.array([[7, 7], [0, 9]], dtype="uint32")
    assert overlap_iou(left, right) == {(1, 7): 1.0, (2, 9): 1.0}


def test_overlap_ignores_background():
    left = np.array([[1, 0], [0, 0]], dtype="uint32")
    right = np.array([[5, 0], [0, 0]], dtype="uint32")
    assert overlap_iou(left, right) == {(1, 5): 1.0}


def test_overlap_penalises_partial_agreement():
    # `left` calls all four pixels one object; `right` splits them in two.
    left = np.array([[1, 1], [1, 1]], dtype="uint32")
    right = np.array([[5, 5], [6, 6]], dtype="uint32")
    scores = overlap_iou(left, right)
    assert scores[(1, 5)] == pytest.approx(0.5)
    assert scores[(1, 6)] == pytest.approx(0.5)


def test_overlap_separates_distinct_neighbours():
    """The case adjacency gets wrong: two objects that touch stay two pairs."""
    left = np.array([[1, 1, 2, 2]], dtype="uint32")
    right = np.array([[8, 8, 9, 9]], dtype="uint32")
    scores = overlap_iou(left, right)
    assert scores == {(1, 8): 1.0, (2, 9): 1.0}
    assert (1, 9) not in scores and (2, 8) not in scores


def test_overlap_requires_matching_shapes():
    with pytest.raises(NgioValueError, match="different shapes"):
        overlap_iou(np.zeros((2, 2), "uint32"), np.zeros((2, 3), "uint32"))


def test_relabel_maps_only_known_ids():
    array = np.array([[0, 1, 2], [3, 4, 5]], dtype="uint32")
    np.testing.assert_array_equal(
        relabel(array, {1: 100, 4: 400}), [[0, 100, 2], [3, 400, 5]]
    )


def test_relabel_is_idempotent_under_a_canonical_mapping():
    array = np.array([[1, 2, 3]], dtype="uint32")
    mapping = {1: 1, 2: 1, 3: 3}
    once = relabel(array, mapping)
    np.testing.assert_array_equal(relabel(once, mapping), once)


def test_relabel_preserves_dtype_and_background():
    array = np.array([[0, 7]], dtype="uint16")
    out = relabel(array, {7: 3})
    assert out.dtype == np.uint16
    assert out[0, 0] == 0


def _array(data: np.ndarray, chunks=(4, 4)) -> zarr.Array:
    array = zarr.create_array(
        store={}, shape=data.shape, dtype=data.dtype, chunks=chunks
    )
    array[...] = data
    return array


def test_relabel_sequential_produces_dense_ids():
    data = np.zeros((8, 8), dtype="uint32")
    data[0:2, 0:2] = 5000
    data[6:8, 6:8] = 9000
    data[0:2, 6:8] = 77

    array = _array(data)
    assert relabel_sequential(array) == 3
    assert sorted({int(v) for v in np.unique(np.asarray(array[...])) if v}) == [1, 2, 3]


def test_relabel_sequential_numbers_by_first_encounter():
    """Not by sorting the old ids — which is what makes it a single pass."""
    data = np.zeros((8, 8), dtype="uint32")
    data[0:2, 0:2] = 9000  # first chunk visited
    data[6:8, 6:8] = 5000  # last chunk visited

    array = _array(data)
    relabel_sequential(array)
    written = np.asarray(array[...])
    assert written[0, 0] == 1, "the id met first becomes 1, however large it was"
    assert written[7, 7] == 2


def test_relabel_sequential_applies_canonical_first():
    """Grouped ids collapse to one number before the renumbering."""
    data = np.zeros((8, 8), dtype="uint32")
    data[0:2, 0:2] = 100
    data[6:8, 6:8] = 200
    data[0:2, 6:8] = 300

    array = _array(data)
    assert relabel_sequential(array, {200: 100}) == 2
    written = np.asarray(array[...])
    assert written[0, 0] == written[7, 7], "the union should have merged these"
    assert written[0, 7] != written[0, 0]


def test_relabel_sequential_leaves_background_alone():
    data = np.zeros((8, 8), dtype="uint32")
    data[0:2, 0:2] = 42
    array = _array(data)
    relabel_sequential(array)
    assert int(np.asarray(array[...])[4, 4]) == 0


def test_chunk_selections_cover_the_array_exactly():
    array = _array(np.zeros((10, 8), dtype="uint32"), chunks=(4, 4))
    seen = np.zeros((10, 8), dtype=int)
    for selection in chunk_selections(array):
        seen[selection] += 1
    assert (seen == 1).all(), "every pixel visited exactly once"


def test_offset_refuses_id_spill_into_the_next_block():
    """An id at or above block_size would collide with the neighbour's labels."""
    from ngio.common._label_ops import check_offset_fits
    from ngio.utils import NgioValueError

    patch = np.array([[0, 150]], dtype="uint32")
    with pytest.raises(NgioValueError, match="spill"):
        check_offset_fits(patch, offset=100, block_size=100)

    # The largest id that still fits is fine.
    check_offset_fits(np.array([[99]], dtype="uint32"), offset=100, block_size=100)
