import numpy as np
import pytest

from ngio.io_pipes._ops_slices_utils import (
    check_if_chunks_overlap,
    chunk_rects_intersect,
    compute_chunk_rect,
    compute_slice_chunks,
)
from ngio.utils import NgioValueError


def _rect_from_chunk_set(chunk_set: set[tuple[int, ...]]) -> tuple | None:
    if not chunk_set:
        return None
    ndim = len(next(iter(chunk_set)))
    return tuple(
        (min(c[ax] for c in chunk_set), max(c[ax] for c in chunk_set))
        for ax in range(ndim)
    )


def _random_slice_tuple(rng: np.random.Generator, shape: tuple[int, ...]) -> tuple:
    slicing = []
    for size in shape:
        kind = rng.integers(0, 3)
        if kind == 0:
            start = int(rng.integers(0, size))
            stop = int(rng.integers(start, size + 1))
            slicing.append(slice(start, stop))
        elif kind == 1:
            slicing.append(int(rng.integers(0, size)))
        else:
            slicing.append(slice(None))
    return tuple(slicing)


def test_compute_chunk_rect_matches_sets_for_slices():
    """For slice/int selections the rect must equal the chunk set's bounding box,
    and (as the per-axis footprints are contiguous) the rect intersection must
    agree exactly with set intersection."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 40)) for _ in range(ndim))
        chunks = tuple(int(rng.integers(1, size + 1)) for size in shape)

        tuple_a = _random_slice_tuple(rng, shape)
        tuple_b = _random_slice_tuple(rng, shape)

        set_a = compute_slice_chunks(shape, chunks, tuple_a)
        set_b = compute_slice_chunks(shape, chunks, tuple_b)
        rect_a = compute_chunk_rect(shape, chunks, tuple_a)
        rect_b = compute_chunk_rect(shape, chunks, tuple_b)

        assert rect_a == _rect_from_chunk_set(set_a)
        assert rect_b == _rect_from_chunk_set(set_b)
        assert chunk_rects_intersect(rect_a, rect_b) == bool(set_a & set_b)


def test_chunk_rect_list_selection_is_conservative_only():
    """For list selections the rect may over-report overlap, never under-report."""
    rng = np.random.default_rng(1)
    shape, chunks = (32,), (4,)
    for _ in range(200):
        list_a = sorted(rng.choice(32, size=rng.integers(1, 6), replace=False))
        list_b = sorted(rng.choice(32, size=rng.integers(1, 6), replace=False))
        tuple_a = ([int(v) for v in list_a],)
        tuple_b = ([int(v) for v in list_b],)

        sets_overlap = bool(
            compute_slice_chunks(shape, chunks, tuple_a)
            & compute_slice_chunks(shape, chunks, tuple_b)
        )
        rects_overlap = chunk_rects_intersect(
            compute_chunk_rect(shape, chunks, tuple_a),
            compute_chunk_rect(shape, chunks, tuple_b),
        )
        if sets_overlap:
            assert rects_overlap

    # Pinned over-approximation: sets are disjoint, bounding boxes intersect.
    shape, chunks = (16,), (1,)
    assert not (
        compute_slice_chunks(shape, chunks, ([0, 8],))
        & compute_slice_chunks(shape, chunks, ([4],))
    )
    assert chunk_rects_intersect(
        compute_chunk_rect(shape, chunks, ([0, 8],)),
        compute_chunk_rect(shape, chunks, ([4],)),
    )


def test_chunk_rect_empty_selection_is_none():
    shape, chunks = (16, 16), (4, 4)
    assert compute_chunk_rect(shape, chunks, (slice(5, 5), slice(None))) is None
    assert compute_chunk_rect(shape, chunks, (slice(7, 3), slice(None))) is None
    assert compute_chunk_rect(shape, chunks, ([], slice(None))) is None

    rect = compute_chunk_rect(shape, chunks, (slice(None), slice(None)))
    assert rect == ((0, 3), (0, 3))
    assert chunk_rects_intersect(None, rect) is False
    assert chunk_rects_intersect(rect, None) is False
    assert chunk_rects_intersect(None, None) is False


def test_chunk_rect_validation():
    with pytest.raises(NgioValueError):
        compute_chunk_rect((16, 16), (4, 4), (slice(None),))
    with pytest.raises(IndexError):
        compute_chunk_rect((16,), (4,), (16,))
    with pytest.raises(IndexError):
        compute_chunk_rect((16,), (4,), ([3, 16],))
    with pytest.raises(TypeError):
        compute_chunk_rect((16,), (4,), ([3, "x"],))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        compute_chunk_rect((16,), (4,), ("x",))  # ty: ignore[invalid-argument-type]
    with pytest.raises(NgioValueError):
        chunk_rects_intersect(((0, 1),), ((0, 1), (0, 1)))


def test_check_if_chunks_overlap_rect_semantics():
    shape, chunks = (8, 64), (1, 16)
    disjoint = [(0, slice(0, 16)), (0, slice(16, 32)), (1, slice(0, 16))]
    assert not check_if_chunks_overlap(disjoint, shape, chunks)

    overlapping = [(0, slice(0, 16)), (0, slice(8, 24))]
    assert check_if_chunks_overlap(overlapping, shape, chunks)

    # Empty selections conflict with nothing, even against themselves.
    empties = [(0, slice(4, 4)), (0, slice(4, 4)), (0, slice(0, 16))]
    assert not check_if_chunks_overlap(empties, shape, chunks)


def test_check_if_chunks_overlap_no_warning_at_scale():
    """~20k pairs of disjoint tuples: must stay silent under filterwarnings=error
    (the old implementation warned at exactly 10k pairs)."""
    shape, chunks = (200,), (1,)
    slices = [(slice(i, i + 1),) for i in range(200)]  # 19900 pairs, all disjoint
    assert not check_if_chunks_overlap(slices, shape, chunks)
