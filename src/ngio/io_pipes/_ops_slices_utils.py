import logging
import warnings
from collections.abc import Iterable, Iterator
from itertools import product
from typing import TypeAlias, TypeVar

from ngio.utils import NgioUserWarning, NgioValueError

logger = logging.getLogger(f"ngio:{__name__}")


T = TypeVar("T")

##############################################################
#
# Check slice overlaps
#
##############################################################


def _pairs_stream(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    # Same as combinations but yields pairs as soon as they are generated
    seen: list[T] = []
    for a in iterable:
        for b in seen:
            yield b, a
        seen.append(a)


SlicingType: TypeAlias = slice | list[int] | int


def check_elem_intersection(s1: SlicingType, s2: SlicingType) -> bool:
    """Compare if two SlicingType elements intersect.

    If they are a slice, check if they overlap.
    If they are integers, check if they are equal.
    If they are lists, check if they have any common elements.
    """
    if not isinstance(s1, type(s2)):
        raise NgioValueError(
            f"Slices must be of the same type. Got {type(s1)} and {type(s2)}"
        )

    if isinstance(s1, slice) and isinstance(s2, slice):
        # Handle slice objects
        start1, stop1, step1 = s1.start or 0, s1.stop or float("inf"), s1.step or 1
        start2, stop2, step2 = s2.start or 0, s2.stop or float("inf"), s2.step or 1

        if step1 != 1 or step2 != 1:
            raise NotImplementedError(
                "Intersection for slices with step != 1 is not implemented"
            )

        return not (stop1 <= start2 or stop2 <= start1)
    elif isinstance(s1, int) and isinstance(s2, int):
        # Handle integer indices
        return s1 == s2
    elif isinstance(s1, list) and isinstance(s2, list):
        if set(s1) & set(s2):
            return True
        return False
    else:
        raise TypeError("Unsupported slice type")


def check_slicing_tuple_intersection(
    s1: tuple[SlicingType, ...], s2: tuple[SlicingType, ...]
) -> bool:
    """For a tuple of SlicingType, check if all elements intersect."""
    if len(s1) != len(s2):
        raise NgioValueError("Slices must have the same length")
    return all(check_elem_intersection(a, b) for a, b in zip(s1, s2, strict=True))


def _sweep_axis_intervals(
    slices: list[tuple[SlicingType, ...]], axis: int
) -> list[tuple[float, float]] | None:
    """The [start, stop) interval of each region on `axis`, or `None`.

    `None` means the axis cannot be swept: a list selection is not an
    interval, and a stepped slice is not contiguous.
    """
    intervals: list[tuple[float, float]] = []
    for slicing_tuple in slices:
        sel = slicing_tuple[axis]
        if isinstance(sel, slice):
            if sel.step not in (None, 1):
                return None
            start = 0 if sel.start is None else sel.start
            stop = float("inf") if sel.stop is None else sel.stop
        elif isinstance(sel, int):
            start, stop = sel, sel + 1
        else:
            return None
        intervals.append((start, stop))
    return intervals


def check_if_regions_overlap(slices: Iterable[tuple[SlicingType, ...]]) -> bool:
    """Check whether any two slicing tuples select intersecting regions.

    Sort-and-sweep on the most discriminating axis: regions are sorted by
    their start there, and each is compared only against the ones still open
    at that start — for regions laid out on a grid (the iterator case) that
    is near-linear, where the all-pairs scan was a clean O(n²) and took over
    a second at ~2000 ROIs. Falls back to all pairs only when no axis is an
    interval (list selections or stepped slices on every axis).
    """
    slices = list(slices)
    if len(slices) < 2:
        return False
    if any(len(s) != len(slices[0]) for s in slices):
        raise NgioValueError("Slices must have the same length")

    # Sweep the axis with the most distinct starts: an axis every region
    # spans fully (a channel axis, say) keeps every region open at once and
    # degrades the sweep back to all pairs.
    best_intervals, best_distinct = None, 1
    for axis in range(len(slices[0])):
        intervals = _sweep_axis_intervals(slices, axis)
        if intervals is None:
            continue
        distinct = len({start for start, _ in intervals})
        if distinct > best_distinct:
            best_intervals, best_distinct = intervals, distinct

    if best_intervals is None:
        for it, (si, sj) in enumerate(_pairs_stream(slices)):
            if check_slicing_tuple_intersection(si, sj):
                return True
            if it == 10_000:
                warnings.warn(
                    "Performance Warning check_for_overlaps is O(n^2) and may be "
                    "slow for large numbers of regions.",
                    NgioUserWarning,
                    stacklevel=2,
                )
        return False

    order = sorted(range(len(slices)), key=lambda i: best_intervals[i][0])
    active: list[tuple[float, int]] = []
    for i in order:
        start, _ = best_intervals[i]
        # Regions whose interval closed before this start can never overlap
        # anything later — the starts only grow from here.
        active = [(stop, j) for stop, j in active if stop > start]
        for _, j in active:
            if check_slicing_tuple_intersection(slices[j], slices[i]):
                return True
        active.append((best_intervals[i][1], i))
    return False


##############################################################
#
# Check chunk overlaps
#
##############################################################


def _normalize_slice(slc: slice, size: int) -> tuple[int, int]:
    if slc.step not in (None, 1):
        raise NgioValueError(f"Only step=1 slices supported, got step={slc.step}")
    start = 0 if slc.start is None else slc.start
    stop = size if slc.stop is None else slc.stop
    if start < 0 or stop < 0:
        raise NgioValueError("Negative slice bounds are not supported")
    # clamp to [0, size]
    start = min(start, size)
    stop = min(stop, size)
    if start > stop:
        # empty selection
        return (0, 0)
    return start, stop


def _chunk_indices_for_axis(sel: SlicingType, size: int, csize: int) -> list[int]:
    """From a selection for a single axis, return the list chunk indices touched."""
    if isinstance(sel, slice):
        start, stop = _normalize_slice(sel, size)
        if start >= stop:  # empty
            return []
        first = start // csize
        last = (stop - 1) // csize
        return list(range(first, last + 1))

    if isinstance(sel, int):
        if sel < 0 or sel >= size:
            raise IndexError(f"index {sel} out of bounds for axis of size {size}")
        return [sel // csize]

    if isinstance(sel, list):
        if not sel:
            return []
        chunks_hit = {}
        for v in sel:
            if not isinstance(v, int):
                raise TypeError("Only integers allowed inside tuple selections")
            if v < 0 or v >= size:
                raise IndexError(f"index {v} out of bounds for axis of size {size}")
            chunks_hit[v // csize] = None
        return sorted(chunks_hit.keys())

    raise TypeError(f"Unsupported index type: {type(sel)!r}")


def compute_slice_chunks(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    slicing_tuple: tuple[SlicingType, ...],
) -> set[tuple[int, ...]]:
    """Compute the set of chunk coordinates touched by `slicing_tuple`.

    Args:
        shape: overall array shape (s1, s2, ...)
        chunks: chunk shape (c1, c2, ...)
        slicing_tuple: tuple of slices, ints, or tuples of ints
    """
    if len(slicing_tuple) != len(shape):
        raise NgioValueError(
            f"key must have {len(shape)} items, got {len(slicing_tuple)}"
        )

    per_axis_chunks: list[list[int]] = [
        _chunk_indices_for_axis(sel, size, csize)
        for sel, size, csize in zip(slicing_tuple, shape, chunks, strict=True)
    ]

    # If any axis yields no chunks, the overall selection is empty.
    if any(len(ax) == 0 for ax in per_axis_chunks):
        return set()

    return {tuple(idx) for idx in product(*per_axis_chunks)}


ChunkRect: TypeAlias = tuple[tuple[int, int], ...]
"""Per-axis inclusive `(first, last)` chunk-index ranges.

Empty selections are represented as `None` wherever a `ChunkRect` is expected,
never as a `first > last` range: an empty selection touches no chunks and must
conflict with nothing.
"""


def _chunk_range_for_axis(
    sel: SlicingType, size: int, csize: int
) -> tuple[int, int] | None:
    """The inclusive (first, last) chunk-index range touched on a single axis.

    Exact for `slice` and `int` selections (their chunk footprint is contiguous,
    see `_chunk_indices_for_axis`). For `list[int]` selections this is the
    bounding box — a conservative over-approximation. Returns `None` for an
    empty selection.
    """
    if isinstance(sel, slice):
        start, stop = _normalize_slice(sel, size)
        if start >= stop:  # empty
            return None
        return start // csize, (stop - 1) // csize

    if isinstance(sel, int):
        if sel < 0 or sel >= size:
            raise IndexError(f"index {sel} out of bounds for axis of size {size}")
        return sel // csize, sel // csize

    if isinstance(sel, list):
        if not sel:
            return None
        for v in sel:
            if not isinstance(v, int):
                raise TypeError("Only integers allowed inside tuple selections")
            if v < 0 or v >= size:
                raise IndexError(f"index {v} out of bounds for axis of size {size}")
        return min(sel) // csize, max(sel) // csize

    raise TypeError(f"Unsupported index type: {type(sel)!r}")


def compute_chunk_rect(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    slicing_tuple: tuple[SlicingType, ...],
) -> ChunkRect | None:
    """Bounding rectangle, in chunk-index space, of the chunks a selection touches.

    Exact for `slice`/`int` selections; for `list[int]` selections the per-axis
    bounding box is used — a conservative over-approximation that may include
    chunks the selection does not touch, never fewer.

    Args:
        shape: Overall array shape.
        chunks: Chunk (or shard) shape — the conflict granularity.
        slicing_tuple: Tuple of slices, ints, or lists of ints.

    Returns:
        Per-axis inclusive `(first, last)` chunk-index ranges, or `None` when
        the selection is empty on any axis (touches no chunks).
    """
    if len(slicing_tuple) != len(shape):
        raise NgioValueError(
            f"key must have {len(shape)} items, got {len(slicing_tuple)}"
        )

    per_axis_ranges = []
    for sel, size, csize in zip(slicing_tuple, shape, chunks, strict=True):
        ax_range = _chunk_range_for_axis(sel, size, csize)
        if ax_range is None:
            return None
        per_axis_ranges.append(ax_range)
    return tuple(per_axis_ranges)


def chunk_rects_intersect(rect_a: ChunkRect | None, rect_b: ChunkRect | None) -> bool:
    """Whether two chunk rectangles share at least one chunk.

    `None` encodes an empty selection and never intersects anything.
    """
    if rect_a is None or rect_b is None:
        return False
    if len(rect_a) != len(rect_b):
        raise NgioValueError("Chunk rectangles must have the same rank")
    return all(
        a_first <= b_last and b_first <= a_last
        for (a_first, a_last), (b_first, b_last) in zip(rect_a, rect_b, strict=True)
    )


def check_if_chunks_overlap(
    slices: Iterable[tuple[SlicingType, ...]],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> bool:
    """Check whether any two slicing tuples touch a common chunk.

    Two regions share a chunk exactly when some chunk coordinate appears in
    more than one region's set, so one accumulated set answers it in a single
    pass — the pairwise set intersections were O(n²).
    """
    seen: set[tuple[int, ...]] = set()
    for si in slices:
        slice_chunks = compute_slice_chunks(shape, chunks, si)
        if not seen.isdisjoint(slice_chunks):
            return True
        seen |= slice_chunks
    return False
