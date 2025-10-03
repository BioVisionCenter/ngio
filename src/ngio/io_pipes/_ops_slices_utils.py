from itertools import product
from typing import TypeAlias

from ngio.utils import ngio_logger

##############################################################
#
# Check slice overlaps
#
##############################################################

SlicingType: TypeAlias = slice | tuple[int, ...] | int


def check_elem_intersection(s1: SlicingType, s2: SlicingType) -> bool:
    """Compare if two SlicingType elements intersect.

    If they are a slice, check if they overlap.
    If they are integers, check if they are equal.
    If they are tuples, check if they have any common elements.
    """
    if not isinstance(s1, type(s2)):
        raise ValueError(
            f"Slices must be of the same type. Got {type(s1)} and {type(s2)}"
        )

    if isinstance(s1, slice) and isinstance(s2, slice):
        # Handle slice objects
        start1, stop1, step1 = s1.start or 0, s1.stop or float("inf"), s1.step or 1
        start2, stop2, step2 = s2.start or 0, s2.stop or float("inf"), s2.step or 1

        if step1 is not None and step2 != 1:
            raise NotImplementedError(
                "Intersection for slices with step != 1 is not implemented"
            )

        if step2 is not None and step1 != 1:
            raise NotImplementedError(
                "Intersection for slices with step != 1 is not implemented"
            )

        return not (stop1 <= start2 or stop2 <= start1)
    elif isinstance(s1, int) and isinstance(s2, int):
        # Handle integer indices
        return s1 == s2
    elif isinstance(s1, tuple) and isinstance(s2, tuple):
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
        raise ValueError("Slices must have the same length")
    return all(check_elem_intersection(a, b) for a, b in zip(s1, s2, strict=True))


def check_for_logical_overlaps(slices: list[tuple[SlicingType, ...]]) -> bool:
    """Check for overlaps in a list of slicing tuples using brute-force method.

    This is O(n^2) and not efficient for large lists.
    Returns True if any overlaps are found.
    """
    if len(slices) > 10_000:
        ngio_logger.warning(
            "Performance Warning check_for_overlaps is O(n^2) and may be slow for "
            f"large numbers of regions. Current number of regions: {len(slices)}."
        )
    for i, si in enumerate(slices):
        for sj in slices[i + 1 :]:
            overalap = check_slicing_tuple_intersection(si, sj)
            if overalap:
                return True
    return False


##############################################################
#
# Check chunk overlaps
#
##############################################################


def _normalize_slice(slc: slice, size: int) -> tuple[int, int]:
    if slc.step not in (None, 1):
        raise ValueError(f"Only step=1 slices supported, got step={slc.step}")
    start = 0 if slc.start is None else slc.start
    stop = size if slc.stop is None else slc.stop
    if start < 0 or stop < 0:
        raise ValueError("Negative slice bounds are not supported")
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

    if isinstance(sel, tuple):
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
        raise ValueError(f"key must have {len(shape)} items, got {len(slicing_tuple)}")

    per_axis_chunks: list[list[int]] = [
        _chunk_indices_for_axis(sel, size, csize)
        for sel, size, csize in zip(slicing_tuple, shape, chunks, strict=True)
    ]

    # If any axis yields no chunks, the overall selection is empty.
    if any(len(ax) == 0 for ax in per_axis_chunks):
        return set()

    return {tuple(idx) for idx in product(*per_axis_chunks)}


def check_for_chunks_overlap(
    slices: list[tuple[SlicingType, ...]],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> bool:
    """Check for overlaps in a list of slicing tuples using brute-force method.

    This is O(n^2) and not efficient for large lists.
    Returns True if any overlaps are found.
    """
    if len(slices) > 10_000:
        ngio_logger.warning(
            "Performance Warning check_for_chunks_overlaps is O(n^2) and may be "
            "slow for large numbers of regions. Current number of "
            f"regions: {len(slices)}."
        )
    slices_chunks = [compute_slice_chunks(shape, chunks, si) for si in slices]
    for i, si in enumerate(slices_chunks):
        for sj in slices_chunks[i + 1 :]:
            if si & sj:
                return True
    return False
