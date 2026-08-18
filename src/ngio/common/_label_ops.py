"""Array operations on label images: id offsets, overlap tallies, relabelling."""

import itertools
from collections.abc import Iterator

import numpy as np
import zarr

from ngio.utils import NgioValueError


def offset_labels(array: np.ndarray, offset: int) -> np.ndarray:
    """Shift every non-zero label by `offset`, leaving background at 0."""
    if offset == 0:
        return array
    return np.where(array > 0, array + offset, 0).astype(array.dtype)


def check_offset_fits(array: np.ndarray, offset: int, block_size: int) -> None:
    """Refuse an offset that would not survive the array's dtype or its block.

    Raises:
        NgioValueError: If the patch is not integer-typed, the largest id
            plus `offset` would exceed what the dtype can hold, or the largest
            id does not fit inside one block of `block_size` ids.
    """
    dtype = array.dtype
    if not np.issubdtype(dtype, np.integer):
        raise NgioValueError(
            f"Label ids must be integer-typed, got {dtype}. A float patch here "
            "usually means the segmentation was not cast before writing."
        )
    largest = int(np.max(array)) if array.size else 0
    if largest >= block_size:
        raise NgioValueError(
            f"This patch holds label id {largest}, which does not fit in a "
            f"block of {block_size} ids: it would spill into the next region's "
            "block and collide with its labels. Raise block_size above the "
            "largest id any single region can produce."
        )
    limit = int(np.iinfo(dtype).max)
    if largest + offset > limit:
        raise NgioValueError(
            f"Offsetting this patch by {offset} would reach {largest + offset}, "
            f"past what {dtype} can hold ({limit}). Lower block_size (currently "
            f"{block_size}), use fewer blocks, or widen the label dtype — "
            "`derive_label` defaults to uint32, but a uint16 label runs out fast."
        )


def overlap_iou(left: np.ndarray, right: np.ndarray) -> dict[tuple[int, int], float]:
    """Intersection-over-union of every id pair that co-occurs in two arrays.

    The two arrays are independent labellings of the *same* pixels — one tile's
    prediction over a band, and the neighbour's ids for that band. A high score
    means both tiles found the same object there.

    Background (`0`) takes part in neither the intersections nor the areas, so
    an object touching a lot of background is not penalised for it.

    Returns:
        `{(left_id, right_id): iou}` for pairs that intersect at all.
    """
    if left.shape != right.shape:
        raise NgioValueError(
            f"Cannot compare labellings of different shapes: {left.shape} vs "
            f"{right.shape}."
        )
    both = (left > 0) & (right > 0)
    if not both.any():
        return {}

    left_ids, right_ids = left[both], right[both]
    pairs = np.stack([left_ids, right_ids])
    unique_pairs, intersections = np.unique(pairs, axis=1, return_counts=True)

    # Areas are measured over the whole band, not just the intersecting pixels,
    # so a pair that agrees on a sliver of two large objects scores low.
    left_area = dict(zip(*np.unique(left[left > 0], return_counts=True), strict=True))
    right_area = dict(zip(*np.unique(right[right > 0], return_counts=True), strict=True))

    scores: dict[tuple[int, int], float] = {}
    for column, intersection in enumerate(intersections):
        a, b = int(unique_pairs[0, column]), int(unique_pairs[1, column])
        union = int(left_area[a]) + int(right_area[b]) - int(intersection)
        if union > 0:
            scores[(a, b)] = float(intersection) / union
    return scores


def relabel(array: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Apply an id mapping, leaving ids absent from it untouched.

    Uses a sorted-key binary search rather than a dense lookup table: ids are
    sparse by construction here (each tile writes into its own block), so a
    dense table would be sized by the largest id rather than by how many there
    are.
    """
    if not mapping:
        return array
    keys = np.fromiter(sorted(mapping), dtype=np.int64, count=len(mapping))
    values = np.array([mapping[int(key)] for key in keys], dtype=np.int64)

    positions = np.searchsorted(keys, array)
    positions = np.clip(positions, 0, len(keys) - 1)
    matched = keys[positions] == array
    return np.where(matched, values[positions], array).astype(array.dtype)


def chunk_selections(array: zarr.Array) -> Iterator[tuple[slice, ...]]:
    """Yield one index tuple per chunk, in a fixed order.

    Fixed order matters twice over: it keeps a pass over a large label bounded
    to one chunk at a time, and it makes `relabel_sequential` deterministic.
    """
    ranges = [
        range(0, size, chunk)
        for size, chunk in zip(array.shape, array.chunks, strict=True)
    ]
    for corner in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + chunk, size))
            for start, chunk, size in zip(corner, array.chunks, array.shape, strict=True)
        )


def relabel_sequential(
    array: zarr.Array, canonical: dict[int, int] | None = None
) -> int:
    """Rewrite the ids to a dense `1..N` in place, in a single pass.

    Numbers are handed out in **first-encounter order** over the chunk grid,
    not by sorting the original ids. That is what makes one pass enough: a
    sorted numbering cannot be decided until every id has been seen, so it needs
    one traversal to collect the ids and a second to write them. The ids are
    unique and gapless either way — only which object gets `1` differs, and it
    follows the array rather than whatever offsets happened to be assigned.

    Deterministic for a given chunking; a differently chunked copy of the same
    label numbers its objects in a different order.

    Args:
        array: The label array, rewritten in place.
        canonical: Optional id -> id map applied first, e.g. the equivalences a
            stitch resolved. Ids it does not mention are kept as they are.

    Returns:
        How many distinct objects were numbered.
    """
    dense: dict[int, int] = {}
    for selection in chunk_selections(array):
        block = np.asarray(array[selection])
        found = [int(value) for value in np.unique(block) if value]
        if not found:
            continue
        mapping = {}
        for old in found:
            root = canonical.get(old, old) if canonical else old
            if root not in dense:
                dense[root] = len(dense) + 1
            mapping[old] = dense[root]
        remapped = relabel(block, mapping)
        if not np.array_equal(remapped, block):
            array[selection] = remapped
    return len(dense)
