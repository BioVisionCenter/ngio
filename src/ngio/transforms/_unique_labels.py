from typing import cast

import dask.array as da
import numpy as np

from ngio.io_pipes import IoPipeContext
from ngio.io_pipes._ops_transforms import ArrayLike, elementwise
from ngio.utils import NgioValueError


class UniqueLabelsTransform:
    """Shift a patch's label ids into a block of their own.

    Each region gets a disjoint slice of the id space — `block_index` picks the
    slice, `block_size` sizes it — so labels segmented independently in
    different regions can never collide when they land in one array. Background
    (`0`) is left alone.

    The offset is derived, not counted, which is what makes this usable in
    parallel: there is no shared state to synchronize, it works under
    `ProcessMapper` where a counter could not, and it is **idempotent** — a
    region re-run after a failure produces exactly the ids it produced before.

    `on_get` subtracts the same offset, so a read through this transform hands
    back the region-local ids that were written. That only means anything for a
    region whose ids this transform assigned; reading across several blocks
    gives values from a mix of id spaces.

    Composes with a merge: `transforms=[UniqueLabelsTransform(...)]` alongside
    `merge="max"` or `merge=MaskMerge(...)` does what it looks like, because the
    merge runs after this transform and against raw on-disk ids rather than ids
    this transform has shifted.

    Example:
        ```python
        # Region 4's labels 1, 2, 3 are written as 4001, 4002, 4003.
        label.set_roi(roi, patch, transforms=[UniqueLabelsTransform(1000, 4)])
        ```
    """

    def __init__(self, block_size: int, block_index: int | None = None) -> None:
        """Build the transform.

        Args:
            block_size: How many ids each region is given. Must exceed the
                largest label any single region can produce — an id at or above
                it would land in the next region's block.
            block_index: Which block this region gets. Leave unset inside a
                masked iterator, where the ROI's own label supplies it; pass it
                explicitly for grid regions, which carry no label.

        Raises:
            NgioValueError: If `block_size` is not positive, or `block_index`
                is negative.
        """
        if block_size <= 0:
            raise NgioValueError(f"block_size must be > 0, got {block_size}.")
        if block_index is not None and block_index < 0:
            raise NgioValueError(f"block_index must be >= 0, got {block_index}.")
        self._block_size = block_size
        self._block_index = block_index

    def __repr__(self) -> str:
        return (
            f"UniqueLabelsTransform(block_size={self._block_size}, "
            f"block_index={self._block_index})"
        )

    def _offset(self, ctx: IoPipeContext) -> int:
        """The first id of this region's block."""
        if self._block_index is not None:
            return self._block_index * self._block_size
        if ctx.roi is not None and ctx.roi.label is not None:
            return ctx.roi.label * self._block_size
        raise NgioValueError(
            "UniqueLabelsTransform needs a block index and this call carries "
            "no ROI label to take one from. Pass `block_index` explicitly — "
            "for a grid of regions, the region's position in the iterator's "
            "`rois` is the natural choice."
        )

    def _check_fits(self, array: np.ndarray, offset: int) -> None:
        """Refuse ids that would not survive the array's dtype or its block."""
        dtype = array.dtype
        if not np.issubdtype(dtype, np.integer):
            raise NgioValueError(
                f"UniqueLabelsTransform needs an integer patch, got {dtype}. "
                "Label images are integer-typed; a float patch here usually "
                "means the segmentation was not cast before writing."
            )
        largest = int(np.max(array)) if array.size else 0
        if largest >= self._block_size:
            raise NgioValueError(
                f"This patch holds label id {largest}, which does not fit in "
                f"a block of {self._block_size} ids: it would spill into the "
                "next region's block and collide with its labels. Raise "
                "block_size above the largest id any single region can "
                "produce."
            )
        limit = int(np.iinfo(dtype).max)
        if largest + offset > limit:
            raise NgioValueError(
                f"Offsetting this patch by {offset} would reach "
                f"{largest + offset}, past what {dtype} can hold ({limit}). "
                f"Lower block_size (currently {self._block_size}), use fewer "
                "blocks, or widen the label dtype — `derive_label` defaults to "
                "uint32, but a uint16 label runs out quickly."
            )

    def on_get(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Shift this region's labels back down to their local ids."""
        offset = self._offset(ctx)
        if offset == 0:
            return array
        return elementwise(np.where, array > 0, array - offset, 0)

    def on_set(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Shift the patch's labels up into this region's block."""
        offset = self._offset(ctx)
        if isinstance(array, np.ndarray):
            # Checked even when the offset is 0: block 0's ids can still spill
            # into block 1, and a float patch is wrong in any block.
            self._check_fits(array, offset)
        else:
            # The dask path cannot look at the values now; check each block as
            # it materializes, so overflow raises at compute time instead of
            # wrapping silently.
            def _checked(block: np.ndarray) -> np.ndarray:
                self._check_fits(block, offset)
                return block

            # Cast because dask's `map_blocks` stubs are a union the checker
            # cannot resolve at this call site; the result is a dask array.
            array = cast("ArrayLike", da.map_blocks(_checked, array, dtype=array.dtype))
        if offset == 0:
            return array
        return elementwise(np.where, array > 0, array + offset, 0)
