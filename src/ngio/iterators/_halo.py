"""Writing back the core of a haloed read.

An iterator with a halo reads a grown region and writes only the ROI it was
grown from, so the extra context feeds the computation without landing on
disk. The read side is just a bigger ROI; this module is the write side, which
crops the halo off the patch before handing it to the real setter.

The crop is a setter *wrapper* rather than a transform on purpose. Writing
folds the transform chain outermost-first, so a terminal crop transform would
run *after* a read-modify-write transform's read-back — which reads at the
core's shape and would be handed a still-grown patch. Keeping the crop outside
the chain sidesteps the ordering, and leaves the one-terminal-read-modify-write
slot free for a mask or a merge.
"""

from typing import Any, Generic, TypeVar, cast

import numpy as np
from dask.array import Array as DaskArray

from ngio.common._roi import Roi
from ngio.io_pipes._io_pipes_types import DataSetterProtocol
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.iterators._rois_utils import HaloMargins
from ngio.utils import NgioValueError

# Constrained rather than open: the crop indexes the patch, so it needs the
# array API of whichever backend the iterator is running on.
T = TypeVar("T", np.ndarray, DaskArray)


def _core_size(slicing_ops: SlicingOps, axis_name: str) -> int | None:
    """The length the setter will write along `axis_name`, if it is knowable."""
    if axis_name not in slicing_ops.on_disk_axes:
        return None
    selection = slicing_ops.get(axis_name, normalize=True)
    if isinstance(selection, slice) and selection.step in (None, 1):
        start = 0 if selection.start is None else selection.start
        if selection.stop is not None:
            return selection.stop - start
    return None


class HaloCroppingSetter(Generic[T]):
    """A setter that crops a haloed patch down to the region it writes.

    Delegates everything else to the wrapped setter — including `zarr_array`
    and `slicing_ops`, so the mappers' write-footprint check and the chunk
    overlap gate see the real write target and are unaffected by the halo.
    """

    def __init__(self, setter: DataSetterProtocol[T], margins: HaloMargins) -> None:
        """Wrap `setter`, cropping `margins` pixels off each side per axis."""
        self._setter = setter
        self._margins = margins

    def __repr__(self) -> str:
        return f"HaloCroppingSetter({self._setter!r}, margins={self._margins})"

    @property
    def zarr_array(self):
        """The wrapped setter's target array."""
        return self._setter.zarr_array

    @property
    def slicing_ops(self) -> SlicingOps:
        """The wrapped setter's slicing: the core region, not the grown one."""
        return self._setter.slicing_ops

    @property
    def axes_ops(self) -> AxesOps:
        """The wrapped setter's axes operations."""
        return self._setter.axes_ops

    @property
    def transforms(self):
        """The wrapped setter's transform chain."""
        return self._setter.transforms

    @property
    def roi(self) -> Roi:
        """The core ROI being written."""
        return self._setter.roi

    @property
    def margins(self) -> HaloMargins:
        """The applied `(before, after)` halo per axis."""
        return self._margins

    def __call__(self, patch: T) -> None:
        return self.set(patch)

    def set(self, patch: T) -> None:
        """Crop the halo off `patch`, then write it."""
        self._setter.set(self._crop(patch))

    def _crop(self, patch: T) -> T:
        axes = self._setter.axes_ops.output_axes
        shape = tuple(int(size) for size in patch.shape)
        if len(axes) != len(shape):
            raise NgioValueError(
                f"Halo crop expected a patch over axes {axes} but got one with "
                f"{len(shape)} dimensions (shape {shape})."
            )

        selection = []
        for axis_name, size in zip(axes, shape, strict=True):
            before, after = self._margins.get(axis_name, (0, 0))
            if before or after:
                core = _core_size(self._setter.slicing_ops, axis_name)
                if core is None:
                    raise NgioValueError(
                        f"Cannot crop the halo along '{axis_name}': the width "
                        "the setter will write is not a plain contiguous "
                        f"range ({self._setter.slicing_ops.get(axis_name)!r}), "
                        "so the margin cannot be checked. Drop the halo on "
                        "this axis, or select it with a simple slice."
                    )
                if size - before - after != core:
                    raise NgioValueError(
                        f"Halo crop along '{axis_name}' expected a patch of "
                        f"{core + before + after} px — the {core} px written "
                        f"plus the {before}+{after} px halo — but got {size}. "
                        "Either the function returned a core-sized array "
                        "instead of the grown region it was handed, or a "
                        "shape-changing transform (a zoom, say) sits in the "
                        "chain: halo margins are counted in the reference "
                        "image's pixels, so they do not survive a rescale. "
                        "To work at a lower resolution, iterate on a coarser "
                        "pyramid level instead of zooming on the data path."
                    )
            selection.append(slice(before, size - after))
        # Indexed through `Any`: the stubs describe only fixed-arity index
        # tuples, and the rank here is the patch's — known at runtime, not to
        # the checker. Both backends slice identically.
        cropped = cast("Any", patch)[tuple(selection)]
        return cast("T", cropped)
