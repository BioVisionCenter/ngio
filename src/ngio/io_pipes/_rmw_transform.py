"""The read-modify-write transform base.

A transform on this base reads the destination back on the write path and
merges the incoming patch with what is already there, instead of overwriting
it. `MaskTransform` is the archetype — it keeps the on-disk pixels outside its
mask — and `MergeTransform` is the general form.

Reading the destination brings three rules with it. The pipes enforce the
first two at construction:

- **Terminal.** The read-back replays the transforms placed *before* this one,
  so a transform placed after it would be skipped during that read and the
  merge would compare two arrays taken from different points in the chain.
- **One per chain.** The write path folds the chain outermost-first, so a
  second read-modify-write transform would read the destination again and
  merge an already-merged array. There is no coherent reading of that.
- **Never read outside your own write footprint.** ngio's parallel mappers are
  safe because each unit owns its write units outright, with no lock; a read
  confined to `ctx.slicing` stays inside that ownership, and a read that
  reaches past it — a border of context, say — breaks the guarantee silently.
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from dask.array import Array as DaskArray

from ngio.io_pipes._io_pipe_ops import read_as_dask, read_as_numpy
from ngio.io_pipes._ops_transforms import (
    IoPipeContext,
    TransformProtocol,
    normalize_transforms,
)

# The union rather than a constrained TypeVar, matching the mask transform:
# these methods dispatch on the backend at runtime, so the two arms genuinely
# differ and a TypeVar would only promise a link the bodies do not keep.
ArrayLike: TypeAlias = np.ndarray | DaskArray


class ReadModifyWriteTransform(ABC):
    """Base for transforms that merge their patch with the destination.

    Subclasses implement `reconcile`; `on_get` defaults to the identity, so a
    subclass that only changes the write path implements one method.

    The transforms placed before this one in the chain have to be replayed
    when reading the destination back, or the merge would compare a patch that
    has been through them with on-disk data that has not. The pipe supplies
    them at construction, so this is wired for you; `set_transforms` is the
    override for a pipe-less caller driving the transform directly.
    """

    def __init__(
        self, *, set_transforms: Sequence[TransformProtocol] | None = None
    ) -> None:
        """Build the transform.

        Args:
            set_transforms: The transforms to replay when reading the
                destination back. Leave it unset: the pipe binds the chain
                preceding this transform, which is what this should be.
        """
        self._explicit_set_transforms = normalize_transforms(set_transforms)
        self._bound_set_transforms: Sequence[TransformProtocol] | None = None

    @property
    def _set_transforms(self) -> Sequence[TransformProtocol] | None:
        """The chain replayed on the read-back: explicit wins over bound."""
        if self._explicit_set_transforms is not None:
            return self._explicit_set_transforms
        return self._bound_set_transforms

    def bind_preceding(
        self, transforms: Sequence[TransformProtocol] | None
    ) -> "ReadModifyWriteTransform":
        """Return a copy bound to the transforms placed before this one.

        A copy rather than a mutation: one transform instance is meant to be
        reusable across pipes (the mask docstring promises it across ROIs), and
        two pipes with different chains must not fight over one binding.

        An explicit `set_transforms` is left alone — the caller said what to
        replay, so the pipe does not overrule it.
        """
        if self._explicit_set_transforms is not None:
            return self
        bound = copy.copy(self)
        bound._bound_set_transforms = normalize_transforms(transforms) or None
        return bound

    def read_existing(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Read the destination region, on the backend of `array`.

        Reads exactly `ctx.slicing` — the region the pipe is about to write —
        which is what keeps this safe under the parallel mappers.
        """
        if isinstance(array, DaskArray):
            return read_as_dask(ctx, self._set_transforms)
        return read_as_numpy(ctx, self._set_transforms)

    def on_get(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Pass the array through unchanged."""
        return array

    def on_set(self, array: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
        """Merge the patch with the destination read back off disk."""
        return self.reconcile(self.read_existing(array, ctx), array, ctx)

    @abstractmethod
    def reconcile(
        self, existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
    ) -> ArrayLike:
        """Combine the on-disk data with the patch about to be written.

        Args:
            existing: The destination region as it is on disk, read through
                the transforms preceding this one.
            patch: The incoming data, already through those same transforms.
            ctx: The pipe's context — axes, slicing, and the ROI when there
                is one.

        Returns:
            The array to write, on the same backend as `patch`.
        """
        ...
