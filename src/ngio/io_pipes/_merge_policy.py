"""How a patch combines with what is already on disk.

A transform is a function of the patch alone: the read chain applies it, the
write chain inverts it, and a list of them folds cleanly in either direction. A
*merge* is a function of the patch **and** the destination's current contents,
which is a different kind of thing and has no coherent position inside that
fold. It gets its own slot instead — `merge=` on the write, one per write, run
after the transform chain has finished.

Running it last also fixes where the merge happens. By that point the patch has
been through every transform's inverse, so it is in the array's own space; the
destination is read raw, with no chain replayed over it. Both sides are in the
same space by construction, which is what makes the comparison meaningful and
keeps protected pixels byte-identical rather than round-tripped through a
transform's inverse.
"""

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from ngio.io_pipes._ops_transforms import ArrayLike, IoPipeContext, elementwise
from ngio.utils import NgioValueError

MergeRule = Literal["max", "min", "sum", "keep_nonzero"]

MergeFunction: TypeAlias = Callable[[ArrayLike, ArrayLike, IoPipeContext], ArrayLike]


@runtime_checkable
class MergePolicy(Protocol):
    """Combines the patch about to be written with the destination's contents.

    Both arrays arrive in the array's own space — the patch has been through
    the transform chain's inverse, the destination is read raw — and cover
    exactly the region being written.

    Note:
        Read only the region you are given. ngio's parallel mappers schedule
        ROIs that share a write unit into separate waves, so each running
        write has sole ownership of the pixels it reads; a policy reaching
        past `ctx.slicing` would break that silently.

    Compatibility policy: future ngio versions will only ever add *optional*
    members to this protocol, probed with `getattr` — a policy implementing
    `reconcile` keeps working.
    """

    def reconcile(
        self, existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
    ) -> ArrayLike:
        """Return the array to write.

        Args:
            existing: The destination region as it is on disk.
            patch: The incoming data, already through the transform chain.
            ctx: The pipe's context — axes, slicing, and the ROI if there is one.
        """
        ...


MergeInput: TypeAlias = MergeRule | MergeFunction | MergePolicy


def _merge_max(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return elementwise(np.maximum, existing, patch)


def _merge_min(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return elementwise(np.minimum, existing, patch)


def _merge_sum(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return existing + patch


def _merge_keep_nonzero(
    existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
) -> ArrayLike:
    return elementwise(np.where, patch != 0, patch, existing)


# `max`, `min` and `sum` are commutative and associative, so overlapping
# regions give the same answer whatever order they are visited in — exactly,
# because the merge runs in the array's own space and nothing is resampled
# between visits. `keep_nonzero` is not: it is "the last nonzero write wins".
# `sum` follows numpy's arithmetic: on integer dtypes it wraps on overflow.
_MERGE_RULES: dict[str, MergeFunction] = {
    "max": _merge_max,
    "min": _merge_min,
    "sum": _merge_sum,
    "keep_nonzero": _merge_keep_nonzero,
}


class FunctionMerge:
    """A merge policy wrapping a plain `(existing, patch, ctx)` callable."""

    def __init__(self, merge: MergeFunction, name: str | None = None) -> None:
        """Wrap `merge`; `name` is used for `repr` when it is a built-in rule."""
        self._merge = merge
        self._name = name

    def __repr__(self) -> str:
        return f"FunctionMerge({self._name or self._merge!r})"

    def reconcile(
        self, existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
    ) -> ArrayLike:
        """Apply the wrapped callable."""
        return self._merge(existing, patch, ctx)


def resolve_merge(merge: MergeInput | None) -> MergePolicy | None:
    """Turn a rule name, a callable, or a policy into a policy.

    Returns:
        The policy, or `None` when no merge was asked for — in which case the
        patch simply overwrites the region.

    Raises:
        NgioValueError: If `merge` is none of those things.
    """
    if merge is None:
        return None
    if isinstance(merge, str):
        if merge not in _MERGE_RULES:
            known = ", ".join(repr(name) for name in _MERGE_RULES)
            raise NgioValueError(
                f"Unknown merge rule {merge!r}; expected one of {known}, a "
                "callable (existing, patch, ctx) -> array, or a MergePolicy."
            )
        return FunctionMerge(_MERGE_RULES[merge], name=merge)
    if isinstance(merge, MergePolicy):
        return merge
    if callable(merge):
        return FunctionMerge(merge)
    raise NgioValueError(
        f"Cannot use {type(merge).__name__} as a merge: expected a rule name, "
        "a callable (existing, patch, ctx) -> array, or a MergePolicy."
    )
