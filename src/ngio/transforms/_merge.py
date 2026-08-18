from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from ngio.io_pipes import IoPipeContext, TransformProtocol
from ngio.io_pipes._rmw_transform import ArrayLike, ReadModifyWriteTransform
from ngio.utils import NgioValueError

MergeRule = Literal["max", "min", "sum", "keep_nonzero"]

MergeFunction: TypeAlias = Callable[[ArrayLike, ArrayLike, IoPipeContext], ArrayLike]


def _elementwise(op: Callable[..., Any], *args: Any) -> ArrayLike:
    """Apply a numpy ufunc or array-function across either backend.

    Dask implements numpy's dispatch protocols, so one call serves both — but
    numpy's stubs only describe the numpy side, so the call is opaque here on
    purpose rather than suppressed at each site.
    """
    return cast("ArrayLike", op(*args))


def _merge_max(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return _elementwise(np.maximum, existing, patch)


def _merge_min(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return _elementwise(np.minimum, existing, patch)


def _merge_sum(existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext) -> ArrayLike:
    return existing + patch


def _merge_keep_nonzero(
    existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
) -> ArrayLike:
    return _elementwise(np.where, patch != 0, patch, existing)


# `max`, `min` and `sum` are commutative and associative, so overlapping
# regions give the same answer whatever order they are visited in.
# `keep_nonzero` is not: it is "the last nonzero write wins", which differs by
# visit order wherever two patches are both nonzero on the same pixel.
_MERGE_RULES: dict[str, MergeFunction] = {
    "max": _merge_max,
    "min": _merge_min,
    "sum": _merge_sum,
    "keep_nonzero": _merge_keep_nonzero,
}


class MergeTransform(ReadModifyWriteTransform):
    """Combine the patch with what is already on disk instead of overwriting.

    On read this is the identity. On write it reads the destination region
    back and hands both arrays to `merge`, whose result is written.

    `"max"`, `"min"` and `"sum"` are commutative and associative, so
    overlapping regions give the same answer whatever order they are written
    in. `"keep_nonzero"` is "the last nonzero write wins" and a custom `merge`
    promises nothing — for either, overlapping regions make the result depend
    on the visit order.

    Must be the last transform in the transforms list, and the only
    read-modify-write one — the pipes raise otherwise.

    Note:
        The read-back covers exactly the region being written, so this stays
        safe under `ThreadedMapper` and `ProcessMapper`: they already refuse
        two ROIs sharing a write unit, which is what gives each unit sole
        ownership of the pixels it reads.

    Example:
        ```python
        # Overlapping tiles keep the brighter value.
        image.set_roi(roi, patch, transforms=[MergeTransform("max")])


        # Or supply your own rule.
        def half(existing, patch, ctx):
            return (existing + patch) // 2


        image.set_roi(roi, patch, transforms=[MergeTransform(half)])
        ```
    """

    def __init__(
        self,
        merge: MergeRule | MergeFunction,
        *,
        set_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Build a merge transform.

        Args:
            merge: One of `"max"`, `"min"`, `"sum"`, `"keep_nonzero"`, or a
                callable `(existing, patch, ctx) -> array` returning an array
                on the same backend as `patch`.
            set_transforms: Override for the chain replayed on the read-back;
                the pipe binds it for you.

        Raises:
            NgioValueError: If `merge` is neither a known rule name nor
                callable.
        """
        super().__init__(set_transforms=set_transforms)
        if callable(merge):
            self._merge: MergeFunction = merge
            self._rule: str | None = None
        elif merge in _MERGE_RULES:
            self._merge = _MERGE_RULES[merge]
            self._rule = merge
        else:
            known = ", ".join(repr(name) for name in _MERGE_RULES)
            raise NgioValueError(
                f"Unknown merge rule {merge!r}; expected one of {known}, or a "
                "callable (existing, patch, ctx) -> array."
            )

    def __repr__(self) -> str:
        rule = self._rule if self._rule is not None else self._merge
        return f"MergeTransform({rule!r})"

    def reconcile(
        self, existing: ArrayLike, patch: ArrayLike, ctx: IoPipeContext
    ) -> ArrayLike:
        """Apply the merge rule to the on-disk data and the incoming patch."""
        return self._merge(existing, patch, ctx)
