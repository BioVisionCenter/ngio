import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._roi import Roi
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.utils import NgioDeprecationWarning, NgioValueError

ArrayT = TypeVar("ArrayT")

# The union rather than a constrained TypeVar for implementations that dispatch
# on the backend at runtime: the two arms genuinely differ, so a TypeVar would
# promise a link the bodies do not keep.
ArrayLike: TypeAlias = np.ndarray | DaskArray


def elementwise(op: Any, *args: Any) -> ArrayLike:
    """Apply a numpy ufunc or array-function across either backend.

    Dask implements numpy's dispatch protocols, so one call serves both; the
    call is opaque here on purpose, because numpy's stubs describe only the
    numpy side and would reject a dask argument at every call site.
    """
    return cast("ArrayLike", op(*args))


@dataclass(frozen=True, eq=False)
class IoPipeContext:
    """Everything an io pipe knows about one read/write path.

    Transforms may depend on the transform-stable fields: `axes`, `slicing`
    and `roi`. The remaining fields (`zarr_array`, `axes_ops`) are pipe
    plumbing, exposed for the pipes and adapters; transforms should not
    reach for them.

    Attributes:
        zarr_array: The zarr array the pipe reads from or writes to.
        slicing: Where the patch sits in the parent image.
        axes_ops: The axes operations mapping on-disk axes to the user's.
        roi: The physical ROI when the pipe has one, `None` otherwise.
    """

    zarr_array: zarr.Array
    slicing: SlicingOps
    axes_ops: AxesOps
    roi: Roi | None = None

    @property
    def axes(self) -> tuple[str, ...]:
        """Axis names of the array as a transform sees it."""
        return tuple(self.axes_ops.output_axes)


#: Legacy alias for `IoPipeContext` — the pre-1.1 name for the context handed
#: to transforms. Kept importable indefinitely; new code should spell it
#: `IoPipeContext`.
TransformContext = IoPipeContext


@runtime_checkable
class TransformProtocol(Protocol[ArrayT]):
    """Protocol for a transform on the io pipes' data path.

    The same contract serves the numpy and the dask path: the backend is
    expressed by the array type. A numpy-only transform is complete; a
    transform supporting both backends dispatches on the array type inside
    `on_get`/`on_set`.

    Compatibility policy: future ngio versions will only ever add *optional*
    members to this protocol, probed with `getattr` — a transform satisfying
    today's two-method contract keeps working.
    """

    def on_get(self, array: ArrayT, ctx: IoPipeContext) -> ArrayT:
        """A transformation to be applied after reading an array."""
        ...

    def on_set(self, array: ArrayT, ctx: IoPipeContext) -> ArrayT:
        """The inverse transformation, applied before writing an array."""
        ...


@runtime_checkable
class _LegacyTransform(Protocol):
    """The pre-1.1 transform contract, kept alive via an adapter until 1.2."""

    def get_as_numpy_transform(
        self, array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ): ...

    def set_as_numpy_transform(
        self, array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ): ...


class _LegacyTransformAdapter:
    """Adapts a legacy 4-method transform to the `on_get`/`on_set` contract."""

    def __init__(self, wrapped: _LegacyTransform) -> None:
        self._wrapped = wrapped

    def __repr__(self) -> str:
        return f"_LegacyTransformAdapter({self._wrapped!r})"

    def _forward(self, method_name: str, array: ArrayT, ctx: IoPipeContext):
        method = getattr(self._wrapped, method_name, None)
        if method is None:
            raise NgioValueError(
                f"Transform '{type(self._wrapped).__name__}' does not implement "
                f"{method_name}(), which this data path requires."
            )
        return method(array, slicing_ops=ctx.slicing, axes_ops=ctx.axes_ops)

    def on_get(self, array: ArrayT, ctx: IoPipeContext) -> ArrayT:
        if isinstance(array, DaskArray):
            return self._forward("get_as_dask_transform", array, ctx)
        return self._forward("get_as_numpy_transform", array, ctx)

    def on_set(self, array: ArrayT, ctx: IoPipeContext) -> ArrayT:
        if isinstance(array, DaskArray):
            return self._forward("set_as_dask_transform", array, ctx)
        return self._forward("set_as_numpy_transform", array, ctx)


def normalize_transforms(
    transforms: Sequence[TransformProtocol] | None,
) -> Sequence[TransformProtocol] | None:
    """Validate transforms, adapting legacy ones with a deprecation warning.

    Returns:
        The same transforms, with legacy-style ones wrapped in an adapter.

    Raises:
        NgioValueError: If an object satisfies neither the current nor the
            legacy transform contract.
    """
    if transforms is None:
        return None

    normalized: list[TransformProtocol] = []
    for transform in transforms:
        if isinstance(transform, TransformProtocol):
            normalized.append(transform)
        elif isinstance(transform, _LegacyTransform):
            warnings.warn(
                f"Transform '{type(transform).__name__}' implements the legacy "
                "get_as_*_transform/set_as_*_transform contract, which is "
                "deprecated and will be removed in ngio=1.2. Implement "
                "on_get/on_set instead (see ngio.transforms.TransformProtocol).",
                NgioDeprecationWarning,
                stacklevel=2,
            )
            normalized.append(_LegacyTransformAdapter(transform))
        else:
            raise NgioValueError(
                f"Transform '{type(transform).__name__}' does not implement the "
                "transform protocol (on_get / on_set, see "
                "ngio.transforms.TransformProtocol)."
            )
    return normalized


def _require_backend_type(
    array: object, expected_type: type, transform: TransformProtocol
) -> None:
    if not isinstance(array, expected_type):
        raise NgioValueError(
            f"Transform '{type(transform).__name__}' returned a "
            f"{type(array).__name__} instead of a {expected_type.__name__} — "
            "it does not support this data path (e.g. a numpy-only transform "
            "used through the dask API, or a transform materialising a dask "
            "array)."
        )


def apply_get_transforms(
    array: ArrayT,
    *,
    ctx: IoPipeContext,
    transforms: Sequence[TransformProtocol] | None,
    expected_type: type,
) -> ArrayT:
    """Fold an array read from disk through the transforms' `on_get`."""
    if transforms is None:
        return array

    for transform in transforms:
        array = transform.on_get(array, ctx)
        _require_backend_type(array, expected_type, transform)
    return array


def apply_set_transforms(
    array: ArrayT,
    *,
    ctx: IoPipeContext,
    transforms: Sequence[TransformProtocol] | None,
    expected_type: type,
) -> ArrayT:
    """Fold an array about to be written through the transforms' `on_set`."""
    if transforms is None:
        return array

    # Reading applies [A, B] as B(A(x)), so writing must invert the chain
    # outermost-first: A_inv(B_inv(y)).
    for transform in reversed(transforms):
        array = transform.on_set(array, ctx)
        _require_backend_type(array, expected_type, transform)
    return array
