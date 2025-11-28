import warnings
from collections.abc import Sequence
from enum import Enum

import dask.array as da
import numpy as np

from ngio.utils import NgioValueError


class Action(str, Enum):
    NONE = "none"
    PAD = "pad"
    TRIM = "trim"
    RESCALING = "rescaling"


def _compute_pad_widths(
    array_shape: tuple[int, ...],
    actions: list[Action],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    pad_def = []
    for act, s, ts in zip(actions, array_shape, target_shape, strict=True):
        if act == Action.PAD:
            total_pad = ts - s
            before = total_pad // 2
            after = total_pad - before
            pad_def.append((before, after))
        else:
            pad_def.append((0, 0))
    warnings.warn(
        f"Images have a different shape ({array_shape} vs {target_shape}). "
        f"Resolving by padding: {pad_def}",
        stacklevel=2,
    )
    return tuple(pad_def)


def _numpy_pad(
    array: np.ndarray,
    actions: list[Action],
    target_shape: tuple[int, ...],
    pad_mode: str = "constant",
    constant_values: int | float = 0,
) -> np.ndarray:
    if all(act != Action.PAD for act in actions):
        return array
    pad_widths = _compute_pad_widths(array.shape, actions, target_shape)
    return np.pad(array, pad_widths, mode=pad_mode, constant_values=constant_values)  # type: ignore


def _dask_pad(
    array: da.Array,
    actions: list[Action],
    target_shape: tuple[int, ...],
    pad_mode: str = "constant",
    constant_values: int | float = 0,
) -> da.Array:
    if all(act != Action.PAD for act in actions):
        return array
    shape = tuple(int(s) for s in array.shape)
    pad_widths = _compute_pad_widths(shape, actions, target_shape)
    return da.pad(array, pad_widths, mode=pad_mode, constant_values=constant_values)


def _compute_trim_slices(
    array_shape: tuple[int, ...],
    actions: list[Action],
    target_shape: tuple[int, ...],
) -> tuple[slice, ...]:
    slices = []
    for act, s, ts in zip(actions, array_shape, target_shape, strict=True):
        if act == Action.TRIM:
            slices.append(slice(0, ts))
        else:
            slices.append(slice(0, s))

    warnings.warn(
        f"Images have a different shape ({array_shape} vs {target_shape}). "
        f"Resolving by trimming: {slices}",
        stacklevel=2,
    )
    return tuple(slices)


def _numpy_trim(
    array: np.ndarray, actions: list[Action], target_shape: tuple[int, ...]
) -> np.ndarray:
    if all(act != Action.TRIM for act in actions):
        return array
    slices = _compute_trim_slices(array.shape, actions, target_shape)
    return array[tuple(slices)]


def _dask_trim(
    array: da.Array, actions: list[Action], target_shape: tuple[int, ...]
) -> da.Array:
    if all(act != Action.TRIM for act in actions):
        return array
    shape = tuple(int(s) for s in array.shape)
    slices = _compute_trim_slices(shape, actions, target_shape)
    return array[tuple(slices)]


def _compute_rescaling_shape(
    array_shape: tuple[int, ...],
    actions: list[Action],
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    rescaling_shape = []
    factor = []
    for act, s, ts in zip(actions, array_shape, target_shape, strict=True):
        if act == Action.RESCALING:
            rescaling_shape.append(ts)
            factor.append(ts / s)
        else:
            rescaling_shape.append(s)
            factor.append(1.0)

    warnings.warn(
        f"Images have a different shape ({array_shape} vs {target_shape}). "
        f"Resolving by scaling with factors {factor}.",
        stacklevel=2,
    )
    return tuple(rescaling_shape)


def _numpy_rescaling(
    array: np.ndarray, actions: list[Action], target_shape: tuple[int, ...]
) -> np.ndarray:
    if all(act != Action.RESCALING for act in actions):
        return array
    from ngio.common._zoom import numpy_zoom

    rescaling_shape = _compute_rescaling_shape(array.shape, actions, target_shape)
    return numpy_zoom(source_array=array, target_shape=rescaling_shape, order="nearest")


def _dask_rescaling(
    array: da.Array, actions: list[Action], target_shape: tuple[int, ...]
) -> da.Array:
    if all(act != Action.RESCALING for act in actions):
        return array
    from ngio.common._zoom import dask_zoom

    shape = tuple(int(s) for s in array.shape)
    rescaling_shape = _compute_rescaling_shape(shape, actions, target_shape)
    return dask_zoom(source_array=array, target_shape=rescaling_shape, order="nearest")


def _check_axes(array_shape, reference_shape, array_axes, reference_axes):
    if len(array_shape) != len(array_axes):
        raise NgioValueError(
            f"Array shape {array_shape} and reference axes {array_axes} "
            "must have the same number of dimensions."
        )
    if len(reference_shape) != len(reference_axes):
        raise NgioValueError(
            f"Reference shape {reference_shape} and reference axes {reference_axes} "
            "must have the same number of dimensions."
        )

    # Check if the array axes are a subset of the target axes
    diff = set(array_axes) - set(reference_axes)
    if diff:
        raise NgioValueError(
            f"Array axes {array_axes} are not a subset "
            f"of reference axes {reference_axes}"
        )

    # Array must be smaller or equal in number of dimensions
    if len(array_axes) > len(reference_axes):
        raise NgioValueError(
            f"Array has more dimensions ({len(array_axes)}) "
            f"than reference ({len(reference_axes)}). "
            "Cannot match shapes if the array has more dimensions."
        )


def _compute_reshape_and_actions(
    array_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    array_axes: list[str],
    reference_axes: list[str],
    tolerance: int = 1,
    allow_rescaling: bool = True,
) -> tuple[tuple[int, ...], list[Action]]:
    # Reshape array to match reference shape
    # And determine actions to be taken
    # to match the shapes
    reshape_tuple = []
    actions = []
    errors = []
    left_pointer = 0
    for ref_ax, ref_shape in zip(reference_axes, reference_shape, strict=True):
        if ref_ax not in array_axes:
            reshape_tuple.append(1)
            actions.append(Action.NONE)
        elif ref_ax == array_axes[left_pointer]:
            s2 = array_shape[left_pointer]
            reshape_tuple.append(s2)
            left_pointer += 1

            if s2 == ref_shape or s2 == 1:
                actions.append(Action.NONE)
            elif s2 < ref_shape:
                if (ref_shape - s2) <= tolerance:
                    actions.append(Action.PAD)
                elif allow_rescaling:
                    actions.append(Action.RESCALING)
                else:
                    errors.append(
                        f"Cannot pad axis={ref_ax}:{s2}->{ref_shape} "
                        "because shape difference is outside tolerance "
                        f"{tolerance}."
                    )
            elif s2 > ref_shape:
                if (s2 - ref_shape) <= tolerance:
                    actions.append(Action.TRIM)
                elif allow_rescaling:
                    actions.append(Action.RESCALING)
                else:
                    errors.append(
                        f"Cannot trim axis={ref_ax}:{s2}->{ref_shape} "
                        "because shape difference is outside tolerance "
                        f"{tolerance}."
                    )
            else:
                raise RuntimeError("Unreachable code reached.")
        else:
            raise NgioValueError(
                f"Axes order mismatch {array_axes} -> {reference_axes}. "
                "Cannot match shapes if the order is different."
            )
    if errors:
        raise NgioValueError(
            "Array shape cannot be matched to reference shape:\n\n".join(errors)
        )
    return tuple(reshape_tuple), actions


def numpy_match_shape(
    array: np.ndarray,
    reference_shape: tuple[int, ...],
    array_axes: Sequence[str],
    reference_axes: Sequence[str],
    tolerance: int = 1,
    pad_mode: str = "constant",
    pad_values: int | float = 0,
    allow_rescaling: bool = True,
):
    """Match the shape of a numpy array to a reference shape.

    This function will reshape, pad, trim and broadcast the input array
    to match the reference shape. If the shapes cannot be matched within
    the specified tolerance, an error is raised.

    The reference axes must be a superset of the array axes, and the order
    of the axes must be the same.

    Args:
        array (np.ndarray): The input array to be reshaped.
        reference_shape (tuple[int, ...]): The target shape to match.
        array_axes (Sequence[str]): The axes names of the input array.
        reference_axes (Sequence[str]): The axes names of the reference shape.
        tolerance (int): The maximum number of pixels by which dimensions
            can differ when matching shapes.
        allow_broadcast (bool): If True, allow broadcasting new dimensions to
            match the reference shape. If False, single-dimension axes will
            be left as is.
        pad_mode (str): The mode to use for padding. See numpy.pad for options.
        pad_values (int | float): The constant value to use for padding if
            pad_mode is 'constant'.
        allow_rescaling (bool): If True, when the array differs more than the
            tolerance, it will be rescalingd to the reference shape. If False,
            an error will be raised.
    """
    _check_axes(
        array_shape=array.shape,
        reference_shape=reference_shape,
        array_axes=array_axes,
        reference_axes=reference_axes,
    )
    if array.shape == reference_shape:
        # Shapes already match
        return array

    array_axes = list(array_axes)
    reference_axes = list(reference_axes)

    reshape_tuple, actions = _compute_reshape_and_actions(
        array_shape=array.shape,
        reference_shape=reference_shape,
        array_axes=array_axes,
        reference_axes=reference_axes,
        tolerance=tolerance,
        allow_rescaling=allow_rescaling,
    )
    array = array.reshape(reshape_tuple)
    array = _numpy_rescaling(array=array, actions=actions, target_shape=reference_shape)
    array = _numpy_pad(
        array=array,
        actions=actions,
        target_shape=reference_shape,
        pad_mode=pad_mode,
        constant_values=pad_values,
    )
    array = _numpy_trim(array=array, actions=actions, target_shape=reference_shape)
    return array


def dask_match_shape(
    array: da.Array,
    reference_shape: tuple[int, ...],
    array_axes: Sequence[str],
    reference_axes: Sequence[str],
    tolerance: int = 1,
    pad_mode: str = "constant",
    pad_values: int | float = 0,
    allow_rescaling: bool = True,
) -> da.Array:
    """Match the shape of a dask array to a reference shape.

    This function will reshape, pad, trim and broadcast the input array
    to match the reference shape. If the shapes cannot be matched within
    the specified tolerance, an error is raised.

    The reference axes must be a superset of the array axes, and the order
    of the axes must be the same.

    Args:
        array (da.Array): The input array to be reshaped.
        reference_shape (tuple[int, ...]): The target shape to match.
        array_axes (Sequence[str]): The axes names of the input array.
        reference_axes (Sequence[str]): The axes names of the reference shape.
        tolerance (int): The maximum number of pixels by which dimensions
            can differ when matching shapes.
        pad_mode (str): The mode to use for padding. See numpy.pad for options.
        pad_values (int | float): The constant value to use for padding if
            pad_mode is 'constant'.
        allow_rescaling (bool): If True, when the array differs more than the
            tolerance, it will be rescalingd to the reference shape. If False,
            an error will be raised.
    """
    array_shape = tuple(int(s) for s in array.shape)
    _check_axes(
        array_shape=array_shape,
        reference_shape=reference_shape,
        array_axes=array_axes,
        reference_axes=reference_axes,
    )
    if array_shape == reference_shape:
        # Shapes already match
        return array
    array_axes = list(array_axes)
    reference_axes = list(reference_axes)

    reshape_tuple, actions = _compute_reshape_and_actions(
        array_shape=tuple(int(s) for s in array.shape),
        reference_shape=reference_shape,
        array_axes=array_axes,
        reference_axes=reference_axes,
        tolerance=tolerance,
        allow_rescaling=allow_rescaling,
    )
    array = da.reshape(array, reshape_tuple)
    array = _dask_rescaling(array=array, actions=actions, target_shape=reference_shape)
    array = _dask_pad(
        array=array,
        actions=actions,
        target_shape=reference_shape,
        pad_mode=pad_mode,
        constant_values=pad_values,
    )
    array = _dask_trim(array=array, actions=actions, target_shape=reference_shape)
    return array
