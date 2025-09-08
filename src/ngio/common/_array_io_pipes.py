from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._array_io_utils import (
    SlicingInputType,
    TransformProtocol,
    apply_dask_axes_ops,
    apply_dask_transforms,
    apply_inverse_dask_transforms,
    apply_inverse_numpy_transforms,
    apply_numpy_axes_ops,
    apply_numpy_transforms,
    get_slice_as_dask,
    get_slice_as_numpy,
    set_dask_patch,
    set_numpy_patch,
    setup_from_disk_pipe,
    setup_to_disk_pipe,
)
from ngio.common._dimensions import Dimensions
from ngio.common._zoom import dask_zoom, numpy_zoom
from ngio.ome_zarr_meta.ngio_specs import SlicingOps

##############################################################
#
# Concrete "From Disk" Pipes
#
##############################################################


def _numpy_get_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None = None,
) -> np.ndarray:
    _array = get_slice_as_numpy(zarr_array, slice_tuple=slicing_ops.slice_tuple)
    _array = apply_numpy_axes_ops(
        _array,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )

    _array = apply_numpy_transforms(
        _array, transforms=transforms, slicing_ops=slicing_ops
    )
    return _array


def _dask_get_pipe(
    zarr_array: zarr.Array,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> DaskArray:
    _array = get_slice_as_dask(zarr_array, slice_tuple=slicing_ops.slice_tuple)
    _array = apply_dask_axes_ops(
        _array,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )

    _array = apply_dask_transforms(
        _array, transforms=transforms, slicing_ops=slicing_ops
    )
    return _array


def build_numpy_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]:
    """Get a numpy array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_from_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda: _numpy_get_pipe(
        zarr_array=zarr_array,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


def build_dask_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_from_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda: _dask_get_pipe(
        zarr_array=zarr_array,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


##############################################################
#
# Concrete "To Disk" Pipes
#
##############################################################


def _numpy_set_pipe(
    zarr_array: zarr.Array,
    patch: np.ndarray,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> None:
    _patch = apply_inverse_numpy_transforms(
        patch, transforms=transforms, slicing_ops=slicing_ops
    )
    _patch = apply_numpy_axes_ops(
        _patch,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )

    if not np.can_cast(_patch.dtype, zarr_array.dtype, casting="safe"):
        raise ValueError(
            f"Cannot safely cast patch of dtype {_patch.dtype} to "
            f"zarr array of dtype {zarr_array.dtype}."
        )
    set_numpy_patch(zarr_array, _patch, slicing_ops.slice_tuple)


def _dask_set_pipe(
    zarr_array: zarr.Array,
    patch: DaskArray,
    slicing_ops: SlicingOps,
    transforms: Sequence[TransformProtocol] | None,
) -> None:
    _patch = apply_inverse_dask_transforms(
        patch, transforms=transforms, slicing_ops=slicing_ops
    )
    _patch = apply_dask_axes_ops(
        _patch,
        squeeze_axes=slicing_ops.squeeze_axes,
        transpose_axes=slicing_ops.transpose_axes,
        expand_axes=slicing_ops.expand_axes,
    )
    if not np.can_cast(_patch.dtype, zarr_array.dtype, casting="safe"):
        raise ValueError(
            f"Cannot safely cast patch of dtype {_patch.dtype} to "
            f"zarr array of dtype {zarr_array.dtype}."
        )
    set_dask_patch(zarr_array, _patch, slicing_ops.slice_tuple)


def build_numpy_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]:
    """Set a numpy array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_to_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda patch: _numpy_set_pipe(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


def build_dask_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    remove_channel_selection: bool = False,
) -> Callable[[DaskArray], None]:
    """Set a dask array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    slicing_ops = setup_to_disk_pipe(
        dimensions=dimensions,
        axes_order=axes_order,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )
    return lambda patch: _dask_set_pipe(
        zarr_array=zarr_array,
        patch=patch,
        slicing_ops=slicing_ops,
        transforms=transforms,
    )


################################################################
#
# Masked Array Pipes
#
################################################################


def _match_data_shape(mask: np.ndarray, data_shape: tuple[int, ...]) -> np.ndarray:
    """Scale the mask data to match the shape of the data."""
    if mask.ndim < len(data_shape):
        mask = np.reshape(mask, (1,) * (len(data_shape) - mask.ndim) + mask.shape)
    elif mask.ndim > len(data_shape):
        raise ValueError(
            "The mask has more dimensions than the data and cannot be matched."
        )

    zoom_factors = []
    for s_d, s_m in zip(data_shape, mask.shape, strict=True):
        if s_m == s_d:
            zoom_factors.append(1.0)
        elif s_m == 1:
            zoom_factors.append(s_d)  # expand singleton
        else:
            zoom_factors.append(s_d / s_m)

    mask_matched: np.ndarray = numpy_zoom(mask, scale=tuple(zoom_factors), order=0)
    return mask_matched


def _label_to_bool_mask_numpy(
    label_data: np.ndarray | DaskArray,
    label: int | None = None,
    data_shape: tuple[int, ...] | None = None,
    allow_scaling: bool = True,
) -> np.ndarray:
    """Convert label data to a boolean mask."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    if data_shape is not None and label_data.shape != data_shape:
        if allow_scaling:
            bool_mask = _match_data_shape(bool_mask, data_shape)
        else:
            bool_mask = np.broadcast_to(bool_mask, data_shape)
    return bool_mask


def build_masked_numpy_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    fill_value: int | float = 0,
    allow_scaling: bool = True,
) -> Callable[[], np.ndarray]:
    """Get a numpy array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    data_getter = build_numpy_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    label_data_getter = build_numpy_getter(
        zarr_array=label_zarr_array,
        dimensions=label_dimensions,
        axes_order=axes_order,
        transforms=label_transforms,
        slicing_dict=label_slicing_dict,
        remove_channel_selection=True,
    )

    def get_masked_data_as_numpy() -> np.ndarray:
        data = data_getter()
        label_data = label_data_getter()
        bool_mask = _label_to_bool_mask_numpy(
            label_data=label_data,
            label=label_id,
            data_shape=data.shape,
            allow_scaling=allow_scaling,
        )
        masked_data = np.where(bool_mask, data, fill_value)
        return masked_data

    return get_masked_data_as_numpy


def _match_data_shape_dask(mask: da.Array, data_shape: tuple[int, ...]) -> da.Array:
    """Scale the mask data to match the shape of the data."""
    if mask.ndim < len(data_shape):
        mask = da.reshape(mask, (1,) * (len(data_shape) - mask.ndim) + mask.shape)
    elif mask.ndim > len(data_shape):
        raise ValueError(
            "The mask has more dimensions than the data and cannot be matched."
        )

    zoom_factors = []
    for s_d, s_m in zip(data_shape, mask.shape, strict=True):
        if s_m == s_d:
            zoom_factors.append(1.0)
        elif s_m == 1:
            zoom_factors.append(s_d)  # expand singleton
        else:
            zoom_factors.append(s_d / s_m)

    mask_matched: da.Array = dask_zoom(mask, scale=tuple(zoom_factors), order=0)
    return mask_matched


def _label_to_bool_mask_dask(
    label_data: DaskArray,
    label: int | None = None,
    data_shape: tuple[int, ...] | None = None,
    allow_scaling: bool = True,
) -> DaskArray:
    """Convert label data to a boolean mask for Dask arrays."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    if data_shape is not None and label_data.shape != data_shape:
        if allow_scaling:
            bool_mask = _match_data_shape_dask(bool_mask, data_shape)
        else:
            bool_mask = da.broadcast_to(bool_mask, data_shape)
    return bool_mask


def build_masked_dask_getter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    fill_value: int | float = 0,
    allow_scaling: bool = True,
) -> Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    data_getter = build_dask_getter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    label_data_getter = build_dask_getter(
        zarr_array=label_zarr_array,
        dimensions=label_dimensions,
        axes_order=axes_order,
        transforms=label_transforms,
        slicing_dict=label_slicing_dict,
        remove_channel_selection=True,
    )

    def get_masked_data_as_dask() -> DaskArray:
        data = data_getter()
        label_data = label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)
        bool_mask = _label_to_bool_mask_dask(
            label_data=label_data,
            label=label_id,
            data_shape=data_shape,
            allow_scaling=allow_scaling,
        )
        masked_data = da.where(bool_mask, data, fill_value)
        return masked_data

    return get_masked_data_as_dask


def build_masked_numpy_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    data_getter: Callable[[], np.ndarray] | None = None,
    label_data_getter: Callable[[], np.ndarray] | None = None,
    allow_scaling: bool = True,
) -> Callable[[np.ndarray], None]:
    """Set a numpy array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_numpy_getter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=False,
        )

    if label_data_getter is None:
        label_data_getter = build_numpy_getter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    masked_data_setter = build_numpy_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    def set_patch_masked_as_numpy(patch: np.ndarray) -> None:
        """Set a numpy patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        bool_mask = _label_to_bool_mask_numpy(
            label_data=label_data,
            label=label_id,
            data_shape=data.shape,
            allow_scaling=allow_scaling,
        )
        mask_data = np.where(bool_mask, patch, data)
        masked_data_setter(mask_data)

    return set_patch_masked_as_numpy


def build_masked_dask_setter(
    *,
    zarr_array: zarr.Array,
    dimensions: Dimensions,
    label_zarr_array: zarr.Array,
    label_dimensions: Dimensions,
    label_id: int | None,
    axes_order: Sequence[str] | None = None,
    transforms: Sequence[TransformProtocol] | None = None,
    label_transforms: Sequence[TransformProtocol] | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
    label_slicing_dict: dict[str, SlicingInputType] | None = None,
    data_getter: Callable[[], DaskArray] | None = None,
    label_data_getter: Callable[[], DaskArray] | None = None,
    allow_scaling: bool = True,
) -> Callable[[DaskArray], None]:
    """Set a dask array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_dask_getter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=False,
        )

    if label_data_getter is None:
        label_data_getter = build_dask_getter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    data_setter = build_dask_setter(
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=False,
    )

    def set_patch_masked_as_dask(patch: DaskArray) -> None:
        """Set a dask patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)
        bool_mask = _label_to_bool_mask_dask(
            label_data=label_data,
            label=label_id,
            data_shape=data_shape,
            allow_scaling=allow_scaling,
        )
        mask_data = da.where(bool_mask, patch, data)
        data_setter(mask_data)

    return set_patch_masked_as_dask
