from collections.abc import Callable, Sequence
from typing import Literal, assert_never, overload

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._zoom import dask_zoom, numpy_zoom
from ngio.io_pipes._io_pipes import build_getter_pipe, build_setter_pipe
from ngio.io_pipes._io_pipes_utils import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol


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

    mask_matched: np.ndarray = numpy_zoom(
        mask, scale=tuple(zoom_factors), order="nearest"
    )
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

    mask_matched: da.Array = dask_zoom(mask, scale=tuple(zoom_factors), order="nearest")
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


def _build_numpy_getter_pipe(
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
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict
    data_getter = build_getter_pipe(
        mode="numpy",
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    label_data_getter = build_getter_pipe(
        mode="numpy",
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


def _build_dask_getter_pipe(
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
    remove_channel_selection: bool = False,
) -> Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict
    data_getter = build_getter_pipe(
        mode="dask",
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    label_data_getter = build_getter_pipe(
        mode="dask",
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


@overload
def build_masked_getter_pipe(
    *,
    mode: Literal["numpy"],
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
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray]: ...


@overload
def build_masked_getter_pipe(
    *,
    mode: Literal["dask"],
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
    remove_channel_selection: bool = False,
) -> Callable[[], DaskArray]: ...


def build_masked_getter_pipe(
    *,
    mode: Literal["numpy", "dask"],
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
    remove_channel_selection: bool = False,
) -> Callable[[], np.ndarray] | Callable[[], DaskArray]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict
    if mode == "numpy":
        return _build_numpy_getter_pipe(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=label_id,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=slicing_dict,
            label_slicing_dict=label_slicing_dict,
            fill_value=fill_value,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )
    elif mode == "dask":
        return _build_dask_getter_pipe(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=label_id,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=slicing_dict,
            label_slicing_dict=label_slicing_dict,
            fill_value=fill_value,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )

    assert_never(mode)


def _build_numpy_setter_pipe(
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
    data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    label_data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    allow_scaling: bool = True,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]:
    """Set a numpy array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_getter_pipe(
            mode="numpy",
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

    if label_data_getter is None:
        label_data_getter = build_getter_pipe(
            mode="numpy",
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    masked_data_setter = build_setter_pipe(
        mode="numpy",
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    def set_patch_masked_as_numpy(patch: np.ndarray) -> None:
        """Set a numpy patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        if not isinstance(label_data, np.ndarray):
            raise ValueError(
                "Something went wrong in setting up the label data getter. "
                f"Expected a numpy array, got type {type(label_data)}."
            )
        if not isinstance(data, np.ndarray):
            raise ValueError(
                "Something went wrong in setting up the data getter. "
                f"Expected a numpy array, got type {type(data)}."
            )
        bool_mask = _label_to_bool_mask_numpy(
            label_data=label_data,
            label=label_id,
            data_shape=data.shape,
            allow_scaling=allow_scaling,
        )
        mask_data = np.where(bool_mask, patch, data)
        masked_data_setter(mask_data)

    return set_patch_masked_as_numpy


def _build_dask_setter_pipe(
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
    data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    label_data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    allow_scaling: bool = True,
    remove_channel_selection: bool = False,
) -> Callable[[DaskArray], None]:
    """Set a dask array to the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict

    if data_getter is None:
        data_getter = build_getter_pipe(
            mode="dask",
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

    if label_data_getter is None:
        label_data_getter = build_getter_pipe(
            mode="dask",
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

    data_setter = build_setter_pipe(
        mode="dask",
        zarr_array=zarr_array,
        dimensions=dimensions,
        axes_order=axes_order,
        transforms=transforms,
        slicing_dict=slicing_dict,
        remove_channel_selection=remove_channel_selection,
    )

    def set_patch_masked_as_dask(patch: DaskArray) -> None:
        """Set a dask patch to the array, masked by the label array."""
        data = data_getter()
        label_data = label_data_getter()
        if not isinstance(label_data, da.Array):
            raise ValueError(
                "Something went wrong in setting up the label data getter. "
                f"Expected a Dask array, got type {type(label_data)}."
            )
        if not isinstance(data, da.Array):
            raise ValueError(
                "Something went wrong in setting up the data getter. "
                f"Expected a Dask array, got type {type(data)}."
            )
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


@overload
def build_masked_setter_pipe(
    *,
    mode: Literal["numpy"],
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
    data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    label_data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    allow_scaling: bool = True,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]: ...


@overload
def build_masked_setter_pipe(
    *,
    mode: Literal["dask"],
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
    data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    label_data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    allow_scaling: bool = True,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None]: ...


def build_masked_setter_pipe(
    *,
    mode: Literal["numpy", "dask"],
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
    data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    label_data_getter: Callable[[], np.ndarray | DaskArray] | None = None,
    allow_scaling: bool = True,
    remove_channel_selection: bool = False,
) -> Callable[[np.ndarray], None] | Callable[[DaskArray], None]:
    """Get a dask array from the zarr array with the given slice kwargs."""
    slicing_dict = slicing_dict or {}
    label_slicing_dict = label_slicing_dict or slicing_dict
    if mode == "numpy":
        return _build_numpy_setter_pipe(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=label_id,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=slicing_dict,
            label_slicing_dict=label_slicing_dict,
            data_getter=data_getter,
            label_data_getter=label_data_getter,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )
    elif mode == "dask":
        return _build_dask_setter_pipe(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=label_id,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=slicing_dict,
            label_slicing_dict=label_slicing_dict,
            data_getter=data_getter,
            label_data_getter=label_data_getter,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )

    assert_never(mode)
