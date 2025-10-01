from collections.abc import Sequence

import dask.array as da
import numpy as np
import zarr
from dask.array import Array as DaskArray

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi, RoiPixels
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    DataGetter,
    DataSetter,
    NumpyGetter,
    NumpySetter,
)
from ngio.io_pipes._io_pipes_roi import roi_to_slicing_dict
from ngio.io_pipes._io_pipes_utils import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol
from ngio.io_pipes._utils import dask_match_shape, numpy_match_shape
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize


def _numpy_label_to_bool_mask(
    label_data: np.ndarray,
    label: int | None,
    data_shape: tuple[int, ...],
    label_axes: tuple[str, ...],
    data_axes: tuple[str, ...],
    allow_resize: bool = True,
) -> np.ndarray:
    """Convert label data to a boolean mask."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    bool_mask = numpy_match_shape(
        array=bool_mask,
        reference_shape=data_shape,
        array_axes=label_axes,
        reference_axes=data_axes,
        allow_resize=allow_resize,
    )
    return bool_mask


def _dask_label_to_bool_mask(
    label_data: DaskArray,
    label: int | None,
    data_shape: tuple[int, ...],
    label_axes: tuple[str, ...],
    data_axes: tuple[str, ...],
    allow_resize: bool = True,
) -> DaskArray:
    """Convert label data to a boolean mask."""
    if label is not None:
        bool_mask = label_data == label
    else:
        bool_mask = label_data != 0

    bool_mask = dask_match_shape(
        array=bool_mask,
        reference_shape=data_shape,
        array_axes=label_axes,
        reference_axes=data_axes,
        allow_resize=allow_resize,
    )
    return bool_mask


class NumpyMaskedRoiGetter(DataGetter[np.ndarray]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        fill_value: int | float = 0,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for getting a masked array."""
        slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=label_dimensions.pixel_size,
            slicing_dict=label_slicing_dict,
        )
        self._data_getter = NumpyGetter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

        self._label_data_getter = NumpyGetter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )

        self._label_id = roi.label
        self._fill_value = fill_value
        self._allow_scaling = allow_scaling
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=self._data_getter.slicing_ops,
            axes_ops=self._data_getter.axes_ops,
            transforms=self._data_getter.transforms,
        )

    @property
    def label_id(self) -> int | None:
        return self._label_id

    def get(self) -> np.ndarray:
        """Get the masked data as a numpy array."""
        data = self._data_getter()
        label_data = self._label_data_getter()
        bool_mask = _numpy_label_to_bool_mask(
            label_data=label_data,
            label=self.label_id,
            data_shape=data.shape,
            label_axes=self._label_data_getter.axes_ops.in_memory_axes,
            data_axes=self._data_getter.axes_ops.in_memory_axes,
            allow_resize=self._allow_scaling,
        )
        masked_data = np.where(bool_mask, data, self._fill_value)
        return masked_data


class DaskMaskedRoiGetter(DataGetter[DaskArray]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        fill_value: int | float = 0,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for getting a masked array."""
        slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=label_dimensions.pixel_size,
            slicing_dict=label_slicing_dict,
        )
        self._data_getter = DaskGetter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

        self._label_data_getter = DaskGetter(
            zarr_array=label_zarr_array,
            dimensions=label_dimensions,
            axes_order=axes_order,
            transforms=label_transforms,
            slicing_dict=label_slicing_dict,
            remove_channel_selection=True,
        )
        self._label_id = roi.label
        self._fill_value = fill_value
        self._allow_scaling = allow_scaling
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=self._data_getter.slicing_ops,
            axes_ops=self._data_getter.axes_ops,
            transforms=self._data_getter.transforms,
        )

    @property
    def label_id(self) -> int | None:
        return self._label_id

    def get(self) -> DaskArray:
        data = self._data_getter()
        label_data = self._label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)
        bool_mask = _dask_label_to_bool_mask(
            label_data=label_data,
            label=self.label_id,
            data_shape=data_shape,
            label_axes=self._label_data_getter.axes_ops.in_memory_axes,
            data_axes=self._data_getter.axes_ops.in_memory_axes,
            allow_resize=self._allow_scaling,
        )
        masked_data = da.where(bool_mask, data, self._fill_value)
        return masked_data


class NumpyMaskedRoiSetter(DataSetter[np.ndarray]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        label_pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        data_getter: DataGetter[np.ndarray] | None = None,
        label_data_getter: DataGetter[np.ndarray] | None = None,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for setting a masked array."""
        slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=label_dimensions.pixel_size,
            slicing_dict=label_slicing_dict,
        )

        if data_getter is None:
            self._data_getter = NumpyGetter(
                zarr_array=zarr_array,
                dimensions=dimensions,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_dict,
                remove_channel_selection=remove_channel_selection,
            )
        else:
            self._data_getter = data_getter

        if label_data_getter is None:
            self._label_data_getter = NumpyGetter(
                zarr_array=label_zarr_array,
                dimensions=label_dimensions,
                axes_order=axes_order,
                transforms=label_transforms,
                slicing_dict=label_slicing_dict,
                remove_channel_selection=True,
            )
        else:
            self._label_data_getter = label_data_getter

        self._data_setter = NumpySetter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

        self._label_id = roi.label
        self._allow_scaling = allow_scaling
        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=self._data_setter.slicing_ops,
            axes_ops=self._data_setter.axes_ops,
            transforms=self._data_setter.transforms,
        )

    @property
    def label_id(self) -> int | None:
        return self._label_id

    def set(self, patch: np.ndarray) -> None:
        data = self._data_getter()
        label_data = self._label_data_getter()

        bool_mask = _numpy_label_to_bool_mask(
            label_data=label_data,
            label=self.label_id,
            data_shape=data.shape,
            label_axes=self._label_data_getter.axes_ops.in_memory_axes,
            data_axes=self._data_getter.axes_ops.in_memory_axes,
            allow_resize=self._allow_scaling,
        )
        mask_data = np.where(bool_mask, patch, data)
        self._data_setter(mask_data)


class DaskMaskedRoiSetter(DataSetter[DaskArray]):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        label_pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        data_getter: DataGetter[da.Array] | None = None,
        label_data_getter: DataGetter[da.Array] | None = None,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for setting a masked array."""
        slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slicing_dict = roi_to_slicing_dict(
            roi=roi,
            pixel_size=label_dimensions.pixel_size,
            slicing_dict=label_slicing_dict,
        )

        if data_getter is None:
            self._data_getter = DaskGetter(
                zarr_array=zarr_array,
                dimensions=dimensions,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_dict,
                remove_channel_selection=remove_channel_selection,
            )
        else:
            self._data_getter = data_getter

        if label_data_getter is None:
            self._label_data_getter = DaskGetter(
                zarr_array=label_zarr_array,
                dimensions=label_dimensions,
                axes_order=axes_order,
                transforms=label_transforms,
                slicing_dict=label_slicing_dict,
                remove_channel_selection=True,
            )
        else:
            self._label_data_getter = label_data_getter
        self._label_id = roi.label
        self._allow_scaling = allow_scaling

        self._data_setter = DaskSetter(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
        )

        super().__init__(
            zarr_array=zarr_array,
            slicing_ops=self._data_setter.slicing_ops,
            axes_ops=self._data_setter.axes_ops,
            transforms=self._data_setter.transforms,
        )

    @property
    def label_id(self) -> int | None:
        return self._label_id

    def set(self, patch: DaskArray) -> None:
        data = self._data_getter()
        label_data = self._label_data_getter()
        data_shape = tuple(int(dim) for dim in data.shape)

        bool_mask = _dask_label_to_bool_mask(
            label_data=label_data,
            label=self.label_id,
            data_shape=data_shape,
            label_axes=self._label_data_getter.axes_ops.in_memory_axes,
            data_axes=self._data_getter.axes_ops.in_memory_axes,
            allow_resize=self._allow_scaling,
        )
        mask_data = da.where(bool_mask, patch, data)
        self._data_setter(mask_data)
