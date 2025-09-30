from collections.abc import Sequence

import dask.array as da
import numpy as np
import zarr

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi, RoiPixels
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    DataGetter,
    NumpyGetter,
    NumpySetter,
)
from ngio.io_pipes._io_pipes_masked import (
    DaskMaskedGetter,
    DaskMaskedSetter,
    NumpyMaskedGetter,
    NumpyMaskedSetter,
)
from ngio.io_pipes._io_pipes_utils import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize
from ngio.utils import NgioValueError


def roi_to_slicing_dict(
    roi: Roi | RoiPixels,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> dict[str, SlicingInputType]:
    """Convert a ROI to a slicing dictionary."""
    if isinstance(roi, Roi):
        if pixel_size is None:
            raise NgioValueError(
                "pixel_size must be provided when converting a Roi to slice_kwargs."
            )
        roi = roi.to_roi_pixels(pixel_size=pixel_size, dimensions=dimensions)

    roi_slicing_dict: dict[str, SlicingInputType] = roi.to_slicing_dict()  # type: ignore
    if slicing_dict is None:
        return roi_slicing_dict

    # Additional slice kwargs can be provided
    # and will override the ones from the ROI
    roi_slicing_dict.update(slicing_dict)
    return roi_slicing_dict


class NumpyRoiGetter(NumpyGetter):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
        )


class DaskRoiGetter(DaskGetter):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
        )


class NumpyRoiSetter(NumpySetter):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
        )


class DaskRoiSetter(DaskSetter):
    def __init__(
        self,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi | RoiPixels,
        pixel_size: PixelSize | None = None,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
        )


################################################################
#
# Masked ROIs array pipes
#
################################################################


class NumpyMaskedRoiGetter(NumpyMaskedGetter):
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
        fill_value: int | float = 0,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for getting a masked array."""
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=label_dimensions,
            pixel_size=label_pixel_size,
            slicing_dict=label_slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=roi.label,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=input_slice_kwargs,
            label_slicing_dict=label_slice_kwargs,
            fill_value=fill_value,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )


class DaskMaskedRoiGetter(DaskMaskedGetter):
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
        fill_value: int | float = 0,
        allow_scaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        """Prepare slice kwargs for getting a masked array."""
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=label_dimensions,
            pixel_size=label_pixel_size,
            slicing_dict=label_slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=roi.label,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=input_slice_kwargs,
            label_slicing_dict=label_slice_kwargs,
            fill_value=fill_value,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )


class NumpyMaskedRoiSetter(NumpyMaskedSetter):
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
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=label_dimensions,
            pixel_size=label_pixel_size,
            slicing_dict=label_slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=roi.label,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=input_slice_kwargs,
            label_slicing_dict=label_slice_kwargs,
            data_getter=data_getter,
            label_data_getter=label_data_getter,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )


class DaskMaskedRoiSetter(DaskMaskedSetter):
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
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=dimensions,
            pixel_size=pixel_size,
            slicing_dict=slicing_dict,
        )
        label_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            dimensions=label_dimensions,
            pixel_size=label_pixel_size,
            slicing_dict=label_slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_id=roi.label,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=input_slice_kwargs,
            label_slicing_dict=label_slice_kwargs,
            data_getter=data_getter,
            label_data_getter=label_data_getter,
            allow_scaling=allow_scaling,
            remove_channel_selection=remove_channel_selection,
        )
