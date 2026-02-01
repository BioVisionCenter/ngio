from collections.abc import Sequence

import zarr

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    NumpyGetter,
    NumpySetter,
)
from ngio.io_pipes._ops_slices import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize


def roi_to_slicing_dict(
    *,
    roi: Roi,
    pixel_size: PixelSize,
    slicing_dict: dict[str, SlicingInputType] | None = None,
) -> dict[str, SlicingInputType]:
    """Convert a ROI to a slicing dictionary."""
    roi_slicing_dict: dict[str, SlicingInputType] = roi.to_slicing_dict(
        pixel_size=pixel_size
    )  # type: ignore
    if slicing_dict is None:
        return roi_slicing_dict

    # Additional slice kwargs can be provided
    # and will override the ones from the ROI
    roi_slicing_dict.update(slicing_dict)
    return roi_slicing_dict


class NumpyRoiGetter(NumpyGetter):
    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )


class DaskRoiGetter(DaskGetter):
    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )


class NumpyRoiSetter(NumpySetter):
    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )


class DaskRoiSetter(DaskSetter):
    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        remove_channel_selection: bool = False,
    ) -> None:
        input_slice_kwargs = roi_to_slicing_dict(
            roi=roi,
            pixel_size=dimensions.pixel_size,
            slicing_dict=slicing_dict,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=input_slice_kwargs,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )
