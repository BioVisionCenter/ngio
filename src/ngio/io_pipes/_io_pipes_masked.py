from collections.abc import Sequence

import zarr

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    NumpyGetter,
    NumpySetter,
    _FromDimensionsInit,
)
from ngio.io_pipes._mask_transform import BaseMaskTransform
from ngio.io_pipes._ops_slices import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol

__all__ = [
    "DaskGetterMasked",
    "DaskSetterMasked",
    "NumpyGetterMasked",
    "NumpySetterMasked",
]


class _MaskedInit(_FromDimensionsInit):
    """The bare pipes' constructor plus a terminal mask transform."""

    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        fill_value: int | float = 0,
        allow_rescaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        mask_transform = BaseMaskTransform(
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            label_transforms=label_transforms,
            label_slicing_dict=label_slicing_dict,
            axes_order=axes_order,
            fill_value=fill_value,
            allow_rescaling=allow_rescaling,
            target_dimensions=dimensions,
            set_transforms=transforms,
        )
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=[*(transforms or []), mask_transform],
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )

    @property
    def label_id(self) -> int | None:
        return self.roi.label


class _MaskedSetterInit(_MaskedInit):
    """The masked constructor without the get-only `fill_value`."""

    def __init__(
        self,
        *,
        zarr_array: zarr.Array,
        dimensions: Dimensions,
        label_zarr_array: zarr.Array,
        label_dimensions: Dimensions,
        roi: Roi,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
        slicing_dict: dict[str, SlicingInputType] | None = None,
        label_slicing_dict: dict[str, SlicingInputType] | None = None,
        allow_rescaling: bool = True,
        remove_channel_selection: bool = False,
    ):
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            label_zarr_array=label_zarr_array,
            label_dimensions=label_dimensions,
            roi=roi,
            axes_order=axes_order,
            transforms=transforms,
            label_transforms=label_transforms,
            slicing_dict=slicing_dict,
            label_slicing_dict=label_slicing_dict,
            allow_rescaling=allow_rescaling,
            remove_channel_selection=remove_channel_selection,
        )


class NumpyGetterMasked(_MaskedInit, NumpyGetter):
    pass


class DaskGetterMasked(_MaskedInit, DaskGetter):
    pass


class NumpySetterMasked(_MaskedSetterInit, NumpySetter):
    pass


class DaskSetterMasked(_MaskedSetterInit, DaskSetter):
    pass
