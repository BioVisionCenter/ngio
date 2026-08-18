"""The pre-1.1 ROI and masked pipe classes, kept as thin shells until 1.2.

The ROI pipes collapsed into the bare pipes (whose `roi` argument now
slices); the masked pipes collapsed into a bare pipe with a terminal
`MaskTransform`. Every class here delegates to that replacement and emits an
`NgioDeprecationWarning`. This whole module is removed in ngio=1.2.
"""

import warnings
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
from ngio.utils import NgioDeprecationWarning

__all__ = [
    "DaskGetterMasked",
    "DaskRoiGetter",
    "DaskRoiSetter",
    "DaskSetterMasked",
    "NumpyGetterMasked",
    "NumpyRoiGetter",
    "NumpyRoiSetter",
    "NumpySetterMasked",
]


def _warn_deprecated_pipe(old: str, replacement: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in ngio=1.2. "
        f"Use {replacement} instead.",
        NgioDeprecationWarning,
        stacklevel=3,
    )


class _DeprecatedRoiInit(_FromDimensionsInit):
    """The bare pipes' constructor with `roi` required, plus a warning."""

    _replacement: str

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
        _warn_deprecated_pipe(type(self).__name__, self._replacement)
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )


class NumpyRoiGetter(_DeprecatedRoiInit, NumpyGetter):
    _replacement = "NumpyGetter(..., roi=...)"


class DaskRoiGetter(_DeprecatedRoiInit, DaskGetter):
    _replacement = "DaskGetter(..., roi=...)"


class NumpyRoiSetter(_DeprecatedRoiInit, NumpySetter):
    _replacement = "NumpySetter(..., roi=...)"


class DaskRoiSetter(_DeprecatedRoiInit, DaskSetter):
    _replacement = "DaskSetter(..., roi=...)"


class _DeprecatedMaskedInit(_FromDimensionsInit):
    """The bare pipes' constructor plus a terminal mask transform."""

    _replacement: str

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
        _warn_deprecated_pipe(type(self).__name__, self._replacement)
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


class _DeprecatedMaskedSetterInit(_DeprecatedMaskedInit):
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


class NumpyGetterMasked(_DeprecatedMaskedInit, NumpyGetter):
    _replacement = (
        "NumpyGetter(..., roi=..., transforms=[..., ngio.transforms.MaskTransform])"
    )


class DaskGetterMasked(_DeprecatedMaskedInit, DaskGetter):
    _replacement = (
        "DaskGetter(..., roi=..., transforms=[..., ngio.transforms.MaskTransform])"
    )


class NumpySetterMasked(_DeprecatedMaskedSetterInit, NumpySetter):
    _replacement = (
        "NumpySetter(..., roi=..., transforms=[..., ngio.transforms.MaskTransform])"
    )


class DaskSetterMasked(_DeprecatedMaskedSetterInit, DaskSetter):
    _replacement = (
        "DaskSetter(..., roi=..., transforms=[..., ngio.transforms.MaskTransform])"
    )
