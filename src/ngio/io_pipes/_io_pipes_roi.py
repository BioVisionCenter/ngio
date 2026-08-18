from collections.abc import Sequence

import zarr

from ngio.common._dimensions import Dimensions
from ngio.common._roi import Roi
from ngio.io_pipes._io_pipe_ops import roi_to_slicing_dict
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    NumpyGetter,
    NumpySetter,
    _FromDimensionsInit,
)
from ngio.io_pipes._ops_slices import SlicingInputType
from ngio.io_pipes._ops_transforms import TransformProtocol

__all__ = [
    "DaskRoiGetter",
    "DaskRoiSetter",
    "NumpyRoiGetter",
    "NumpyRoiSetter",
    "roi_to_slicing_dict",
]


class _RoiRequiredInit(_FromDimensionsInit):
    """The bare pipes' constructor with `roi` required instead of optional."""

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
        super().__init__(
            zarr_array=zarr_array,
            dimensions=dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_dict,
            remove_channel_selection=remove_channel_selection,
            roi=roi,
        )


class NumpyRoiGetter(_RoiRequiredInit, NumpyGetter):
    pass


class DaskRoiGetter(_RoiRequiredInit, DaskGetter):
    pass


class NumpyRoiSetter(_RoiRequiredInit, NumpySetter):
    pass


class DaskRoiSetter(_RoiRequiredInit, DaskSetter):
    pass
