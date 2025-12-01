from collections.abc import Sequence
from typing import Protocol, TypeVar

import zarr

from ngio.common._roi import Roi
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.io_pipes._ops_transforms import TransformProtocol

GetterDataType = TypeVar("GetterDataType", covariant=True)
SetterDataType = TypeVar("SetterDataType", contravariant=True)


class DataGetterProtocol(Protocol[GetterDataType]):
    @property
    def zarr_array(self) -> zarr.Array: ...

    @property
    def slicing_ops(self) -> SlicingOps: ...

    @property
    def axes_ops(self) -> AxesOps: ...

    @property
    def transforms(self) -> Sequence[TransformProtocol] | None: ...

    @property
    def roi(self) -> Roi: ...

    def __call__(self) -> GetterDataType:
        return self.get()

    def get(self) -> GetterDataType: ...


class DataSetterProtocol(Protocol[SetterDataType]):
    @property
    def zarr_array(self) -> zarr.Array: ...

    @property
    def slicing_ops(self) -> SlicingOps: ...

    @property
    def axes_ops(self) -> AxesOps: ...

    @property
    def transforms(self) -> Sequence[TransformProtocol] | None: ...

    @property
    def roi(self) -> Roi: ...

    def __call__(self, patch: SetterDataType) -> None:
        return self.set(patch)

    def set(self, patch: SetterDataType) -> None: ...
