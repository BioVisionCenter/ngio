"""I/O pipes for reading and writing data from zarr to numpy and dask arrays."""

from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    DataGetter,
    DataSetter,
    NumpyGetter,
    NumpySetter,
)
from ngio.io_pipes._io_pipes_masked import (
    DaskGetterMasked,
    DaskSetterMasked,
    NumpyGetterMasked,
    NumpySetterMasked,
)
from ngio.io_pipes._io_pipes_roi import (
    DaskRoiGetter,
    DaskRoiSetter,
    NumpyRoiGetter,
    NumpyRoiSetter,
)
from ngio.io_pipes._io_pipes_utils import SlicingInputType
from ngio.io_pipes._match_shape import dask_match_shape, numpy_match_shape
from ngio.io_pipes._ops_slices import SlicingOps, SlicingType
from ngio.io_pipes._ops_transforms import TransformProtocol

__all__ = [
    "DaskGetter",
    "DaskGetterMasked",
    "DaskRoiGetter",
    "DaskRoiSetter",
    "DaskSetter",
    "DaskSetterMasked",
    "DataGetter",
    "DataSetter",
    "NumpyGetter",
    "NumpyGetterMasked",
    "NumpyRoiGetter",
    "NumpyRoiSetter",
    "NumpySetter",
    "NumpySetterMasked",
    "SlicingInputType",
    "SlicingOps",
    "SlicingType",
    "TransformProtocol",
    "dask_match_shape",
    "numpy_match_shape",
]
