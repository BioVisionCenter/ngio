"""I/O pipes for reading and writing data from zarr to numpy and dask arrays.

There are 3 main types of I/O pipes:
- Standard I/O pipes: NumpyGetter, NumpySetter, DaskGetter, DaskSetter:
    These pipes read and write data from simple integer indexing and slicing.
- ROI I/O pipes: NumpyRoiGetter, NumpyRoiSetter, DaskRoiGetter, DaskRoiSetter:
    These pipes read and write data from a region of interest (ROI) defined in physical
    coordinates.
- Masked I/O pipes: NumpyGetterMasked, NumpySetterMasked, DaskGetterMasked,
    DaskSetterMasked: These pipes like the ROI pipes read and write data
    from a region of interest (ROI). However they also load a boolean mask
    from a label zarr array to mask the data being read or written.

"""

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
from ngio.io_pipes._match_shape import dask_match_shape, numpy_match_shape
from ngio.io_pipes._ops_slices import SlicingInputType, SlicingOps, SlicingType
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
