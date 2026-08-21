"""I/O pipes for reading and writing data from zarr to numpy and dask arrays.

There are four pipes: `NumpyGetter`, `NumpySetter`, `DaskGetter`,
`DaskSetter` — (numpy, dask) x (read, write). Everything else is expressed
through their arguments:

- Slicing comes from integer indexing and slices (`slicing_dict`), from a
    region of interest in physical coordinates (`roi`), or both — explicit
    `slicing_dict` entries override the ROI-derived ones per axis.
- Behaviours on the data path are transforms (`transforms`), e.g.
    `ngio.transforms.ZoomTransform` to rescale between pyramid levels and
    `ngio.transforms.MaskTransform` to mask by a label image.

When reading, the order of operations is:

- Step 1: Slice the zarr array to load only the data needed into memory.
- Step 2: Apply axes operations — reorder, squeeze or expand the axes to
    match the caller's requested axes order.
- Step 3: Apply the transforms' `on_get` in order.

When writing the order is inverted: the transforms' `on_set` run in reverse
order, then the axes operations are undone, then the slice is written.

The Transforms must implement the `TransformProtocol`: `on_get(array, ctx)`
runs after reading, `on_set(array, ctx)` (the inverse) before writing. One
contract serves both backends — the array type expresses the backend, and a
transform supporting both dispatches on it. The `ctx` is the pipe's
`IoPipeContext`; transforms may depend on its stable fields — `axes`,
`slicing`, and `roi` (when there is one) — while `zarr_array` and `axes_ops`
are pipe plumbing.

A transform is a function of the patch alone; combining with what is already
on disk is a merge and gets its own slot (`merge=` on a setter) — see
`ngio.transforms` for the split.

The ROI pipe classes (`*RoiGetter`, `*RoiSetter`) and the masked pipe
classes (`*GetterMasked`, `*SetterMasked`) are deprecated shells over the
constructs above; they are removed in ngio=1.2.
"""

from ngio.io_pipes._deprecated_pipes import (
    DaskGetterMasked,
    DaskRoiGetter,
    DaskRoiSetter,
    DaskSetterMasked,
    NumpyGetterMasked,
    NumpyRoiGetter,
    NumpyRoiSetter,
    NumpySetterMasked,
)
from ngio.io_pipes._io_pipes import (
    DaskGetter,
    DaskSetter,
    DataGetter,
    DataSetter,
    NumpyGetter,
    NumpySetter,
)
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._match_shape import dask_match_shape, numpy_match_shape
from ngio.io_pipes._merge_policy import MergeInput, MergePolicy, MergeRule
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingInputType, SlicingOps, SlicingType
from ngio.io_pipes._ops_slices_utils import ChunkRect
from ngio.io_pipes._ops_transforms import (
    IoPipeContext,
    TransformContext,
    TransformProtocol,
)

__all__ = [
    "AxesOps",
    "ChunkRect",
    "DaskGetter",
    "DaskGetterMasked",
    "DaskRoiGetter",
    "DaskRoiSetter",
    "DaskSetter",
    "DaskSetterMasked",
    "DataGetter",
    "DataGetterProtocol",
    "DataSetter",
    "DataSetterProtocol",
    "IoPipeContext",
    "MergeInput",
    "MergePolicy",
    "MergeRule",
    "NumpyGetter",
    "NumpyGetterMasked",
    "NumpyRoiGetter",
    "NumpyRoiSetter",
    "NumpySetter",
    "NumpySetterMasked",
    "SlicingInputType",
    "SlicingOps",
    "SlicingType",
    "TransformContext",
    "TransformProtocol",
    "dask_match_shape",
    "numpy_match_shape",
]
