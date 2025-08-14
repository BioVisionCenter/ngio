"""Common classes and functions that are used across the package."""

from ngio.common._array_io_pipes import (
    build_dask_getter,
    build_dask_setter,
    build_masked_dask_getter,
    build_masked_dask_setter,
    build_masked_numpy_getter,
    build_masked_numpy_setter,
    build_numpy_getter,
    build_numpy_setter,
)
from ngio.common._array_io_utils import (
    ArrayLike,
    SlicingInputType,
    TransformProtocol,
    apply_dask_axes_ops,
    apply_numpy_axes_ops,
    apply_sequence_axes_ops,
)
from ngio.common._dimensions import Dimensions
from ngio.common._masking_roi import compute_masking_roi
from ngio.common._pyramid import consolidate_pyramid, init_empty_pyramid, on_disk_zoom
from ngio.common._roi import (
    Roi,
    RoiPixels,
    build_roi_dask_getter,
    build_roi_dask_setter,
    build_roi_masked_dask_getter,
    build_roi_masked_dask_setter,
    build_roi_masked_numpy_getter,
    build_roi_masked_numpy_setter,
    build_roi_numpy_getter,
    build_roi_numpy_setter,
    roi_to_slicing_dict,
)
from ngio.common._zoom import dask_zoom, numpy_zoom

__all__ = [
    "ArrayLike",
    "Dimensions",
    "Roi",
    "RoiPixels",
    "SlicingInputType",
    "TransformProtocol",
    "apply_dask_axes_ops",
    "apply_numpy_axes_ops",
    "apply_sequence_axes_ops",
    "build_dask_getter",
    "build_dask_setter",
    "build_masked_dask_getter",
    "build_masked_dask_setter",
    "build_masked_numpy_getter",
    "build_masked_numpy_setter",
    "build_numpy_getter",
    "build_numpy_setter",
    "build_roi_dask_getter",
    "build_roi_dask_setter",
    "build_roi_masked_dask_getter",
    "build_roi_masked_dask_setter",
    "build_roi_masked_numpy_getter",
    "build_roi_masked_numpy_setter",
    "build_roi_numpy_getter",
    "build_roi_numpy_setter",
    "compute_masking_roi",
    "consolidate_pyramid",
    "dask_zoom",
    "init_empty_pyramid",
    "numpy_zoom",
    "on_disk_zoom",
    "roi_to_slicing_dict",
]
