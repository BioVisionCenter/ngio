"""ngio internal specs module.

Since the OME-Zarr specification are still evolving, this module provides a
set of classes to internally handle the metadata.

This models can be tr
"""

from ngio.ome_zarr_meta.ngio_specs._axes import (
    AxesHandler,
    AxesSetup,
    Axis,
    AxisType,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
    build_canonical_axes_handler,
    canonical_axes_order,
    canonical_label_axes_order,
)
from ngio.ome_zarr_meta.ngio_specs._channels import (
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    NgioColors,
    default_channel_name,
)
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._ngio_hcs import (
    ImageInWellPath,
    NgioPlateMeta,
    NgioWellMeta,
    path_in_well_validation,
)
from ngio.ome_zarr_meta.ngio_specs._ngio_image import (
    DefaultNgffVersion,
    ImageLabelSource,
    NgffVersions,
    NgioImageLabelMeta,
    NgioImageMeta,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
)
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize

__all__ = [
    "AxesHandler",
    "AxesSetup",
    "Axis",
    "AxisType",
    "Channel",
    "ChannelVisualisation",
    "ChannelsMeta",
    "Dataset",
    "DefaultNgffVersion",
    "DefaultSpaceUnit",
    "DefaultTimeUnit",
    "ImageInWellPath",
    "ImageLabelSource",
    "NgffVersions",
    "NgioColors",
    "NgioImageLabelMeta",
    "NgioImageMeta",
    "NgioLabelMeta",
    "NgioLabelsGroupMeta",
    "NgioPlateMeta",
    "NgioWellMeta",
    "PixelSize",
    "SpaceUnits",
    "TimeUnits",
    "build_canonical_axes_handler",
    "canonical_axes_order",
    "canonical_label_axes_order",
    "default_channel_name",
    "path_in_well_validation",
]
