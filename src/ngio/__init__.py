"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.common import ArrayLike, Dimensions, Roi, RoiPixels
from ngio.hcs import (
    OmeZarrPlate,
    OmeZarrWell,
    create_empty_plate,
    create_empty_well,
    open_ome_zarr_plate,
    open_ome_zarr_well,
)
from ngio.images import (
    ChannelSelectionModel,
    Image,
    Label,
    OmeZarrContainer,
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    create_synthetic_ome_zarr,
    open_image,
    open_label,
    open_ome_zarr_container,
)
from ngio.ome_zarr_meta.ngio_specs import (
    AxesSetup,
    DefaultNgffVersion,
    ImageInWellPath,
    NgffVersions,
    PixelSize,
)

__all__ = [
    "ArrayLike",
    "AxesSetup",
    "ChannelSelectionModel",
    "DefaultNgffVersion",
    "Dimensions",
    "Image",
    "ImageInWellPath",
    "Label",
    "NgffVersions",
    "OmeZarrContainer",
    "OmeZarrPlate",
    "OmeZarrWell",
    "PixelSize",
    "Roi",
    "RoiPixels",
    "create_empty_ome_zarr",
    "create_empty_plate",
    "create_empty_well",
    "create_ome_zarr_from_array",
    "create_synthetic_ome_zarr",
    "open_image",
    "open_label",
    "open_ome_zarr_container",
    "open_ome_zarr_plate",
    "open_ome_zarr_well",
]
