"""Next Generation file format IO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ngio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

from ngio.common import Dimensions, Roi, RoiSlice
from ngio.config import (
    ConsolidationConfig,
    DaskConfig,
    NgioConfig,
    RetryConfig,
    S3FSConfig,
    ZarrConfig,
    get_config,
)
from ngio.hcs import (
    OmeZarrPlate,
    OmeZarrWell,
    create_empty_plate,
    create_empty_well,
    derive_ome_zarr_plate,
    open_ome_zarr_plate,
    open_ome_zarr_well,
)
from ngio.images import (
    ChannelSelectionModel,
    Image,
    Label,
    MaskedImage,
    MaskedLabel,
    OmeZarrContainer,
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    create_synthetic_ome_zarr,
    open_image,
    open_label,
    open_ome_zarr_container,
)
from ngio.iterators import (
    BasicMapper,
    FeatureExtractorIterator,
    ImageProcessingIterator,
    MapperProtocol,
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.ome_zarr_meta import (
    get_ngio_image_meta,
    get_ngio_label_meta,
    get_ngio_labels_group_meta,
    get_ngio_meta,
    get_ngio_plate_meta,
    get_ngio_well_meta,
)
from ngio.ome_zarr_meta.ngio_specs import (
    AxesSetup,
    Channel,
    DefaultNgffVersion,
    ImageInWellPath,
    NgffVersions,
    PixelSize,
)
from ngio.resources import get_sample_info
from ngio.utils import (
    NgioError,
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioKeyError,
    NgioSupportedStore,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
)

__all__ = [
    "AxesSetup",
    "BasicMapper",
    "Channel",
    "ChannelSelectionModel",
    "ConsolidationConfig",
    "DaskConfig",
    "DefaultNgffVersion",
    "Dimensions",
    "FeatureExtractorIterator",
    "Image",
    "ImageInWellPath",
    "ImageProcessingIterator",
    "Label",
    "MapperProtocol",
    "MaskedImage",
    "MaskedLabel",
    "MaskedSegmentationIterator",
    "NgffVersions",
    "NgioConfig",
    "NgioError",
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioKeyError",
    "NgioSupportedStore",
    "NgioTableValidationError",
    "NgioValidationError",
    "NgioValueError",
    "OmeZarrContainer",
    "OmeZarrPlate",
    "OmeZarrWell",
    "PixelSize",
    "RetryConfig",
    "Roi",
    "RoiSlice",
    "S3FSConfig",
    "SegmentationIterator",
    "StoreOrGroup",
    "ZarrConfig",
    "__version__",
    "create_empty_ome_zarr",
    "create_empty_plate",
    "create_empty_well",
    "create_ome_zarr_from_array",
    "create_synthetic_ome_zarr",
    "derive_ome_zarr_plate",
    "get_config",
    "get_ngio_image_meta",
    "get_ngio_label_meta",
    "get_ngio_labels_group_meta",
    "get_ngio_meta",
    "get_ngio_plate_meta",
    "get_ngio_well_meta",
    "get_sample_info",
    "open_image",
    "open_label",
    "open_ome_zarr_container",
    "open_ome_zarr_plate",
    "open_ome_zarr_well",
]
