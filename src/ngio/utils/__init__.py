"""Various utilities for the ngio package."""

from ngio.utils._datasets import (
    download_ome_zarr_dataset,
    list_ome_zarr_datasets,
    print_datasets_infos,
)
from ngio.utils._deprecate import deprecated, deprecated_alias
from ngio.utils._errors import (
    NgioError,
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioKeyError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
)
from ngio.utils._fractal_fsspec_store import fractal_fsspec_store
from ngio.utils._retry import retry_io
from ngio.utils._store import NgioStore
from ngio.utils._warnings import (
    NgioDeprecationWarning,
    NgioFutureWarning,
    NgioUserWarning,
    stacklevel_of_first_caller,
)
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    NgioCache,
    NgioSupportedStore,
    StoreOrGroup,
    ZarrGroupHandler,
    copy_group,
    open_group_wrapper,
    refresh_s3fs_config,
)

__all__ = [
    "AccessModeLiteral",
    "NgioCache",
    "NgioDeprecationWarning",
    "NgioError",
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioFutureWarning",
    "NgioKeyError",
    "NgioStore",
    "NgioSupportedStore",
    "NgioTableValidationError",
    "NgioUserWarning",
    "NgioValidationError",
    "NgioValueError",
    "StoreOrGroup",
    "ZarrGroupHandler",
    "copy_group",
    "deprecated",
    "deprecated_alias",
    "download_ome_zarr_dataset",
    "fractal_fsspec_store",
    "list_ome_zarr_datasets",
    "open_group_wrapper",
    "print_datasets_infos",
    "refresh_s3fs_config",
    "retry_io",
    "stacklevel_of_first_caller",
]
