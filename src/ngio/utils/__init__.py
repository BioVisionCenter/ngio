"""Various utilities for the ngio package."""

import os

from ngio.utils._datasets import (
    download_ome_zarr_dataset,
    list_ome_zarr_datasets,
    print_datasets_infos,
)
from ngio.utils._errors import (
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
)
from ngio.utils._fractal_fsspec_store import fractal_fsspec_store
from ngio.utils._logger import ngio_logger, ngio_warn, set_logger_level
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    StoreOrGroup,
    ZarrGroupHandler,
    open_group_wrapper,
)

set_logger_level(os.getenv("NGIO_LOGGER_LEVEL", "WARNING"))

__all__ = [
    # Zarr
    "AccessModeLiteral",
    # Errors
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioTableValidationError",
    "NgioValidationError",
    "NgioValueError",
    "StoreOrGroup",
    "ZarrGroupHandler",
    # Other
    "download_ome_zarr_dataset",
    "fractal_fsspec_store",
    "list_ome_zarr_datasets",
    # Logger
    "ngio_logger",
    "ngio_warn",
    "open_group_wrapper",
    "print_datasets_infos",
    "set_logger_level",
]
