"""Various utilities for the ngio package."""

from ngio.utils._datasets import (
    download_ome_zarr_dataset,
    list_ome_zarr_datasets,
    print_datasets_infos,
)
from ngio.utils._errors import (
    NgioError,
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
)
from ngio.utils._fractal_fsspec_store import fractal_fsspec_store
from ngio.utils._zarr_utils import (
    AccessModeLiteral,
    NgioCache,
    StoreOrGroup,
    ZarrGroupHandler,
    open_group_wrapper,
)

__all__ = [
    "AccessModeLiteral",
    "NgioCache",
    "NgioError",
    "NgioFileExistsError",
    "NgioFileNotFoundError",
    "NgioTableValidationError",
    "NgioValidationError",
    "NgioValueError",
    "StoreOrGroup",
    "ZarrGroupHandler",
    "download_ome_zarr_dataset",
    "fractal_fsspec_store",
    "list_ome_zarr_datasets",
    "open_group_wrapper",
    "print_datasets_infos",
]
