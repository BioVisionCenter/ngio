from typing import Annotated

from ome_zarr_models.v05.well import WellAttrs as WellAttrs05
from ome_zarr_models.v05.well_types import WellImage as WellImage05
from ome_zarr_models.v05.well_types import WellMeta as WellMeta05
from pydantic import SkipValidation


class CustomWellImage(WellImage05):
    path: Annotated[str, SkipValidation]


class CustomWellMeta(WellMeta05):
    images: list[CustomWellImage]  # type: ignore[valid-type]


class CustomWellAttrs(WellAttrs05):
    well: CustomWellMeta  # type: ignore[valid-type]
