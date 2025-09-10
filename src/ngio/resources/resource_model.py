"""Models for programmatic description of image resources."""

from pathlib import Path

from pydantic import BaseModel, Field

from ngio.ome_zarr_meta.ngio_specs import (
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
)


class LabelsInfo(BaseModel):
    """Metadata for a label image."""

    name: str
    label_path: Path
    ensure_unique_labels: bool = True
    create_masking_table: bool = False


class SampleInfo(BaseModel):
    """Metadata necessary to create an OME-Ngff from image files."""

    img_path: Path
    labels: list[LabelsInfo] = Field(default_factory=list)
    xy_pixelsize: float
    z_spacing: float = 1.0
    time_spacing: float = 1.0
    space_unit: SpaceUnits = DefaultSpaceUnit
    time_unit: TimeUnits = DefaultTimeUnit
    name: str | None = None
    info: str | None = None
