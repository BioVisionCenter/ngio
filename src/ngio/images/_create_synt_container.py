"""Abstract class for handling OME-NGFF images."""

from collections.abc import Sequence

import numpy as np
import PIL.Image

from ngio.common._synt_images_utils import fit_to_shape
from ngio.images._ome_zarr_container import OmeZarrContainer, create_ome_zarr_from_array
from ngio.ome_zarr_meta.ngio_specs import (
    DefaultNgffVersion,
    NgffVersions,
)
from ngio.resources import AVAILABLE_SAMPLES, SampleInfo, get_sample_info
from ngio.tables import (
    DefaultTableBackend,
    TableBackend,
)
from ngio.utils import (
    StoreOrGroup,
)


def create_synthetic_ome_zarr(
    store: StoreOrGroup,
    shape: Sequence[int],
    reference_sample: AVAILABLE_SAMPLES | SampleInfo = "Cardiomyocyte",
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2,
    z_scaling_factor: float = 1.0,
    axes_names: Sequence[str] | None = None,
    chunks: Sequence[int] | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_colors: Sequence[str] | None = None,
    channel_active: Sequence[bool] | None = None,
    table_backend: TableBackend = DefaultTableBackend,
    overwrite: bool = False,
    version: NgffVersions = DefaultNgffVersion,
) -> OmeZarrContainer:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        reference_sample (AVAILABLE_SAMPLES | SampleInfo): The reference sample to use.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        axes_names (Sequence[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        chunks (Sequence[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_colors (Sequence[str] | None, optional): The colors of the channels.
            Defaults to None.
        channel_active (Sequence[bool] | None, optional): Whether the channels are
            active. Defaults to None.
        table_backend (TableBackend): Table backend to be used to store tables
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (NgffVersion, optional): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
    """
    if isinstance(reference_sample, str):
        sample_info = get_sample_info(reference_sample)
    else:
        sample_info = reference_sample

    raw = np.asarray(PIL.Image.open(sample_info.img_path))
    raw = fit_to_shape(arr=raw, out_shape=tuple(shape))
    raw = raw / np.max(raw) * (2**16 - 1)
    raw = raw.astype(np.uint16)
    ome_zarr = create_ome_zarr_from_array(
        store=store,
        array=raw,
        xy_pixelsize=sample_info.xy_pixelsize,
        z_spacing=sample_info.z_spacing,
        time_spacing=sample_info.time_spacing,
        levels=levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=sample_info.space_unit,
        time_unit=sample_info.time_unit,
        axes_names=axes_names,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_colors=channel_colors,
        channel_active=channel_active,
        name=sample_info.name,
        chunks=chunks,
        overwrite=overwrite,
        version=version,
    )

    image = ome_zarr.get_image()
    well_table = image.build_image_roi_table()
    ome_zarr.add_table("well_ROI_table", table=well_table, backend=table_backend)

    for label_info in sample_info.labels:
        ome_zarr.derive_label(name=label_info.name)
        label = ome_zarr.get_label(name=label_info.name)

        ref_label = np.asarray(PIL.Image.open(label_info.label_path))
        ref_label = fit_to_shape(
            arr=ref_label,
            out_shape=label.shape,
            ensure_unique_info=label_info.ensure_unique_labels,
        )
        ref_label = ref_label.astype(np.uint32)
        label.set_array(ref_label)
        label.consolidate()

        if label_info.create_masking_table:
            masking_table = label.build_masking_roi_table()
            ome_zarr.add_table(
                name=f"{label_info.name}_masking_table",
                table=masking_table,
                backend=table_backend,
            )

    return ome_zarr
