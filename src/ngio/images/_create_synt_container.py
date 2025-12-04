"""Abstract class for handling OME-NGFF images."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import PIL.Image
from zarr.core.array import CompressorLike

from ngio.common._pyramid import ChunksLike, ShardsLike
from ngio.common._synt_images_utils import fit_to_shape
from ngio.images._ome_zarr_container import OmeZarrContainer, create_ome_zarr_from_array
from ngio.ome_zarr_meta.ngio_specs import (
    Channel,
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
    table_backend: TableBackend = DefaultTableBackend,
    scaling_factors: Sequence[float] | Literal["auto"] = "auto",
    axes_names: Sequence[str] | None = None,
    channels_meta: Sequence[str | Channel] | None = None,
    ngff_version: NgffVersions = DefaultNgffVersion,
    chunks: ChunksLike = "auto",
    shards: ShardsLike | None = None,
    dimension_separator: Literal[".", "/"] = "/",
    compressors: CompressorLike = "auto",
    extra_array_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> OmeZarrContainer:
    """Create a synthetic OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Sequence[int]): The shape of the image.
        reference_sample (AVAILABLE_SAMPLES | SampleInfo): The reference sample to use.
            Defaults to "Cardiomyocyte".
        levels (int | list[str]): The number of levels in the pyramid or a list of
            level names. Defaults to 5.
        table_backend (TableBackend): Table backend to be used to store tables.
            Defaults to DefaultTableBackend.
        scaling_factors (Sequence[float] | Literal["auto"]): The down-scaling factors
            for the pyramid levels. Defaults to "auto".
        axes_names (Sequence[str] | None): The names of the axes. If None the
            canonical names are used. Defaults to None.
        channels_meta (Sequence[str | Channel] | None): The channels metadata.
            Defaults to None.
        ngff_version (NgffVersions): The version of the OME-Zarr specification.
            Defaults to DefaultNgffVersion.
        chunks (ChunksLike): The chunk shape. Defaults to "auto".
        shards (ShardsLike | None): The shard shape. Defaults to None.
        dimension_separator (Literal[".", "/"]): The separator to use for
            dimensions. Defaults to "/".
        compressors (CompressorLike): The compressors to use. Defaults to "auto".
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation. Defaults to None.
        overwrite (bool): Whether to overwrite an existing image. Defaults to False.
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
        pixelsize=sample_info.xy_pixelsize,
        z_spacing=sample_info.z_spacing,
        time_spacing=sample_info.time_spacing,
        levels=levels,
        space_unit=sample_info.space_unit,
        time_unit=sample_info.time_unit,
        axes_names=axes_names,
        channels_meta=channels_meta,
        scaling_factors=scaling_factors,
        extra_array_kwargs=extra_array_kwargs,
        name=sample_info.name,
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        dimension_separator=dimension_separator,
        compressors=compressors,
        ngff_version=ngff_version,
    )

    image = ome_zarr.get_image()
    well_table = image.build_image_roi_table()
    ome_zarr.add_table("well_ROI_table", table=well_table, backend=table_backend)

    for label_info in sample_info.labels:
        ome_zarr.derive_label(name=label_info.name)
        label = ome_zarr.get_label(name=label_info.name)

        ref_label = np.asarray(PIL.Image.open(label_info.label_path))
        ref_label = ref_label.astype(label_info.dtype)

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
