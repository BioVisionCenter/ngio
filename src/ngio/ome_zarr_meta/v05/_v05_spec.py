"""Utilities for OME-Zarr v05 specs.

This module provides a set of classes to internally handle the metadata
of the OME-Zarr v05 specification.

For Images and Labels implements the following functionalities:
- A function to find if a dict view of the metadata is a valid OME-Zarr v05 metadata.
- A function to convert a v05 image metadata to a ngio image metadata.
- A function to convert a ngio image metadata to a v05 image metadata.
"""

from ome_zarr_models.common.omero import Channel as ChannelV05
from ome_zarr_models.common.omero import Omero as OmeroV05
from ome_zarr_models.common.omero import Window as WindowV05
from ome_zarr_models.v05.axes import Axis as AxisV05
from ome_zarr_models.v05.coordinate_transformations import VectorScale as VectorScaleV05
from ome_zarr_models.v05.coordinate_transformations import (
    VectorTranslation as VectorTranslationV05,
)
from ome_zarr_models.v05.hcs import HCSAttrs as HCSAttrsV05
from ome_zarr_models.v05.image import ImageAttrs as ImageAttrsV05
from ome_zarr_models.v05.image_label import ImageLabelAttrs as ImageLabelAttrsV05
from ome_zarr_models.v05.labels import Labels as Labels
from ome_zarr_models.v05.labels import LabelsAttrs as LabelsAttrsV05
from ome_zarr_models.v05.multiscales import Dataset as DatasetV05
from ome_zarr_models.v05.multiscales import Multiscale as MultiscaleV05
from ome_zarr_models.v05.multiscales import ValidTransform as ValidTransformV05
from pydantic import BaseModel

from ngio.ome_zarr_meta.ngio_specs import (
    AxesHandler,
    AxesSetup,
    Axis,
    AxisType,
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    Dataset,
    ImageLabelSource,
    NgioImageMeta,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    NgioPlateMeta,
    NgioWellMeta,
    default_channel_name,
)
from ngio.ome_zarr_meta.v05._custom_models import CustomWellAttrs as WellAttrsV05


class ImageV05AttrsWithOmero(ImageAttrsV05):
    omero: OmeroV05 | None = None


class ImageV05WithOmero(BaseModel):
    ome: ImageV05AttrsWithOmero


class ImageLabelV05(BaseModel):
    ome: ImageLabelAttrsV05


def _v05_omero_to_channels(v05_omero: OmeroV05 | None) -> ChannelsMeta | None:
    if v05_omero is None:
        return None

    ngio_channels = []
    for idx, v05_channel in enumerate(v05_omero.channels):
        channel_extra = v05_channel.model_extra

        if channel_extra is None:
            channel_extra = {}

        if "label" in channel_extra:
            label = channel_extra.pop("label")
        else:
            label = default_channel_name(idx)

        if "wavelength_id" in channel_extra:
            wavelength_id = channel_extra.pop("wavelength_id")
        else:
            wavelength_id = label

        if "active" in channel_extra:
            active = channel_extra.pop("active")
        else:
            active = True

        channel_visualisation = ChannelVisualisation(
            color=v05_channel.color,
            start=v05_channel.window.start,
            end=v05_channel.window.end,
            min=v05_channel.window.min,
            max=v05_channel.window.max,
            active=active,
            **channel_extra,
        )

        ngio_channels.append(
            Channel(
                label=label,
                wavelength_id=wavelength_id,
                channel_visualisation=channel_visualisation,
            )
        )

    v05_omero_extra = v05_omero.model_extra if v05_omero.model_extra is not None else {}
    return ChannelsMeta(channels=ngio_channels, **v05_omero_extra)


def _compute_scale_translation(
    v05_transforms: ValidTransformV05,
    scale: list[float],
    translation: list[float],
) -> tuple[list[float], list[float]]:
    for v05_transform in v05_transforms:
        if isinstance(v05_transform, VectorScaleV05):
            scale = [t1 * t2 for t1, t2 in zip(scale, v05_transform.scale, strict=True)]

        elif isinstance(v05_transform, VectorTranslationV05):
            translation = [
                t1 + t2
                for t1, t2 in zip(translation, v05_transform.translation, strict=True)
            ]
        else:
            raise NotImplementedError(
                f"Coordinate transformation {v05_transform} is not supported."
            )
    return scale, translation


def _v05_to_ngio_datasets(
    v05_multiscale: MultiscaleV05,
    axes_setup: AxesSetup,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> list[Dataset]:
    """Convert a v05 multiscale to a list of ngio datasets."""
    datasets = []

    global_scale = [1.0] * len(v05_multiscale.axes)
    global_translation = [0.0] * len(v05_multiscale.axes)

    if v05_multiscale.coordinateTransformations is not None:
        global_scale, global_translation = _compute_scale_translation(
            v05_multiscale.coordinateTransformations, global_scale, global_translation
        )

    # Prepare axes handler
    axes = []
    for v05_axis in v05_multiscale.axes:
        unit = v05_axis.unit
        if unit is not None and not isinstance(unit, str):
            unit = str(unit)
        axes.append(
            Axis(
                name=str(v05_axis.name),
                axis_type=AxisType(v05_axis.type),
                # (for some reason the type is a generic JsonValue,
                # but it should be a string or None)
                unit=v05_axis.unit,  # type: ignore
            )
        )
    axes_handler = AxesHandler(
        axes=axes,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

    for v05_dataset in v05_multiscale.datasets:
        _scale, _translation = _compute_scale_translation(
            v05_dataset.coordinateTransformations, global_scale, global_translation
        )
        datasets.append(
            Dataset(
                path=v05_dataset.path,
                axes_handler=axes_handler,
                scale=_scale,
                translation=_translation,
            )
        )
    return datasets


def v05_to_ngio_image_meta(
    metadata: dict,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> NgioImageMeta:
    """Convert a v05 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v05 image metadata.
        axes_setup (AxesSetup, optional): The axes setup. This is
            required to convert image with non-canonical axes names.
        allow_non_canonical_axes (bool, optional): Allow non-canonical axes.
        strict_canonical_order (bool, optional): Strict canonical order.

    Returns:
        NgioImageMeta: The ngio image metadata.
    """
    v05_image = ImageV05WithOmero(**metadata)
    v05_image = v05_image.ome
    if len(v05_image.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v05_multiscale = v05_image.multiscales[0]

    channels_meta = _v05_omero_to_channels(v05_image.omero)
    axes_setup = axes_setup if axes_setup is not None else AxesSetup()
    datasets = _v05_to_ngio_datasets(
        v05_multiscale,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

    name = v05_multiscale.name
    if name is not None and not isinstance(name, str):
        name = str(name)
    return NgioImageMeta(
        version="0.5",
        name=name,
        datasets=datasets,
        channels=channels_meta,
    )


def v05_to_ngio_label_meta(
    metadata: dict,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> NgioLabelMeta:
    """Convert a v05 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v05 image metadata.
        axes_setup (AxesSetup, optional): The axes setup. This is
            required to convert image with non-canonical axes names.
        allow_non_canonical_axes (bool, optional): Allow non-canonical axes.
        strict_canonical_order (bool, optional): Strict canonical order.

    Returns:
        NgioLabelMeta: The ngio label metadata.
    """
    v05_label = ImageLabelV05(**metadata)
    v05_label = v05_label.ome

    if len(v05_label.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v05_multiscale = v05_label.multiscales[0]

    axes_setup = axes_setup if axes_setup is not None else AxesSetup()
    datasets = _v05_to_ngio_datasets(
        v05_multiscale,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

    if v05_label.image_label is not None:
        source = v05_label.image_label.source
        if source is None:
            image_label_source = None
        else:
            source = v05_label.image_label.source
            if source is None:
                image_label_source = None
            else:
                image_label_source = source.image
            image_label_source = ImageLabelSource(
                version="0.5",
                source={"image": image_label_source},
            )
    else:
        image_label_source = None
    name = v05_multiscale.name
    if name is not None and not isinstance(name, str):
        name = str(name)

    return NgioLabelMeta(
        version="0.5",
        name=name,
        datasets=datasets,
        image_label=image_label_source,
    )


def _ngio_to_v05_multiscale(name: str | None, datasets: list[Dataset]) -> MultiscaleV05:
    """Convert a ngio multiscale to a v05 multiscale.

    Args:
        name (str | None): The name of the multiscale.
        datasets (list[Dataset]): The ngio datasets.

    Returns:
        MultiscaleV05: The v05 multiscale.
    """
    ax_mapper = datasets[0].axes_handler
    v05_axes = []
    for axis in ax_mapper.axes:
        v05_axes.append(
            AxisV05(
                name=axis.name,
                type=axis.axis_type.value if axis.axis_type is not None else None,
                unit=axis.unit if axis.unit is not None else None,
            )
        )

    v05_datasets = []
    for dataset in datasets:
        transform = [VectorScaleV05(type="scale", scale=list(dataset._scale))]
        if sum(dataset._translation) > 0:
            transform = (
                VectorScaleV05(type="scale", scale=list(dataset._scale)),
                VectorTranslationV05(
                    type="translation", translation=list(dataset._translation)
                ),
            )
        else:
            transform = (VectorScaleV05(type="scale", scale=list(dataset._scale)),)

        v05_datasets.append(
            DatasetV05(path=dataset.path, coordinateTransformations=transform)
        )
    return MultiscaleV05(axes=v05_axes, datasets=tuple(v05_datasets), name=name)


def _ngio_to_v05_omero(channels: ChannelsMeta | None) -> OmeroV05 | None:
    """Convert a ngio channels to a v05 omero."""
    if channels is None:
        return None

    v05_channels = []
    for channel in channels.channels:
        _model_extra = {
            "label": channel.label,
            "wavelength_id": channel.wavelength_id,
            "active": channel.channel_visualisation.active,
        }
        if channel.channel_visualisation.model_extra is not None:
            _model_extra.update(channel.channel_visualisation.model_extra)

        v05_channels.append(
            ChannelV05(
                color=channel.channel_visualisation.valid_color,
                window=WindowV05(
                    start=channel.channel_visualisation.start,
                    end=channel.channel_visualisation.end,
                    min=channel.channel_visualisation.min,
                    max=channel.channel_visualisation.max,
                ),
                **_model_extra,
            )
        )

    _model_extra = channels.model_extra if channels.model_extra is not None else {}
    return OmeroV05(channels=v05_channels, **_model_extra)


def ngio_to_v05_image_meta(metadata: NgioImageMeta) -> dict:
    """Convert a ngio image metadata to a v05 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v05 image metadata.
    """
    v05_muliscale = _ngio_to_v05_multiscale(
        name=metadata.name, datasets=metadata.datasets
    )
    v05_omero = _ngio_to_v05_omero(metadata._channels_meta)

    v05_image_attrs = ImageV05AttrsWithOmero(
        multiscales=[v05_muliscale], omero=v05_omero, version="0.5"
    )
    v05_image = ImageV05WithOmero(
        ome=v05_image_attrs,
    )
    return v05_image.model_dump(exclude_none=True, by_alias=True)


def ngio_to_v05_label_meta(metadata: NgioLabelMeta) -> dict:
    """Convert a ngio image metadata to a v05 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v05 image metadata.
    """
    v05_muliscale = _ngio_to_v05_multiscale(
        name=metadata.name, datasets=metadata.datasets
    )
    labels_meta = {
        "multiscales": [v05_muliscale],
        "image-label": metadata.image_label.model_dump(),
    }
    v05_label = ImageLabelAttrsV05(**labels_meta, version="0.5")
    v05_label = ImageLabelV05(
        ome=v05_label,
    )
    return v05_label.model_dump(exclude_none=True, by_alias=True)


class WellV05(BaseModel):
    ome: WellAttrsV05


class HCSV05(BaseModel):
    ome: HCSAttrsV05


def v05_to_ngio_well_meta(
    metadata: dict,
) -> NgioWellMeta:
    """Convert a v05 well metadata to a ngio well metadata.

    Args:
        metadata (dict): The v05 well metadata.

    Returns:
        NgioWellMeta: The ngio well metadata.
    """
    v05_well = WellV05(**metadata).ome.well.model_dump()
    images = v05_well.get("images", [])
    return NgioWellMeta(images=images, version="0.5")


def v05_to_ngio_plate_meta(
    metadata: dict,
) -> NgioPlateMeta:
    """Convert a v05 plate metadata to a ngio plate metadata.

    Args:
        metadata (dict): The v05 plate metadata.

    Returns:
        NgioPlateMeta: The ngio plate metadata.
    """
    v05_plate = HCSV05(**metadata).ome.plate.model_dump()
    return NgioPlateMeta(plate=v05_plate, version="0.5")  # type: ignore


def ngio_to_v05_well_meta(metadata: NgioWellMeta) -> dict:
    """Convert a ngio well metadata to a v05 well metadata.

    Args:
        metadata (NgioWellMeta): The ngio well metadata.

    Returns:
        dict: The v05 well metadata.
    """
    v05_well = WellAttrsV05(well=metadata.model_dump())  # type: ignore
    v05_well = WellV05(ome=v05_well)
    return v05_well.model_dump(exclude_none=True, by_alias=True)


def ngio_to_v05_plate_meta(metadata: NgioPlateMeta) -> dict:
    """Convert a ngio plate metadata to a v05 plate metadata.

    Args:
        metadata (NgioPlateMeta): The ngio plate metadata.

    Returns:
        dict: The v05 plate metadata.
    """
    v05_plate = HCSAttrsV05(**metadata.model_dump())
    v05_plate = HCSV05(ome=v05_plate)
    return v05_plate.model_dump(exclude_none=True, by_alias=True)


class LabelsV05(BaseModel):
    ome: LabelsAttrsV05


def v05_to_ngio_labels_group_meta(
    metadata: dict,
) -> NgioLabelsGroupMeta:
    """Convert a v04 label group metadata to a ngio label group metadata.

    Args:
        metadata (dict): The v04 label group metadata.

    Returns:
        NgioLabelGroupMeta: The ngio label group metadata.
    """
    v05_label_group = LabelsV05(**metadata)
    return NgioLabelsGroupMeta(labels=v05_label_group.ome.labels, version="0.5")


def ngio_to_v05_labels_group_meta(metadata: NgioLabelsGroupMeta) -> dict:
    """Convert a ngio label group metadata to a v05 label group metadata.

    Args:
        metadata (NgioLabelsGroupMeta): The ngio label group metadata.

    Returns:
        dict: The v05 label group metadata.
    """
    v05_labels_attrs = LabelsAttrsV05(labels=metadata.labels, version="0.5")
    v05_labels_group = LabelsV05(ome=v05_labels_attrs)
    return v05_labels_group.model_dump(exclude_none=True, by_alias=True)
