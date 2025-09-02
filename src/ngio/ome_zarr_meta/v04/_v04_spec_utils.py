"""Utilities for OME-Zarr v04 specs.

This module provides a set of classes to internally handle the metadata
of the OME-Zarr v04 specification.

For Images and Labels implements the following functionalities:
- A function to find if a dict view of the metadata is a valid OME-Zarr v04 metadata.
- A function to convert a v04 image metadata to a ngio image metadata.
- A function to convert a ngio image metadata to a v04 image metadata.
"""

from ome_zarr_models.common.multiscales import ValidTransform as ValidTransformV04
from ome_zarr_models.v04.axes import Axis as AxisV04
from ome_zarr_models.v04.coordinate_transformations import VectorScale as VectorScaleV04
from ome_zarr_models.v04.coordinate_transformations import (
    VectorTranslation as VectorTranslationV04,
)
from ome_zarr_models.v04.hcs import HCSAttrs as HCSAttrsV04
from ome_zarr_models.v04.image import ImageAttrs as ImageAttrsV04
from ome_zarr_models.v04.image_label import ImageLabelAttrs as LabelAttrsV04
from ome_zarr_models.v04.multiscales import Dataset as DatasetV04
from ome_zarr_models.v04.multiscales import Multiscale as MultiscaleV04
from ome_zarr_models.v04.omero import Channel as ChannelV04
from ome_zarr_models.v04.omero import Omero as OmeroV04
from ome_zarr_models.v04.omero import Window as WindowV04
from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import (
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
    NgioPlateMeta,
    NgioWellMeta,
    default_channel_name,
)
from ngio.ome_zarr_meta.v04._custom_models import CustomWellAttrs as WellAttrsV04


def _is_v04_image_meta(metadata: dict) -> ImageAttrsV04 | ValidationError:
    """Check if the metadata is a valid OME-Zarr v04 metadata.

    Args:
        metadata (dict): The metadata to check.

    Returns:
        bool: True if the metadata is a valid OME-Zarr v04 metadata, False otherwise.
    """
    try:
        return ImageAttrsV04(**metadata)
    except ValidationError as e:
        return e


def _is_v04_label_meta(metadata: dict) -> LabelAttrsV04 | ValidationError:
    """Check if the metadata is a valid OME-Zarr v04 metadata.

    Args:
        metadata (dict): The metadata to check.

    Returns:
        bool: True if the metadata is a valid OME-Zarr v04 metadata, False otherwise.
    """
    try:
        return LabelAttrsV04(**metadata)
    except ValidationError as e:
        return e
    raise RuntimeError("Unreachable code")


def _v04_omero_to_channels(v04_omero: OmeroV04 | None) -> ChannelsMeta | None:
    if v04_omero is None:
        return None

    ngio_channels = []
    for idx, v04_channel in enumerate(v04_omero.channels):
        channel_extra = v04_channel.model_extra

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
            color=v04_channel.color,
            start=v04_channel.window.start,
            end=v04_channel.window.end,
            min=v04_channel.window.min,
            max=v04_channel.window.max,
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

    v04_omero_extra = v04_omero.model_extra if v04_omero.model_extra is not None else {}
    return ChannelsMeta(channels=ngio_channels, **v04_omero_extra)


def _compute_scale_translation(
    v04_transforms: ValidTransformV04,
    scale: list[float],
    translation: list[float],
) -> tuple[list[float], list[float]]:
    for v04_transform in v04_transforms:
        if isinstance(v04_transform, VectorScaleV04):
            scale = [t1 * t2 for t1, t2 in zip(scale, v04_transform.scale, strict=True)]

        elif isinstance(v04_transform, VectorTranslationV04):
            translation = [
                t1 + t2
                for t1, t2 in zip(translation, v04_transform.translation, strict=True)
            ]
        else:
            raise NotImplementedError(
                f"Coordinate transformation {v04_transform} is not supported."
            )
    return scale, translation


def _v04_to_ngio_datasets(
    v04_multiscale: MultiscaleV04,
    axes_setup: AxesSetup,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> list[Dataset]:
    """Convert a v04 multiscale to a list of ngio datasets."""
    datasets = []

    global_scale = [1.0] * len(v04_multiscale.axes)
    global_translation = [0.0] * len(v04_multiscale.axes)

    if v04_multiscale.coordinateTransformations is not None:
        global_scale, global_translation = _compute_scale_translation(
            v04_multiscale.coordinateTransformations, global_scale, global_translation
        )

    for v04_dataset in v04_multiscale.datasets:
        axes = []
        for v04_axis in v04_multiscale.axes:
            unit = v04_axis.unit
            if unit is not None and not isinstance(unit, str):
                unit = str(unit)
            axes.append(
                Axis(
                    on_disk_name=v04_axis.name,
                    axis_type=AxisType(v04_axis.type),
                    # (for some reason the type is a generic JsonValue,
                    # but it should be a string or None)
                    unit=v04_axis.unit,  # type: ignore
                )
            )

        _on_disk_scale, _on_disk_translation = _compute_scale_translation(
            v04_dataset.coordinateTransformations, global_scale, global_translation
        )
        datasets.append(
            Dataset(
                path=v04_dataset.path,
                on_disk_axes=axes,
                on_disk_scale=_on_disk_scale,
                on_disk_translation=_on_disk_translation,
                axes_setup=axes_setup,
                allow_non_canonical_axes=allow_non_canonical_axes,
                strict_canonical_order=strict_canonical_order,
            )
        )
    return datasets


def v04_to_ngio_image_meta(
    metadata: dict,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> tuple[bool, NgioImageMeta | ValidationError]:
    """Convert a v04 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v04 image metadata.
        axes_setup (AxesSetup, optional): The axes setup. This is
            required to convert image with non-canonical axes names.
        allow_non_canonical_axes (bool, optional): Allow non-canonical axes.
        strict_canonical_order (bool, optional): Strict canonical order.

    Returns:
        NgioImageMeta: The ngio image metadata.
    """
    v04_image = _is_v04_image_meta(metadata)
    if isinstance(v04_image, ValidationError):
        return False, v04_image

    if len(v04_image.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v04_muliscale = v04_image.multiscales[0]

    channels_meta = _v04_omero_to_channels(v04_image.omero)
    axes_setup = axes_setup if axes_setup is not None else AxesSetup()
    datasets = _v04_to_ngio_datasets(
        v04_muliscale,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

    name = v04_muliscale.name
    if name is not None and not isinstance(name, str):
        name = str(name)
    return True, NgioImageMeta(
        version="0.4",
        name=name,
        datasets=datasets,
        channels=channels_meta,
    )


def v04_to_ngio_label_meta(
    metadata: dict,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> tuple[bool, NgioLabelMeta | ValidationError]:
    """Convert a v04 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v04 image metadata.
        axes_setup (AxesSetup, optional): The axes setup. This is
            required to convert image with non-canonical axes names.
        allow_non_canonical_axes (bool, optional): Allow non-canonical axes.
        strict_canonical_order (bool, optional): Strict canonical order.

    Returns:
        NgioImageMeta: The ngio image metadata.
    """
    v04_label = _is_v04_label_meta(metadata)
    if isinstance(v04_label, ValidationError):
        return False, v04_label

    if len(v04_label.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v04_muliscale = v04_label.multiscales[0]

    axes_setup = axes_setup if axes_setup is not None else AxesSetup()
    datasets = _v04_to_ngio_datasets(
        v04_muliscale,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )

    source = v04_label.image_label.source
    if source is None:
        image_label_source = None
    else:
        source = v04_label.image_label.source
        if source is None:
            image_label_source = None
        else:
            image_label_source = source.image
        image_label_source = ImageLabelSource(
            version="0.4",
            source={"image": image_label_source},
        )
    name = v04_muliscale.name
    if name is not None and not isinstance(name, str):
        name = str(name)

    return True, NgioLabelMeta(
        version="0.4",
        name=name,
        datasets=datasets,
        image_label=image_label_source,
    )


def _ngio_to_v04_multiscale(name: str | None, datasets: list[Dataset]) -> MultiscaleV04:
    """Convert a ngio multiscale to a v04 multiscale.

    Args:
        name (str | None): The name of the multiscale.
        datasets (list[Dataset]): The ngio datasets.

    Returns:
        MultiscaleV04: The v04 multiscale.
    """
    ax_mapper = datasets[0].axes_mapper
    v04_axes = []
    for axis in ax_mapper.axes:
        v04_axes.append(
            AxisV04(
                name=axis.on_disk_name,
                type=axis.axis_type.value if axis.axis_type is not None else None,
                unit=axis.unit if axis.unit is not None else None,
            )
        )

    v04_datasets = []
    for dataset in datasets:
        transform = [VectorScaleV04(type="scale", scale=list(dataset._on_disk_scale))]
        if sum(dataset._on_disk_translation) > 0:
            transform = (
                VectorScaleV04(type="scale", scale=list(dataset._on_disk_scale)),
                VectorTranslationV04(
                    type="translation", translation=list(dataset._on_disk_translation)
                ),
            )
        else:
            transform = (
                VectorScaleV04(type="scale", scale=list(dataset._on_disk_scale)),
            )

        v04_datasets.append(
            DatasetV04(path=dataset.path, coordinateTransformations=transform)
        )
    return MultiscaleV04(
        axes=v04_axes, datasets=tuple(v04_datasets), version="0.4", name=name
    )


def _ngio_to_v04_omero(channels: ChannelsMeta | None) -> OmeroV04 | None:
    """Convert a ngio channels to a v04 omero."""
    if channels is None:
        return None

    v04_channels = []
    for channel in channels.channels:
        _model_extra = {
            "label": channel.label,
            "wavelength_id": channel.wavelength_id,
            "active": channel.channel_visualisation.active,
        }
        if channel.channel_visualisation.model_extra is not None:
            _model_extra.update(channel.channel_visualisation.model_extra)

        v04_channels.append(
            ChannelV04(
                color=channel.channel_visualisation.valid_color,
                window=WindowV04(
                    start=channel.channel_visualisation.start,
                    end=channel.channel_visualisation.end,
                    min=channel.channel_visualisation.min,
                    max=channel.channel_visualisation.max,
                ),
                **_model_extra,
            )
        )

    _model_extra = channels.model_extra if channels.model_extra is not None else {}
    return OmeroV04(channels=v04_channels, **_model_extra)


def ngio_to_v04_image_meta(metadata: NgioImageMeta) -> dict:
    """Convert a ngio image metadata to a v04 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v04 image metadata.
    """
    v04_muliscale = _ngio_to_v04_multiscale(
        name=metadata.name, datasets=metadata.datasets
    )
    v04_omero = _ngio_to_v04_omero(metadata._channels_meta)

    v04_image = ImageAttrsV04(multiscales=[v04_muliscale], omero=v04_omero)
    return v04_image.model_dump(exclude_none=True, by_alias=True)


def ngio_to_v04_label_meta(metadata: NgioLabelMeta) -> dict:
    """Convert a ngio image metadata to a v04 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v04 image metadata.
    """
    v04_muliscale = _ngio_to_v04_multiscale(
        name=metadata.name, datasets=metadata.datasets
    )
    labels_meta = {
        "multiscales": [v04_muliscale],
        "image-label": metadata.image_label.model_dump(),
    }
    v04_label = LabelAttrsV04(**labels_meta)
    return v04_label.model_dump(exclude_none=True, by_alias=True)


def v04_to_ngio_well_meta(
    metadata: dict,
) -> tuple[bool, NgioWellMeta | ValidationError]:
    """Convert a v04 well metadata to a ngio well metadata.

    Args:
        metadata (dict): The v04 well metadata.

    Returns:
        result (bool): True if the conversion was successful, False otherwise.
        ngio_well_meta (NgioWellMeta): The ngio well metadata.
    """
    try:
        v04_well = WellAttrsV04(**metadata)
    except ValidationError as e:
        return False, e

    return True, NgioWellMeta(**v04_well.model_dump())


def v04_to_ngio_plate_meta(
    metadata: dict,
) -> tuple[bool, NgioPlateMeta | ValidationError]:
    """Convert a v04 plate metadata to a ngio plate metadata.

    Args:
        metadata (dict): The v04 plate metadata.

    Returns:
        result (bool): True if the conversion was successful, False otherwise.
        ngio_plate_meta (NgioPlateMeta): The ngio plate metadata.
    """
    try:
        v04_plate = HCSAttrsV04(**metadata)
    except ValidationError as e:
        return False, e

    return True, NgioPlateMeta(**v04_plate.model_dump())


def ngio_to_v04_well_meta(metadata: NgioWellMeta) -> dict:
    """Convert a ngio well metadata to a v04 well metadata.

    Args:
        metadata (NgioWellMeta): The ngio well metadata.

    Returns:
        dict: The v04 well metadata.
    """
    v04_well = WellAttrsV04(**metadata.model_dump())
    return v04_well.model_dump(exclude_none=True, by_alias=True)


def ngio_to_v04_plate_meta(metadata: NgioPlateMeta) -> dict:
    """Convert a ngio plate metadata to a v04 plate metadata.

    Args:
        metadata (NgioPlateMeta): The ngio plate metadata.

    Returns:
        dict: The v04 plate metadata.
    """
    v04_plate = HCSAttrsV04(**metadata.model_dump())
    return v04_plate.model_dump(exclude_none=True, by_alias=True)
