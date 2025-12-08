"""Base class for handling OME-NGFF metadata in Zarr groups."""

from collections.abc import Callable
from typing import TypeVar

from ngio.ome_zarr_meta.ngio_specs import (
    AxesSetup,
    NgioImageMeta,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    NgioPlateMeta,
    NgioWellMeta,
)
from ngio.ome_zarr_meta.ngio_specs._ngio_image import NgffVersions
from ngio.ome_zarr_meta.v04 import (
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    ngio_to_v04_labels_group_meta,
    ngio_to_v04_plate_meta,
    ngio_to_v04_well_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
    v04_to_ngio_labels_group_meta,
    v04_to_ngio_plate_meta,
    v04_to_ngio_well_meta,
)
from ngio.ome_zarr_meta.v05 import (
    ngio_to_v05_image_meta,
    ngio_to_v05_label_meta,
    ngio_to_v05_labels_group_meta,
    ngio_to_v05_plate_meta,
    ngio_to_v05_well_meta,
    v05_to_ngio_image_meta,
    v05_to_ngio_label_meta,
    v05_to_ngio_labels_group_meta,
    v05_to_ngio_plate_meta,
    v05_to_ngio_well_meta,
)
from ngio.utils import (
    NgioValidationError,
    NgioValueError,
    ZarrGroupHandler,
)

# This could be replaced with a more dynamic registry if needed in the future
_image_encoder_registry = {"0.4": ngio_to_v04_image_meta, "0.5": ngio_to_v05_image_meta}
_image_decoder_registry = {"0.4": v04_to_ngio_image_meta, "0.5": v05_to_ngio_image_meta}
_label_encoder_registry = {"0.4": ngio_to_v04_label_meta, "0.5": ngio_to_v05_label_meta}
_label_decoder_registry = {"0.4": v04_to_ngio_label_meta, "0.5": v05_to_ngio_label_meta}
_plate_encoder_registry = {"0.4": ngio_to_v04_plate_meta, "0.5": ngio_to_v05_plate_meta}
_plate_decoder_registry = {"0.4": v04_to_ngio_plate_meta, "0.5": v05_to_ngio_plate_meta}
_well_encoder_registry = {"0.4": ngio_to_v04_well_meta, "0.5": ngio_to_v05_well_meta}
_well_decoder_registry = {"0.4": v04_to_ngio_well_meta, "0.5": v05_to_ngio_well_meta}
_labels_group_encoder_registry = {
    "0.4": ngio_to_v04_labels_group_meta,
    "0.5": ngio_to_v05_labels_group_meta,
}
_labels_group_decoder_registry = {
    "0.4": v04_to_ngio_labels_group_meta,
    "0.5": v05_to_ngio_labels_group_meta,
}

_meta_type = TypeVar(
    "_meta_type",
    NgioImageMeta,
    NgioLabelMeta,
    NgioLabelsGroupMeta,
    NgioPlateMeta,
    NgioWellMeta,
)


def _find_encoder_registry(
    ngio_meta: _meta_type,
) -> dict[str, Callable]:
    if isinstance(ngio_meta, NgioImageMeta):
        return _image_encoder_registry
    elif isinstance(ngio_meta, NgioLabelMeta):
        return _label_encoder_registry
    elif isinstance(ngio_meta, NgioPlateMeta):
        return _plate_encoder_registry
    elif isinstance(ngio_meta, NgioWellMeta):
        return _well_encoder_registry
    elif isinstance(ngio_meta, NgioLabelsGroupMeta):
        return _labels_group_encoder_registry
    else:
        raise NgioValueError(f"Unsupported NGIO metadata type: {type(ngio_meta)}")


def update_ngio_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: _meta_type,
) -> None:
    """Update the metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (_meta_type): The new NGIO metadata.

    """
    registry = _find_encoder_registry(ngio_meta)
    exporter = registry.get(ngio_meta.version)
    if exporter is None:
        raise NgioValueError(f"Unsupported NGFF version: {ngio_meta.version}")

    zarr_meta = exporter(ngio_meta)
    group_handler.write_attrs(zarr_meta)


def _find_decoder_registry(
    meta_type: type[_meta_type],
) -> dict[str, Callable]:
    if meta_type is NgioImageMeta:
        return _image_decoder_registry
    elif meta_type is NgioLabelMeta:
        return _label_decoder_registry
    elif meta_type is NgioPlateMeta:
        return _plate_decoder_registry
    elif meta_type is NgioWellMeta:
        return _well_decoder_registry
    elif meta_type is NgioLabelsGroupMeta:
        return _labels_group_decoder_registry
    else:
        raise NgioValueError(f"Unsupported NGIO metadata type: {meta_type}")


def get_ngio_meta(
    group_handler: ZarrGroupHandler,
    meta_type: type[_meta_type],
    version: str | None = None,
    **kwargs,
) -> _meta_type:
    """Retrieve the NGIO metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        meta_type (type[_meta_type]): The type of NGIO metadata to retrieve.
        version (str | None): Optional NGFF version to use for decoding.
        **kwargs: Additional arguments to pass to the decoder.

    Returns:
        _meta_type: The NGIO metadata.
    """
    registry = _find_decoder_registry(meta_type)
    if version is not None:
        decoder = registry.get(version)
        if decoder is None:
            raise NgioValueError(f"Unsupported NGFF version: {version}")
        versions_to_try = {version: decoder}
    else:
        versions_to_try = registry

    attrs = group_handler.load_attrs()
    all_errors = []
    for version, decoder in versions_to_try.items():
        try:
            ngio_meta = decoder(attrs, **kwargs)
            return ngio_meta
        except Exception as e:
            all_errors.append(f"Version {version}: {e}")
    error_message = (
        f"Failed to decode NGIO {meta_type.__name__} metadata:\n"
        + "\n".join(all_errors)
    )
    raise NgioValidationError(error_message)


##################################################
#
# Concrete implementations for NGIO metadata types
#
##################################################


def get_ngio_image_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> NgioImageMeta:
    """Retrieve the NGIO image metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        axes_setup (AxesSetup | None): Optional axes setup for validation.
        allow_non_canonical_axes (bool): Whether to allow non-canonical axes.
        strict_canonical_order (bool): Whether to enforce strict canonical order.

    Returns:
        NgioImageMeta: The NGIO image metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioImageMeta,
        version=version,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def update_ngio_image_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: NgioImageMeta,
) -> None:
    """Update the NGIO image metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (NgioImageMeta): The new NGIO image metadata.

    """
    update_ngio_meta(
        group_handler=group_handler,
        ngio_meta=ngio_meta,
    )


class ImageMetaHandler:
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        version: str | None = None,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        self._group_handler = group_handler
        self._version = version
        self._axes_setup = axes_setup
        self._allow_non_canonical_axes = allow_non_canonical_axes
        self._strict_canonical_order = strict_canonical_order

        # Validate metadata
        meta = self.get_meta()
        # Store the resolved version
        self._version = meta.version

    def get_meta(self) -> NgioImageMeta:
        """Retrieve the NGIO image metadata."""
        return get_ngio_image_meta(
            group_handler=self._group_handler,
            version=self._version,
            axes_setup=self._axes_setup,
            allow_non_canonical_axes=self._allow_non_canonical_axes,
            strict_canonical_order=self._strict_canonical_order,
        )

    def update_meta(self, ngio_meta: NgioImageMeta) -> None:
        """Update the NGIO image metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )


def get_ngio_label_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> NgioLabelMeta:
    """Retrieve the NGIO label metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        axes_setup (AxesSetup | None): Optional axes setup for validation.
        allow_non_canonical_axes (bool): Whether to allow non-canonical axes.
        strict_canonical_order (bool): Whether to enforce strict canonical order.

    Returns:
        NgioLabelMeta: The NGIO label metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioLabelMeta,
        version=version,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def update_ngio_label_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: NgioLabelMeta,
) -> None:
    """Update the NGIO label metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (NgioLabelMeta): The new NGIO label metadata.

    """
    update_ngio_meta(
        group_handler=group_handler,
        ngio_meta=ngio_meta,
    )


class LabelMetaHandler:
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        version: str | None = None,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        self._group_handler = group_handler
        self._version = version
        self._axes_setup = axes_setup
        self._allow_non_canonical_axes = allow_non_canonical_axes
        self._strict_canonical_order = strict_canonical_order

        # Validate metadata
        meta = self.get_meta()
        # Store the resolved version
        self._version = meta.version

    def get_meta(self) -> NgioLabelMeta:
        """Retrieve the NGIO label metadata."""
        return get_ngio_label_meta(
            group_handler=self._group_handler,
            version=self._version,
            axes_setup=self._axes_setup,
            allow_non_canonical_axes=self._allow_non_canonical_axes,
            strict_canonical_order=self._strict_canonical_order,
        )

    def update_meta(self, ngio_meta: NgioLabelMeta) -> None:
        """Update the NGIO label metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )


def get_ngio_plate_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
) -> NgioPlateMeta:
    """Retrieve the NGIO plate metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.

    Returns:
        NgioPlateMeta: The NGIO plate metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioPlateMeta,
        version=version,
    )


def update_ngio_plate_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: NgioPlateMeta,
) -> None:
    """Update the NGIO plate metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (NgioPlateMeta): The new NGIO plate metadata.

    """
    update_ngio_meta(
        group_handler=group_handler,
        ngio_meta=ngio_meta,
    )


class PlateMetaHandler:
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        version: str | None = None,
    ):
        self._group_handler = group_handler
        self._version = version

        # Validate metadata
        _ = self.get_meta()
        # Store the resolved version
        # self._version = meta.version

    def get_meta(self) -> NgioPlateMeta:
        """Retrieve the NGIO plate metadata."""
        return get_ngio_plate_meta(
            group_handler=self._group_handler,
            version=self._version,
        )

    def update_meta(self, ngio_meta: NgioPlateMeta) -> None:
        """Update the NGIO plate metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )


def get_ngio_well_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
) -> NgioWellMeta:
    """Retrieve the NGIO well metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.

    Returns:
        NgioWellMeta: The NGIO well metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioWellMeta,
        version=version,
    )


def update_ngio_well_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: NgioWellMeta,
) -> None:
    """Update the NGIO well metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (NgioWellMeta): The new NGIO well metadata.

    """
    update_ngio_meta(
        group_handler=group_handler,
        ngio_meta=ngio_meta,
    )


class WellMetaHandler:
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        version: str | None = None,
    ):
        self._group_handler = group_handler
        self._version = version

        # Validate metadata
        _ = self.get_meta()
        # Store the resolved version
        # self._version = meta.version

    def get_meta(self) -> NgioWellMeta:
        """Retrieve the NGIO well metadata."""
        return get_ngio_well_meta(
            group_handler=self._group_handler,
            version=self._version,
        )

    def update_meta(self, ngio_meta: NgioWellMeta) -> None:
        """Update the NGIO well metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )


def get_ngio_labels_group_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
) -> NgioLabelsGroupMeta:
    """Retrieve the NGIO labels group metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.

    Returns:
        NgioLabelsGroupMeta: The NGIO labels group metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioLabelsGroupMeta,
        version=version,
    )


def update_ngio_labels_group_meta(
    group_handler: ZarrGroupHandler,
    ngio_meta: NgioLabelsGroupMeta,
) -> None:
    """Update the NGIO labels group metadata in the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        ngio_meta (NgioLabelsGroupMeta): The new NGIO labels group metadata.

    """
    update_ngio_meta(
        group_handler=group_handler,
        ngio_meta=ngio_meta,
    )


class LabelsGroupMetaHandler:
    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        version: NgffVersions | None = None,
    ):
        self._group_handler = group_handler
        self._version = version

        meta = self.get_meta()
        self._version = meta.version

    def get_meta(self) -> NgioLabelsGroupMeta:
        """Retrieve the NGIO labels group metadata."""
        return get_ngio_labels_group_meta(
            group_handler=self._group_handler,
            version=self._version,
        )

    def update_meta(self, ngio_meta: NgioLabelsGroupMeta) -> None:
        """Update the NGIO labels group metadata."""
        update_ngio_labels_group_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
