"""Base class for handling OME-NGFF metadata in Zarr groups."""

from collections.abc import Callable
from copy import deepcopy
from typing import Generic, TypeVar

from pydantic import ValidationError

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
_image_encoder_registry: dict[str, Callable] = {
    "0.4": ngio_to_v04_image_meta,
    "0.5": ngio_to_v05_image_meta,
}
_image_decoder_registry: dict[str, Callable] = {
    "0.4": v04_to_ngio_image_meta,
    "0.5": v05_to_ngio_image_meta,
}
_label_encoder_registry: dict[str, Callable] = {
    "0.4": ngio_to_v04_label_meta,
    "0.5": ngio_to_v05_label_meta,
}
_label_decoder_registry: dict[str, Callable] = {
    "0.4": v04_to_ngio_label_meta,
    "0.5": v05_to_ngio_label_meta,
}
_plate_encoder_registry: dict[str, Callable] = {
    "0.4": ngio_to_v04_plate_meta,
    "0.5": ngio_to_v05_plate_meta,
}
_plate_decoder_registry: dict[str, Callable] = {
    "0.4": v04_to_ngio_plate_meta,
    "0.5": v05_to_ngio_plate_meta,
}
_well_encoder_registry: dict[str, Callable] = {
    "0.4": ngio_to_v04_well_meta,
    "0.5": ngio_to_v05_well_meta,
}
_well_decoder_registry: dict[str, Callable] = {
    "0.4": v04_to_ngio_well_meta,
    "0.5": v05_to_ngio_well_meta,
}
_labels_group_encoder_registry: dict[str, Callable] = {
    "0.4": ngio_to_v04_labels_group_meta,
    "0.5": ngio_to_v05_labels_group_meta,
}
_labels_group_decoder_registry: dict[str, Callable] = {
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
    attrs: dict | None = None,
    **kwargs,
) -> _meta_type:
    """Retrieve the NGIO metadata from the Zarr group.

    Args:
        group_handler: The Zarr group handler.
        meta_type: The type of NGIO metadata to retrieve.
        version: Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of reading
            them again. Callers that need the raw attributes anyway pass them
            here so the group is read once rather than twice.
        **kwargs: Additional arguments to pass to the decoder.

    Returns:
        _meta_type: The NGIO metadata.
    """
    registry = _find_decoder_registry(meta_type)
    if version is not None:
        decoder = registry.get(version)
        if decoder is None:
            raise NgioValueError(f"Unsupported NGFF version: {version}")
        # The supplied version is a fast path, not a constraint. A document
        # can legitimately differ from the version handed down — a well
        # rewritten at another NGFF version than its plate — so on a failed
        # decode the rest of the registry is still tried.
        versions_to_try = {version: decoder} | {
            v: d for v, d in registry.items() if v != version
        }
    else:
        versions_to_try = registry

    if attrs is None:
        attrs = group_handler.load_attrs()
    all_errors = []
    for version, decoder in versions_to_try.items():
        try:
            ngio_meta = decoder(attrs, **kwargs)
            return ngio_meta
        except ValidationError as e:
            # Only a failed spec check means "these attrs are not this
            # version". An `NgioError` (bad caller argument, corrupt values)
            # propagates — retrying it against another version would blame
            # the file for it.
            all_errors.append(f"Version {version}: {e}")
    error_message = (
        f"Failed to decode NGIO {meta_type.__name__} metadata:\n"
        + "\n".join(all_errors)
    )
    raise NgioValidationError(error_message)


class _MetaMemo(Generic[_meta_type]):
    """Skips the pydantic decode when the raw attributes have not moved.

    Decoding is roughly 20x the cost of copying the result (~700us against
    ~37us on a four-level image), and the handlers decode on *every* metadata
    access, so anything reading `.dimensions` in a loop pays it per iteration.

    The memo is keyed on the raw attributes, not on time or on a dirty flag:
    `get_meta` still reads the group every call, so a change made by anyone --
    this process or another — invalidates it on its own. That keeps the
    freshness ngio has always had, and only removes repeated work.

    `get` hands out a deep copy and keeps the original. The decoded models are
    mutable and `image.meta` is public, so the usual pattern

        meta = image.meta
        meta.set_channels_meta(...)
        image._meta_handler.update_meta(meta)

    would otherwise edit the memo in place — and a caller who mutates without
    writing back would leave it holding something that is not on disk.

    `generation` moves whenever the decoded value does. It gives objects that
    derive something from the metadata a free way to notice — reading an int
    costs nothing, where re-deriving costs a store round-trip. See
    `AbstractImage.dimensions`.
    """

    __slots__ = ("_snapshot", "generation")

    def __init__(self) -> None:
        # (attrs, meta), stored as one tuple so a concurrent reader sees
        # either the old pair or the new pair, never new attrs with old meta.
        # Plate fan-out threads share one handler, so this is load-bearing.
        self._snapshot: tuple[dict, _meta_type] | None = None
        self.generation: int = 0

    def get(self, attrs: dict, decode: Callable[[], _meta_type]) -> _meta_type:
        """Return the metadata for `attrs`, decoding only on a miss."""
        snapshot = self._snapshot
        if snapshot is not None and snapshot[0] == attrs:
            return deepcopy(snapshot[1])
        meta = decode()
        # `attrs` is not copied: `load_attrs` hands out a private dict, and
        # the memo only ever compares it, never returns it.
        self._snapshot = (attrs, meta)
        # Not atomic; a lost increment between racing threads is harmless as
        # long as the counter moves, which a single racer already guarantees.
        self.generation += 1
        return deepcopy(meta)

    def clear(self) -> None:
        """Drop the memo, so the next `get` decodes again."""
        self._snapshot = None
        self.generation += 1


##################################################
#
# Concrete implementations for NGIO metadata types
#
##################################################


def get_ngio_image_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    axes_setup: AxesSetup | None = None,
    attrs: dict | None = None,
) -> NgioImageMeta:
    """Retrieve the NGIO image metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of
            reading them again.
        axes_setup (AxesSetup | None): Optional axes setup for validation.

    Returns:
        NgioImageMeta: The NGIO image metadata.
    """
    # axes_setup = axes_setup or AxesSetup(x="XX")
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioImageMeta,
        version=version,
        axes_setup=axes_setup,
        attrs=attrs,
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
    ):
        self._group_handler = group_handler
        self._version = version
        self._axes_setup = axes_setup
        self._memo: _MetaMemo[NgioImageMeta] = _MetaMemo()
        # Validate metadata
        meta = self.get_meta()
        # Store the resolved version
        self._version = meta.version

    def invalidate(self) -> None:
        """Drop the decoded metadata, so the next read decodes again.

        Moves `generation`, so anything caching a derived value against it --
        `AbstractImage.dimensions` — is dropped with it. This is what
        `OmeZarrContainer.refresh()` reaches for.
        """
        self._memo.clear()

    @property
    def generation(self) -> int:
        """Moves whenever the decoded metadata does.

        Free to read. Lets a derived value be cached and invalidated without
        anyone having to call back into this handler on every access.
        """
        return self._memo.generation

    def get_meta(self) -> NgioImageMeta:
        """Retrieve the NGIO image metadata."""
        attrs = self._group_handler.load_attrs()
        return self._memo.get(
            attrs,
            lambda: get_ngio_image_meta(
                group_handler=self._group_handler,
                version=self._version,
                axes_setup=self._axes_setup,
                attrs=attrs,
            ),
        )

    def update_meta(self, ngio_meta: NgioImageMeta) -> None:
        """Update the NGIO image metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
        self._memo.clear()


def get_ngio_label_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    axes_setup: AxesSetup | None = None,
    attrs: dict | None = None,
) -> NgioLabelMeta:
    """Retrieve the NGIO label metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of
            reading them again.
        axes_setup (AxesSetup | None): Optional axes setup for validation.

    Returns:
        NgioLabelMeta: The NGIO label metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioLabelMeta,
        version=version,
        axes_setup=axes_setup,
        attrs=attrs,
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
        axes_setup: AxesSetup | None = None,
        version: str | None = None,
    ):
        self._group_handler = group_handler
        self._version = version
        self._axes_setup = axes_setup
        self._memo: _MetaMemo[NgioLabelMeta] = _MetaMemo()

        # Validate metadata
        meta = self.get_meta()
        # Store the resolved version
        self._version = meta.version

    def invalidate(self) -> None:
        """Drop the decoded metadata, so the next read decodes again.

        Moves `generation`, so anything caching a derived value against it --
        `AbstractImage.dimensions` — is dropped with it. This is what
        `OmeZarrContainer.refresh()` reaches for.
        """
        self._memo.clear()

    @property
    def generation(self) -> int:
        """Moves whenever the decoded metadata does.

        Free to read. Lets a derived value be cached and invalidated without
        anyone having to call back into this handler on every access.
        """
        return self._memo.generation

    def get_meta(self) -> NgioLabelMeta:
        """Retrieve the NGIO label metadata."""
        attrs = self._group_handler.load_attrs()
        return self._memo.get(
            attrs,
            lambda: get_ngio_label_meta(
                group_handler=self._group_handler,
                version=self._version,
                axes_setup=self._axes_setup,
                attrs=attrs,
            ),
        )

    def update_meta(self, ngio_meta: NgioLabelMeta) -> None:
        """Update the NGIO label metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
        self._memo.clear()


def get_ngio_plate_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    attrs: dict | None = None,
) -> NgioPlateMeta:
    """Retrieve the NGIO plate metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of
            reading them again.

    Returns:
        NgioPlateMeta: The NGIO plate metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioPlateMeta,
        version=version,
        attrs=attrs,
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
        version: NgffVersions | None = None,
    ):
        self._group_handler = group_handler
        self._version = version
        self._memo: _MetaMemo[NgioPlateMeta] = _MetaMemo()

        # Validate metadata
        meta = self.get_meta()
        # Store the resolved version, so later reads go straight to the right
        # decoder instead of re-walking the (0.4-first) registry and paying a
        # full failed validation on every v0.5 document.
        self._version = meta.version

    @property
    def version(self) -> NgffVersions | None:
        """The NGFF version resolved when this handler was built.

        Free to read, unlike `get_meta().version`, which costs a full metadata
        reload. Callers that only need the version to hand to another handler
        should use this.
        """
        return self._version

    def get_meta(self) -> NgioPlateMeta:
        """Retrieve the NGIO plate metadata."""
        attrs = self._group_handler.load_attrs()
        return self._memo.get(
            attrs,
            lambda: get_ngio_plate_meta(
                group_handler=self._group_handler,
                version=self._version,
                attrs=attrs,
            ),
        )

    def update_meta(self, ngio_meta: NgioPlateMeta) -> None:
        """Update the NGIO plate metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
        self._memo.clear()


def get_ngio_well_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    attrs: dict | None = None,
) -> NgioWellMeta:
    """Retrieve the NGIO well metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of
            reading them again.

    Returns:
        NgioWellMeta: The NGIO well metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioWellMeta,
        version=version,
        attrs=attrs,
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
        self._memo: _MetaMemo[NgioWellMeta] = _MetaMemo()

        if version is None:
            # Read once to resolve the version, and store it so later reads go
            # straight to the right decoder instead of re-walking the
            # (0.4-first) registry and paying a full failed validation on every
            # 0.5 document.
            self._version = self.get_meta().version
        # With a version in hand there is nothing left to resolve, so this read
        # would only be validating early. A plate opening 384 wells pays it 384
        # times for documents it is about to read again anyway, so the check is
        # left to first use — where it raises exactly the same error.

    def get_meta(self) -> NgioWellMeta:
        """Retrieve the NGIO well metadata."""
        attrs = self._group_handler.load_attrs()
        return self._memo.get(
            attrs,
            lambda: get_ngio_well_meta(
                group_handler=self._group_handler,
                version=self._version,
                attrs=attrs,
            ),
        )

    def update_meta(self, ngio_meta: NgioWellMeta) -> None:
        """Update the NGIO well metadata."""
        update_ngio_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
        self._memo.clear()


def get_ngio_labels_group_meta(
    group_handler: ZarrGroupHandler,
    version: str | None = None,
    attrs: dict | None = None,
) -> NgioLabelsGroupMeta:
    """Retrieve the NGIO labels group metadata from the Zarr group.

    Args:
        group_handler (ZarrGroupHandler): The Zarr group handler.
        version (str | None): Optional NGFF version to use for decoding.
        attrs: Already-loaded group attributes, to decode instead of
            reading them again.

    Returns:
        NgioLabelsGroupMeta: The NGIO labels group metadata.
    """
    return get_ngio_meta(
        group_handler=group_handler,
        meta_type=NgioLabelsGroupMeta,
        version=version,
        attrs=attrs,
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
        self._memo: _MetaMemo[NgioLabelsGroupMeta] = _MetaMemo()

        meta = self.get_meta()
        self._version = meta.version

    def get_meta(self) -> NgioLabelsGroupMeta:
        """Retrieve the NGIO labels group metadata."""
        attrs = self._group_handler.load_attrs()
        return self._memo.get(
            attrs,
            lambda: get_ngio_labels_group_meta(
                group_handler=self._group_handler,
                version=self._version,
                attrs=attrs,
            ),
        )

    def update_meta(self, ngio_meta: NgioLabelsGroupMeta) -> None:
        """Update the NGIO labels group metadata."""
        update_ngio_labels_group_meta(
            group_handler=self._group_handler,
            ngio_meta=ngio_meta,
        )
        self._memo.clear()
