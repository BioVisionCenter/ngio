"""Base class for handling OME-NGFF metadata in Zarr groups."""

from typing import Generic, Protocol, TypeVar

from pydantic import ValidationError

from ngio.ome_zarr_meta.ngio_specs import (
    AxesSetup,
    NgioImageMeta,
    NgioLabelMeta,
    NgioPlateMeta,
    NgioWellMeta,
)
from ngio.ome_zarr_meta.v04 import (
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    ngio_to_v04_plate_meta,
    ngio_to_v04_well_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
    v04_to_ngio_plate_meta,
    v04_to_ngio_well_meta,
)
from ngio.ome_zarr_meta.v05 import (
    ngio_to_v05_image_meta,
    ngio_to_v05_label_meta,
    ngio_to_v05_plate_meta,
    ngio_to_v05_well_meta,
    v05_to_ngio_image_meta,
    v05_to_ngio_label_meta,
    v05_to_ngio_plate_meta,
    v05_to_ngio_well_meta,
)
from ngio.utils import (
    NgioValidationError,
    NgioValueError,
    ZarrGroupHandler,
)

ConverterError = ValidationError | Exception | None

###########################################################################
#
# The code below implements a generic class for handling OME-Zarr metadata
# in Zarr groups.
#
###########################################################################


class ImageMetaImporter(Protocol):
    @staticmethod
    def __call__(
        metadata: dict,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> tuple[bool, NgioImageMeta | ConverterError]:
        """Convert the metadata to a NgioImageMeta object.

        Args:
            metadata (dict): The metadata (typically from a Zarr group .attrs).
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.

        Returns:
            tuple[bool, NgioImageMeta | ConverterError]: A tuple with a boolean
                indicating whether the conversion was successful and the
                NgioImageMeta object or an error.

        """
        ...


class ImageMetaExporter(Protocol):
    def __call__(self, metadata: NgioImageMeta) -> dict: ...


class LabelMetaImporter(Protocol):
    @staticmethod
    def __call__(
        metadata: dict,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> tuple[bool, NgioLabelMeta | ConverterError]:
        """Convert the metadata to a NgioLabelMeta object.

        Args:
            metadata (dict): The metadata (typically from a Zarr group .attrs).
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.

        Returns:
            tuple[bool, NgioLabelMeta | ConverterError]: A tuple with a boolean
                indicating whether the conversion was successful and the
                NgioLabelMeta object or an error.

        """
        ...


class LabelMetaExporter(Protocol):
    def __call__(self, metadata: NgioLabelMeta) -> dict: ...


class WellMetaImporter(Protocol):
    def __call__(
        self, metadata: dict
    ) -> tuple[bool, NgioWellMeta | ConverterError]: ...


class WellMetaExporter(Protocol):
    def __call__(self, metadata: NgioWellMeta) -> dict: ...


class PlateMetaImporter(Protocol):
    def __call__(
        self, metadata: dict
    ) -> tuple[bool, NgioPlateMeta | ConverterError]: ...


class PlateMetaExporter(Protocol):
    def __call__(self, metadata: NgioPlateMeta) -> dict: ...


###########################################################################
#
# Image and label metadata handlers
#
###########################################################################

_image_meta = TypeVar("_image_meta", NgioImageMeta, NgioLabelMeta)
_image_meta_importer = TypeVar(
    "_image_meta_importer", ImageMetaImporter, LabelMetaImporter
)
_image_meta_exporter = TypeVar(
    "_image_meta_exporter", ImageMetaExporter, LabelMetaExporter
)


class GenericMetaHandler(
    Generic[_image_meta, _image_meta_importer, _image_meta_exporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_importer: _image_meta_importer,
        meta_exporter: _image_meta_exporter,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        """Initialize the handler.

        Args:
            meta_importer (MetaImporter): The metadata importer.
            meta_exporter (MetaExporter): The metadata exporter.
            group_handler (ZarrGroupHandler): The Zarr group handler.
            axes_setup (AxesSetup, optional): The axes setup.
                This is used to map axes with non-canonical names.
            allow_non_canonical_axes (bool, optional): Whether to allow non-canonical
                axes.
            strict_canonical_order (bool, optional): Whether to enforce a strict
                canonical order.
        """
        self._group_handler = group_handler
        self._meta_importer = meta_importer
        self._meta_exporter = meta_exporter
        self._axes_setup = axes_setup
        self._allow_non_canonical_axes = allow_non_canonical_axes
        self._strict_canonical_order = strict_canonical_order

    def _load_meta(self, return_error: bool = False):
        """Load the metadata from the store."""
        attrs = self._group_handler.load_attrs()
        is_valid, meta_or_error = self._meta_importer(
            metadata=attrs,
            axes_setup=self._axes_setup,
            allow_non_canonical_axes=self._allow_non_canonical_axes,
            strict_canonical_order=self._strict_canonical_order,
        )
        if is_valid:
            return meta_or_error

        if return_error:
            return meta_or_error

        raise NgioValueError(f"Could not load metadata: {meta_or_error}")

    def _write_meta(self, meta) -> None:
        """Write the metadata to the store."""
        _meta = self._meta_exporter(metadata=meta)
        self._group_handler.write_attrs(_meta)

    def write_meta(self, meta: _image_meta) -> None:
        self._write_meta(meta)

    @property
    def meta(self) -> _image_meta:
        """Return the metadata."""
        raise NotImplementedError("This method should be implemented in a subclass.")


class ImageMetaHandler(
    GenericMetaHandler[NgioImageMeta, ImageMetaImporter, ImageMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    @property
    def meta(self) -> NgioImageMeta:
        meta = self._load_meta()
        if isinstance(meta, NgioImageMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def safe_load_meta(self) -> NgioImageMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)


class LabelMetaHandler(
    GenericMetaHandler[NgioLabelMeta, LabelMetaImporter, LabelMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    @property
    def meta(self) -> NgioLabelMeta:
        meta = self._load_meta()
        if isinstance(meta, NgioLabelMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def safe_load_meta(self) -> NgioLabelMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)


###########################################################################
#
# Well and plate metadata handlers
#
###########################################################################

_hcs_meta = TypeVar("_hcs_meta", NgioWellMeta, NgioPlateMeta)
_hcs_meta_importer = TypeVar("_hcs_meta_importer", WellMetaImporter, PlateMetaImporter)
_hcs_meta_exporter = TypeVar("_hcs_meta_exporter", WellMetaExporter, PlateMetaExporter)


class GenericHCSMetaHandler(Generic[_hcs_meta, _hcs_meta_importer, _hcs_meta_exporter]):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    def __init__(
        self,
        meta_importer: _hcs_meta_importer,
        meta_exporter: _hcs_meta_exporter,
        group_handler: ZarrGroupHandler,
    ):
        self._group_handler = group_handler
        self._meta_importer = meta_importer
        self._meta_exporter = meta_exporter

    def _load_meta(self, return_error: bool = False):
        """Load the metadata from the store."""
        attrs = self._group_handler.load_attrs()
        is_valid, meta_or_error = self._meta_importer(metadata=attrs)
        if is_valid:
            return meta_or_error

        if return_error:
            return meta_or_error

        raise NgioValueError(f"Could not load metadata: {meta_or_error}")

    def _write_meta(self, meta) -> None:
        _meta = self._meta_exporter(metadata=meta)
        self._group_handler.write_attrs(_meta)

    def write_meta(self, meta: _hcs_meta) -> None:
        self._write_meta(meta)

    @property
    def meta(self) -> _hcs_meta:
        raise NotImplementedError("This method should be implemented in a subclass.")


class WellMetaHandler(
    GenericHCSMetaHandler[NgioWellMeta, WellMetaImporter, WellMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    @property
    def meta(self) -> NgioWellMeta:
        meta = self._load_meta()
        if isinstance(meta, NgioWellMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def safe_load_meta(self) -> NgioWellMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)


class PlateMetaHandler(
    GenericHCSMetaHandler[NgioPlateMeta, PlateMetaImporter, PlateMetaExporter]
):
    """Generic class for handling OME-Zarr metadata in Zarr groups."""

    @property
    def meta(self) -> NgioPlateMeta:
        meta = self._load_meta()
        if isinstance(meta, NgioPlateMeta):
            return meta
        raise NgioValueError(f"Could not load metadata: {meta}")

    def safe_load_meta(self) -> NgioPlateMeta | ConverterError:
        """Load the metadata from the store."""
        return self._load_meta(return_error=True)


###########################################################################
#
# Metadata importer/exporter registration & builder classes
#
###########################################################################


_meta_exporter = TypeVar(
    "_meta_exporter",
    ImageMetaExporter,
    LabelMetaExporter,
    WellMetaExporter,
    PlateMetaExporter,
)
_meta_importer = TypeVar(
    "_meta_importer",
    ImageMetaImporter,
    LabelMetaImporter,
    WellMetaImporter,
    PlateMetaImporter,
)


class _ImporterExporter(Generic[_meta_importer, _meta_exporter]):
    def __init__(
        self,
        version: str,
        importer: _meta_importer,
        exporter: _meta_exporter,
    ):
        self.importer = importer
        self.exporter = exporter
        self.version = version


ImageImporterExporter = _ImporterExporter[ImageMetaImporter, ImageMetaExporter]
LabelImporterExporter = _ImporterExporter[LabelMetaImporter, LabelMetaExporter]
WellImporterExporter = _ImporterExporter[WellMetaImporter, WellMetaExporter]
PlateImporterExporter = _ImporterExporter[PlateMetaImporter, PlateMetaExporter]

_importer_exporter = TypeVar(
    "_importer_exporter",
    ImageImporterExporter,
    LabelImporterExporter,
    WellImporterExporter,
    PlateImporterExporter,
)
_image_handler = TypeVar("_image_handler", ImageMetaHandler, LabelMetaHandler)
_hcs_handler = TypeVar("_hcs_handler", WellMetaHandler, PlateMetaHandler)


class ImplementedMetaImporterExporter:
    _instance = None
    _image_ie: dict[str, ImageImporterExporter]
    _label_ie: dict[str, LabelImporterExporter]
    _well_ie: dict[str, WellImporterExporter]
    _plate_ie: dict[str, PlateImporterExporter]

    def __new__(cls):
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._image_ie = {}
            cls._label_ie = {}
            cls._well_ie = {}
            cls._plate_ie = {}
        return cls._instance

    def _find_image_handler(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
        _ie_name: str = "_image_ie",
        _handler: type[_image_handler] = ImageMetaHandler,
    ) -> _image_handler:
        """Get an image metadata handler."""
        _errors = {}

        dict_ie = self.__getattribute__(_ie_name)

        for ie in reversed(dict_ie.values()):
            handler = _handler(
                meta_importer=ie.importer,
                meta_exporter=ie.exporter,
                group_handler=group_handler,
                axes_setup=axes_setup,
                allow_non_canonical_axes=allow_non_canonical_axes,
                strict_canonical_order=strict_canonical_order,
            )
            meta = handler.safe_load_meta()
            if isinstance(meta, ValidationError):
                _errors[ie.version] = meta
                continue
            return handler

        raise NgioValidationError(
            f"Could not load metadata from any known version. Errors: {_errors}"
        )

    def find_image_handler(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> ImageMetaHandler:
        """Get an image metadata handler."""
        return self._find_image_handler(
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
            _ie_name="_image_ie",
            _handler=ImageMetaHandler,
        )

    def get_image_meta_handler(
        self,
        group_handler: ZarrGroupHandler,
        version: str,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> ImageMetaHandler:
        """Get an image metadata handler."""
        if version not in self._image_ie:
            raise NgioValueError(f"Image handler for version {version} does not exist.")

        image_ie = self._image_ie[version]
        return ImageMetaHandler(
            meta_importer=image_ie.importer,
            meta_exporter=image_ie.exporter,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

    def _register(
        self,
        version: str,
        importer: _importer_exporter,
        overwrite: bool = False,
        _ie_name: str = "_image_ie",
    ):
        """Register an importer/exporter."""
        ie_dict = self.__getattribute__(_ie_name)
        if version in ie_dict and not overwrite:
            raise NgioValueError(
                f"Importer/exporter for version {version} already exists. "
                "Use 'overwrite=True' to overwrite."
            )
        ie_dict[version] = importer

    def register_image_ie(
        self,
        version: str,
        importer: ImageMetaImporter,
        exporter: ImageMetaExporter,
        overwrite: bool = False,
    ):
        """Register an importer/exporter."""
        importer_exporter = ImageImporterExporter(
            version=version, importer=importer, exporter=exporter
        )
        self._register(
            version=version,
            importer=importer_exporter,
            overwrite=overwrite,
            _ie_name="_image_ie",
        )

    def find_label_handler(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> LabelMetaHandler:
        """Get a label metadata handler."""
        return self._find_image_handler(
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
            _ie_name="_label_ie",
            _handler=LabelMetaHandler,
        )

    def get_label_meta_handler(
        self,
        group_handler: ZarrGroupHandler,
        version: str,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ) -> LabelMetaHandler:
        """Get a label metadata handler."""
        if version not in self._label_ie:
            raise NgioValueError(f"Label handler for version {version} does not exist.")

        label_ie = self._label_ie[version]
        return LabelMetaHandler(
            meta_importer=label_ie.importer,
            meta_exporter=label_ie.exporter,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

    def register_label_ie(
        self,
        version: str,
        importer: LabelMetaImporter,
        exporter: LabelMetaExporter,
        overwrite: bool = False,
    ):
        """Register an importer/exporter."""
        importer_exporter = LabelImporterExporter(
            version=version, importer=importer, exporter=exporter
        )
        self._register(
            version=version,
            importer=importer_exporter,
            overwrite=overwrite,
            _ie_name="_label_ie",
        )

    def _find_hcs_handler(
        self,
        group_handler: ZarrGroupHandler,
        _ie_name: str = "_well_ie",
        _handler: type[_hcs_handler] = WellMetaHandler,
    ) -> _hcs_handler:
        """Get a handler for a HCS metadata."""
        _errors = {}

        dict_ie = self.__getattribute__(_ie_name)

        for ie in reversed(dict_ie.values()):
            handler = _handler(
                meta_importer=ie.importer,
                meta_exporter=ie.exporter,
                group_handler=group_handler,
            )
            meta = handler.safe_load_meta()
            if isinstance(meta, ValidationError):
                _errors[ie.version] = meta
                continue
            return handler

        raise NgioValidationError(
            f"Could not load metadata from any known version. Errors: {_errors}"
        )

    def find_well_handler(
        self,
        group_handler: ZarrGroupHandler,
    ) -> WellMetaHandler:
        """Get a well metadata handler."""
        return self._find_hcs_handler(
            group_handler=group_handler,
            _ie_name="_well_ie",
            _handler=WellMetaHandler,
        )

    def get_well_meta_handler(
        self,
        group_handler: ZarrGroupHandler,
        version: str,
    ) -> WellMetaHandler:
        """Get a well metadata handler."""
        if version not in self._well_ie:
            raise NgioValueError(f"Well handler for version {version} does not exist.")

        well_ie = self._well_ie[version]
        return WellMetaHandler(
            meta_importer=well_ie.importer,
            meta_exporter=well_ie.exporter,
            group_handler=group_handler,
        )

    def register_well_ie(
        self,
        version: str,
        importer: WellMetaImporter,
        exporter: WellMetaExporter,
        overwrite: bool = False,
    ):
        """Register an importer/exporter."""
        importer_exporter = WellImporterExporter(
            version=version, importer=importer, exporter=exporter
        )
        self._register(
            version=version,
            importer=importer_exporter,
            overwrite=overwrite,
            _ie_name="_well_ie",
        )

    def find_plate_handler(
        self,
        group_handler: ZarrGroupHandler,
    ) -> PlateMetaHandler:
        """Get a plate metadata handler."""
        return self._find_hcs_handler(
            group_handler=group_handler,
            _ie_name="_plate_ie",
            _handler=PlateMetaHandler,
        )

    def get_plate_meta_handler(
        self,
        group_handler: ZarrGroupHandler,
        version: str,
    ) -> PlateMetaHandler:
        """Get a plate metadata handler."""
        if version not in self._plate_ie:
            raise NgioValueError(f"Plate handler for version {version} does not exist.")

        plate_ie = self._plate_ie[version]
        return PlateMetaHandler(
            meta_importer=plate_ie.importer,
            meta_exporter=plate_ie.exporter,
            group_handler=group_handler,
        )

    def register_plate_ie(
        self,
        version: str,
        importer: PlateMetaImporter,
        exporter: PlateMetaExporter,
        overwrite: bool = False,
    ):
        """Register an importer/exporter."""
        importer_exporter = PlateImporterExporter(
            version=version, importer=importer, exporter=exporter
        )
        self._register(
            version=version,
            importer=importer_exporter,
            overwrite=overwrite,
            _ie_name="_plate_ie",
        )


###########################################################################
#
# Register metadata importers/exporters
#
###########################################################################


ImplementedMetaImporterExporter().register_image_ie(
    version="0.4",
    importer=v04_to_ngio_image_meta,
    exporter=ngio_to_v04_image_meta,
)

ImplementedMetaImporterExporter().register_label_ie(
    version="0.5",
    importer=v05_to_ngio_label_meta,
    exporter=ngio_to_v05_label_meta,
)

ImplementedMetaImporterExporter().register_label_ie(
    version="0.4",
    importer=v04_to_ngio_label_meta,
    exporter=ngio_to_v04_label_meta,
)
ImplementedMetaImporterExporter().register_image_ie(
    version="0.5",
    importer=v05_to_ngio_image_meta,
    exporter=ngio_to_v05_image_meta,
)

ImplementedMetaImporterExporter().register_well_ie(
    version="0.4", importer=v04_to_ngio_well_meta, exporter=ngio_to_v04_well_meta
)
ImplementedMetaImporterExporter().register_well_ie(
    version="0.5", importer=v05_to_ngio_well_meta, exporter=ngio_to_v05_well_meta
)
ImplementedMetaImporterExporter().register_plate_ie(
    version="0.4", importer=v04_to_ngio_plate_meta, exporter=ngio_to_v04_plate_meta
)
ImplementedMetaImporterExporter().register_plate_ie(
    version="0.5", importer=v05_to_ngio_plate_meta, exporter=ngio_to_v05_plate_meta
)


###########################################################################
#
# Public functions to avoid direct access to the importer/exporter
# registration methods
#
###########################################################################


def find_image_meta_handler(
    group_handler: ZarrGroupHandler,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> ImageMetaHandler:
    """Open an image metadata handler."""
    return ImplementedMetaImporterExporter().find_image_handler(
        group_handler=group_handler,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def get_image_meta_handler(
    group_handler: ZarrGroupHandler,
    version: str,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> ImageMetaHandler:
    """Open an image metadata handler."""
    return ImplementedMetaImporterExporter().get_image_meta_handler(
        group_handler=group_handler,
        version=version,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def find_label_meta_handler(
    group_handler: ZarrGroupHandler,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> LabelMetaHandler:
    """Open a label metadata handler."""
    return ImplementedMetaImporterExporter().find_label_handler(
        group_handler=group_handler,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def get_label_meta_handler(
    group_handler: ZarrGroupHandler,
    version: str,
    axes_setup: AxesSetup | None = None,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> LabelMetaHandler:
    """Open a label metadata handler."""
    return ImplementedMetaImporterExporter().get_label_meta_handler(
        group_handler=group_handler,
        version=version,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )


def find_well_meta_handler(group_handler: ZarrGroupHandler) -> WellMetaHandler:
    """Open a well metadata handler."""
    return ImplementedMetaImporterExporter().find_well_handler(
        group_handler=group_handler,
    )


def get_well_meta_handler(
    group_handler: ZarrGroupHandler,
    version: str,
) -> WellMetaHandler:
    """Open a well metadata handler."""
    return ImplementedMetaImporterExporter().get_well_meta_handler(
        group_handler=group_handler,
        version=version,
    )


def find_plate_meta_handler(group_handler: ZarrGroupHandler) -> PlateMetaHandler:
    """Open a plate metadata handler."""
    return ImplementedMetaImporterExporter().find_plate_handler(
        group_handler=group_handler
    )


def get_plate_meta_handler(
    group_handler: ZarrGroupHandler,
    version: str,
) -> PlateMetaHandler:
    """Open a plate metadata handler."""
    return ImplementedMetaImporterExporter().get_plate_meta_handler(
        group_handler=group_handler,
        version=version,
    )
