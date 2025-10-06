"""A module for handling the Plate Sequence in an OME-Zarr file."""

import asyncio
import warnings
from collections.abc import Sequence
from typing import Literal

from ngio.images import (
    OmeZarrContainer,
    concatenate_image_tables,
    concatenate_image_tables_as,
    concatenate_image_tables_as_async,
    concatenate_image_tables_async,
    list_image_tables,
    list_image_tables_async,
)
from ngio.ome_zarr_meta import (
    ImageInWellPath,
    NgffVersions,
    NgioPlateMeta,
    NgioWellMeta,
    find_plate_meta_handler,
    find_well_meta_handler,
    get_plate_meta_handler,
    get_well_meta_handler,
    path_in_well_validation,
)
from ngio.tables import (
    ConditionTable,
    DefaultTableBackend,
    FeatureTable,
    GenericRoiTable,
    MaskingRoiTable,
    RoiTable,
    Table,
    TableBackend,
    TablesContainer,
    TableType,
    TypedTable,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _default_table_container(handler: ZarrGroupHandler) -> TablesContainer | None:
    """Return a default table container."""
    success, table_handler = handler.safe_derive_handler("tables")
    if success and isinstance(table_handler, ZarrGroupHandler):
        return TablesContainer(table_handler)


# Mock lock class that does nothing
class MockLock:
    """A mock lock class that does nothing."""

    def __enter__(self):
        """Enter the lock."""
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the lock."""
        pass


class OmeZarrWell:
    """A class to handle the Well Sequence in an OME-Zarr file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler.

        Args:
            group_handler: The Zarr group handler that contains the Well.
        """
        self._group_handler = group_handler
        self._meta_handler = find_well_meta_handler(group_handler)

    def __repr__(self) -> str:
        """Return a string representation of the well."""
        return f"Well(#images: {len(self.paths())})"

    @property
    def meta_handler(self):
        """Return the metadata handler."""
        return self._meta_handler

    @property
    def meta(self):
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def acquisition_ids(self) -> list[int]:
        """Return the acquisitions ids in the well."""
        return self.meta.acquisition_ids

    def paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        return self.meta.paths(acquisition)

    def get_image_store(self, image_path: str) -> StoreOrGroup:
        """Get the image store from the well.

        Args:
            image_path (str): The path of the image.
        """
        return self._group_handler.get_group(image_path, create_mode=True)

    def get_image_acquisition_id(self, image_path: str) -> int | None:
        """Get the acquisition id of an image in the well.

        Args:
            image_path (str): The path of the image.

        Returns:
            int | None: The acquisition id of the image.
        """
        return self.meta.get_image_acquisition_id(image_path=image_path)

    def get_image(self, image_path: str) -> OmeZarrContainer:
        """Get an image from the well.

        Args:
            image_path (str): The path of the image.

        Returns:
            OmeZarrContainer: The image.
        """
        handler = self._group_handler.derive_handler(image_path)
        return OmeZarrContainer(handler)

    def _add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
        atomic: bool = False,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr well."""
        image_path = path_in_well_validation(path=image_path)

        if atomic:
            well_lock = self._group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            meta = self.meta.add_image(
                path=image_path, acquisition=acquisition_id, strict=strict
            )
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

        return self._group_handler.get_group(image_path, create_mode=True)

    def atomic_add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
    ) -> StoreOrGroup:
        """Parallel safe version of add_image."""
        return self._add_image(
            image_path=image_path,
            acquisition_id=acquisition_id,
            atomic=True,
            strict=strict,
        )

    def add_image(
        self,
        image_path: str,
        acquisition_id: int | None = None,
        strict: bool = True,
    ) -> StoreOrGroup:
        """Add an image to an ome-zarr well.

        Args:
            image_path (str): The path of the image.
            acquisition_id (int | None): The acquisition id to filter the images.
            strict (bool): Whether to check if the acquisition id is already exists
                in the well. Defaults to True. If False this might lead to
                acquisition in a well that does not exist at the plate level.
        """
        return self._add_image(
            image_path=image_path,
            acquisition_id=acquisition_id,
            atomic=False,
            strict=strict,
        )


def _build_extras(paths: Sequence[str]) -> list[dict[str, str]]:
    """Build the extras for the images.

    Args:
        paths (Sequence[str]): The paths of the images.

    Returns:
        list[dict[str, str]]: The extras for the images.
    """
    extras = []
    for path in paths:
        row, column, path_in_well = path.split("/")
        extras.append(
            {
                "row": row,
                "column": column,
                "path_in_well": path_in_well,
            }
        )
    return extras


class OmeZarrPlate:
    """A class to handle the Plate Sequence in an OME-Zarr file."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        table_container: TablesContainer | None = None,
    ) -> None:
        """Initialize the LabelGroupHandler.

        Args:
            group_handler: The Zarr group handler that contains the Plate.
            table_container: The tables container that contains plate level tables.
        """
        self._group_handler = group_handler
        self._meta_handler = find_plate_meta_handler(group_handler)
        self._tables_container = table_container

    def __repr__(self) -> str:
        """Return a string representation of the plate."""
        return f"Plate([rows x columns] ({len(self.rows)} x {len(self.columns)}))"

    @property
    def meta_handler(self):
        """Return the metadata handler."""
        return self._meta_handler

    @property
    def meta(self):
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def columns(self) -> list[str]:
        """Return the number of columns in the plate."""
        return self.meta.columns

    @property
    def rows(self) -> list[str]:
        """Return the number of rows in the plate."""
        return self.meta.rows

    @property
    def acquisitions_names(self) -> list[str | None]:
        """Return the acquisitions in the plate."""
        return self.meta.acquisitions_names

    @property
    def acquisition_ids(self) -> list[int]:
        """Return the acquisitions ids in the plate."""
        return self.meta.acquisition_ids

    def _well_path(self, row: str, column: int | str) -> str:
        """Return the well path in the plate."""
        return self.meta.get_well_path(row=row, column=column)

    def _image_path(self, row: str, column: int | str, path: str) -> str:
        """Return the image path in the plate."""
        well = self.get_well(row, column)
        if path not in well.paths():
            raise ValueError(f"Image {path} does not exist in well {row}{column}")
        return f"{self._well_path(row, column)}/{path}"

    def wells_paths(self) -> list[str]:
        """Return the wells paths in the plate."""
        return self.meta.wells_paths

    async def images_paths_async(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the plate asynchronously.

        If acquisition is None, return all images paths in the plate.
        Else, return the images paths in the plate for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        wells = await self.get_wells_async()
        paths = []
        for well_path, well in wells.items():
            for img_path in well.paths(acquisition):
                paths.append(f"{well_path}/{img_path}")
        return paths

    def images_paths(self, acquisition: int | None = None) -> list[str]:
        """Return the images paths in the plate.

        If acquisition is None, return all images paths in the plate.
        Else, return the images paths in the plate for the given acquisition.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
        """
        wells = self.get_wells()
        images = []
        for well_path, well in wells.items():
            for img_path in well.paths(acquisition):
                images.append(f"{well_path}/{img_path}")
        return images

    def well_images_paths(
        self, row: str, column: int | str, acquisition: int | None = None
    ) -> list[str]:
        """Return the images paths in a well.

        If acquisition is None, return all images paths in the well.
        Else, return the images paths in the well for the given acquisition.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            acquisition (int | None): The acquisition id to filter the images.
        """
        images = []
        well = self.get_well(row=row, column=column)
        for path in well.paths(acquisition):
            images.append(self._image_path(row=row, column=column, path=path))
        return images

    def get_image_acquisition_id(
        self, row: str, column: int | str, image_path: str
    ) -> int | None:
        """Get the acquisition id of an image in a well.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            image_path (str): The path of the image.

        Returns:
            int | None: The acquisition id of the image.
        """
        well = self.get_well(row=row, column=column)
        return well.get_image_acquisition_id(image_path=image_path)

    def get_well(self, row: str, column: int | str) -> OmeZarrWell:
        """Get a well from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.

        Returns:
            OmeZarrWell: The well.
        """
        well_path = self._well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)
        return OmeZarrWell(group_handler)

    async def get_wells_async(self) -> dict[str, OmeZarrWell]:
        """Get all wells in the plate asynchronously.

        This method processes wells in parallel for improved performance
        when working with a large number of wells.

        Returns:
            dict[str, OmeZarrWell]: A dictionary of wells, where the key is the well
                path and the value is the well object.
        """
        wells = self._group_handler.get_from_cache("wells")
        if wells is not None:
            assert isinstance(wells, dict)
            return wells

        def process_well(well_path):
            group_handler = self._group_handler.derive_handler(well_path)
            well = OmeZarrWell(group_handler)
            return well_path, well

        wells, tasks = {}, []
        for well_path in self.wells_paths():
            task = asyncio.to_thread(process_well, well_path)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        for well_path, well in results:
            wells[well_path] = well

        self._group_handler.add_to_cache("wells", wells)
        return wells

    def get_wells(self) -> dict[str, OmeZarrWell]:
        """Get all wells in the plate.

        Returns:
            dict[str, OmeZarrWell]: A dictionary of wells, where the key is the well
                path and the value is the well object.
        """
        wells = self._group_handler.get_from_cache("wells")
        if wells is not None:
            assert isinstance(wells, dict)
            return wells

        def process_well(well_path):
            group_handler = self._group_handler.derive_handler(well_path)
            well = OmeZarrWell(group_handler)
            return well_path, well

        wells = {}
        for well_path in self.wells_paths():
            _, well = process_well(well_path)
            wells[well_path] = well

        self._group_handler.add_to_cache("wells", wells)
        return wells

    async def get_images_async(
        self, acquisition: int | None = None
    ) -> dict[str, OmeZarrContainer]:
        """Get all images in the plate asynchronously.

        This method processes images in parallel for improved performance
        when working with a large number of images.

        Args:
            acquisition: The acquisition id to filter the images.

        Returns:
            dict[str, OmeZarrContainer]: A dictionary of images, where the key is the
                image path and the value is the image object.
        """
        images = self._group_handler.get_from_cache("images")
        if images is not None:
            assert isinstance(images, dict)
            return images

        paths = await self.images_paths_async(acquisition=acquisition)

        def process_image(image_path):
            """Process a single image and return the image path and image object."""
            img_group_handler = self._group_handler.derive_handler(image_path)
            image = OmeZarrContainer(img_group_handler)
            return image_path, image

        images, tasks = {}, []
        for image_path in paths:
            task = asyncio.to_thread(process_image, image_path)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for image_path, image in results:
            images[image_path] = image

        self._group_handler.add_to_cache("images", images)
        return images

    def get_images(self, acquisition: int | None = None) -> dict[str, OmeZarrContainer]:
        """Get all images in the plate.

        Args:
            acquisition: The acquisition id to filter the images.
        """
        images = self._group_handler.get_from_cache("images")
        if images is not None:
            assert isinstance(images, dict)
            return images
        paths = self.images_paths(acquisition=acquisition)

        def process_image(image_path):
            """Process a single image and return the image path and image object."""
            img_group_handler = self._group_handler.derive_handler(image_path)
            image = OmeZarrContainer(img_group_handler)
            return image_path, image

        images = {}
        for image_path in paths:
            _, image = process_image(image_path)
            images[image_path] = image

        self._group_handler.add_to_cache("images", images)
        return images

    def get_image(
        self, row: str, column: int | str, image_path: str
    ) -> OmeZarrContainer:
        """Get an image from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            image_path (str): The path of the image.

        Returns:
            OmeZarrContainer: The image.
        """
        image_path = self._image_path(row=row, column=column, path=image_path)
        group_handler = self._group_handler.derive_handler(image_path)
        return OmeZarrContainer(group_handler)

    def get_image_store(
        self, row: str, column: int | str, image_path: str
    ) -> StoreOrGroup:
        """Get the image store from the plate.

        Args:
            row (str): The row of the well.
            column (int | str): The column of the well.
            image_path (str): The path of the image.
        """
        well = self.get_well(row=row, column=column)
        return well.get_image_store(image_path=image_path)

    def get_well_images(
        self, row: str, column: str | int, acquisition: int | None = None
    ) -> dict[str, OmeZarrContainer]:
        """Get all images in a well.

        Args:
            row: The row of the well.
            column: The column of the well.
            acquisition: The acquisition id to filter the images.
        """
        images = {}
        for image_paths in self.well_images_paths(
            row=row, column=column, acquisition=acquisition
        ):
            group_handler = self._group_handler.derive_handler(image_paths)
            images[image_paths] = OmeZarrContainer(group_handler)
        return images

    def _add_image(
        self,
        row: str,
        column: int | str,
        image_path: str | None = None,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
        atomic: bool = False,
    ) -> str:
        """Add an image to an ome-zarr plate."""
        if image_path is not None:
            image_path = path_in_well_validation(path=image_path)

        if atomic:
            plate_lock = self._group_handler.lock
        else:
            plate_lock = MockLock()

        with plate_lock:
            meta = self.meta
            meta = meta.add_well(row=row, column=column)
            if acquisition_id is not None:
                meta = meta.add_acquisition(
                    acquisition_id=acquisition_id, acquisition_name=acquisition_name
                )
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

        well_path = self.meta.get_well_path(row=row, column=column)
        group_handler = self._group_handler.derive_handler(well_path)

        if atomic:
            well_lock = group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            attrs = group_handler.load_attrs()
            if len(attrs) == 0:
                # Initialize the well metadata
                # if the group is empty
                well_meta = NgioWellMeta.default_init()
                version = self.meta.plate.version
                version = version if version is not None else "0.4"
                meta_handler = get_well_meta_handler(group_handler, version=version)
            else:
                meta_handler = find_well_meta_handler(group_handler)
                well_meta = meta_handler.meta

            group_handler = self._group_handler.derive_handler(well_path)

            if image_path is not None:
                well_meta = well_meta.add_image(
                    path=image_path, acquisition=acquisition_id, strict=False
                )
            meta_handler.write_meta(well_meta)
            meta_handler._group_handler.clean_cache()

        if image_path is not None:
            return f"{well_path}/{image_path}"
        return well_path

    def atomic_add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> str:
        """Parallel safe version of add_image."""
        if image_path is None:
            raise ValueError(
                "Image path cannot be None for atomic add_image. "
                "If your intent is to add a well, use add_well instead."
            )
        path = self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=True,
        )
        return path

    def add_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        acquisition_id: int | None = None,
        acquisition_name: str | None = None,
    ) -> str:
        """Add an image to an ome-zarr plate."""
        if image_path is None:
            raise ValueError(
                "Image path cannot be None for atomic add_image. "
                "If your intent is to add a well, use add_well instead."
            )
        path = self._add_image(
            row=row,
            column=column,
            image_path=image_path,
            acquisition_id=acquisition_id,
            acquisition_name=acquisition_name,
            atomic=False,
        )
        return path

    def add_well(
        self,
        row: str,
        column: int | str,
    ) -> OmeZarrWell:
        """Add a well to an ome-zarr plate."""
        _ = self._add_image(
            row=row,
            column=column,
            image_path=None,
            acquisition_id=None,
            acquisition_name=None,
            atomic=False,
        )
        return self.get_well(row=row, column=column)

    def add_column(
        self,
        column: int | str,
    ) -> "OmeZarrPlate":
        """Add a column to an ome-zarr plate."""
        meta, _ = self.meta.add_column(column)
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

    def add_row(
        self,
        row: str,
    ) -> "OmeZarrPlate":
        """Add a row to an ome-zarr plate."""
        meta, _ = self.meta.add_row(row)
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

    def add_acquisition(
        self,
        acquisition_id: int,
        acquisition_name: str,
    ) -> "OmeZarrPlate":
        """Add an acquisition to an ome-zarr plate.

        Be aware that this is not a parallel safe operation.

        Args:
            acquisition_id (int): The acquisition id.
            acquisition_name (str): The acquisition name.
        """
        meta = self.meta.add_acquisition(
            acquisition_id=acquisition_id, acquisition_name=acquisition_name
        )
        self.meta_handler.write_meta(meta)
        self.meta_handler._group_handler.clean_cache()
        return self

    def _remove_well(
        self,
        row: str,
        column: int | str,
        atomic: bool = False,
    ):
        """Remove a well from an ome-zarr plate."""
        if atomic:
            plate_lock = self._group_handler.lock
        else:
            plate_lock = MockLock()

        with plate_lock:
            meta = self.meta
            meta = meta.remove_well(row, column)
            self.meta_handler.write_meta(meta)
            self.meta_handler._group_handler.clean_cache()

    def _remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
        atomic: bool = False,
    ):
        """Remove an image from an ome-zarr plate."""
        well = self.get_well(row, column)

        if atomic:
            well_lock = well.meta_handler._group_handler.lock
        else:
            well_lock = MockLock()

        with well_lock:
            well_meta = well.meta
            well_meta = well_meta.remove_image(path=image_path)
            well.meta_handler.write_meta(well_meta)
            well.meta_handler._group_handler.clean_cache()
            if len(well_meta.paths()) == 0:
                self._remove_well(row, column, atomic=atomic)

    def atomic_remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
    ):
        """Parallel safe version of remove_image."""
        return self._remove_image(
            row=row,
            column=column,
            image_path=image_path,
            atomic=True,
        )

    def remove_image(
        self,
        row: str,
        column: int | str,
        image_path: str,
    ):
        """Remove an image from an ome-zarr plate."""
        return self._remove_image(
            row=row,
            column=column,
            image_path=image_path,
            atomic=False,
        )

    def derive_plate(
        self,
        store: StoreOrGroup,
        plate_name: str | None = None,
        version: NgffVersions = "0.4",
        keep_acquisitions: bool = False,
        cache: bool = False,
        overwrite: bool = False,
        parallel_safe: bool = True,
    ) -> "OmeZarrPlate":
        """Derive a new OME-Zarr plate from an existing one.

        Args:
            store (StoreOrGroup): The Zarr store or group that stores the plate.
            plate_name (str | None): The name of the new plate.
            version (NgffVersion): The version of the new plate.
            keep_acquisitions (bool): Whether to keep the acquisitions in the new plate.
            cache (bool): Whether to use a cache for the zarr group metadata.
            overwrite (bool): Whether to overwrite the existing plate.
            parallel_safe (bool): Whether the group handler is parallel safe.
        """
        return derive_ome_zarr_plate(
            ome_zarr_plate=self,
            store=store,
            plate_name=plate_name,
            version=version,
            keep_acquisitions=keep_acquisitions,
            cache=cache,
            overwrite=overwrite,
            parallel_safe=parallel_safe,
        )

    def _get_tables_container(self) -> TablesContainer | None:
        """Return the tables container."""
        if self._tables_container is None:
            _tables_container = _default_table_container(self._group_handler)
            if _tables_container is None:
                return None
            self._tables_container = _tables_container
        return self._tables_container

    @property
    def tables_container(self) -> TablesContainer:
        """Return the tables container."""
        _table_container = self._get_tables_container()
        if _table_container is None:
            raise NgioValueError(
                "No tables container found. Please add a tables container to the plate."
            )
        return _table_container

    def list_tables(self, filter_types: TypedTable | str | None = None) -> list[str]:
        """List all tables in the image."""
        return self.tables_container.list(filter_types=filter_types)

    def list_roi_tables(self) -> list[str]:
        """List all ROI tables in the image."""
        masking_roi = self.tables_container.list(
            filter_types="masking_roi_table",
        )
        roi = self.tables_container.list(
            filter_types="roi_table",
        )
        return masking_roi + roi

    def get_roi_table(self, name: str) -> RoiTable:
        """Get a ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, RoiTable):
            raise NgioValueError(f"Table {name} is not a ROI table. Got {type(table)}")
        return table

    def get_masking_roi_table(self, name: str) -> MaskingRoiTable:
        """Get a masking ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, MaskingRoiTable):
            raise NgioValueError(
                f"Table {name} is not a masking ROI table. Got {type(table)}"
            )
        return table

    def get_feature_table(self, name: str) -> FeatureTable:
        """Get a feature table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, FeatureTable):
            raise NgioValueError(
                f"Table {name} is not a feature table. Got {type(table)}"
            )
        return table

    def get_generic_roi_table(self, name: str) -> GenericRoiTable:
        """Get a generic ROI table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, GenericRoiTable):
            raise NgioValueError(
                f"Table {name} is not a generic ROI table. Got {type(table)}"
            )
        return table

    def get_condition_table(self, name: str) -> ConditionTable:
        """Get a condition table from the image.

        Args:
            name (str): The name of the table.
        """
        table = self.tables_container.get(name=name, strict=True)
        if not isinstance(table, ConditionTable):
            raise NgioValueError(
                f"Table {name} is not a condition table. Got {type(table)}"
            )
        return table

    def get_table(self, name: str, check_type: TypedTable | None = None) -> Table:
        """Get a table from the image.

        Args:
            name (str): The name of the table.
            check_type (TypedTable | None): Deprecated. Please use
                'get_table_as' instead, or one of the type specific
                get_*table() methods.

        """
        if check_type is not None:
            warnings.warn(
                "The 'check_type' argument is deprecated, and will be removed in "
                "ngio=0.3. Use 'get_table_as' instead or one of the "
                "type specific get_*table() methods.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.tables_container.get(name=name, strict=False)

    def get_table_as(
        self,
        name: str,
        table_cls: type[TableType],
        backend: TableBackend | None = None,
    ) -> TableType:
        """Get a table from the image as a specific type.

        Args:
            name (str): The name of the table.
            table_cls (type[TableType]): The type of the table.
            backend (TableBackend | None): The backend to use. If None,
                the default backend is used.
        """
        return self.tables_container.get_as(
            name=name,
            table_cls=table_cls,
            backend=backend,
        )

    def add_table(
        self,
        name: str,
        table: Table,
        backend: TableBackend = DefaultTableBackend,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the image."""
        self.tables_container.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def list_image_tables(
        self,
        acquisition: int | None = None,
        filter_types: str | None = None,
        mode: Literal["common", "all"] = "common",
    ) -> list[str]:
        """List all image tables in the image.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
            filter_types (str | None): The type of tables to filter. If None,
                return all tables. Defaults to None.
            mode (Literal["common", "all"]): The mode to use for listing the tables.
                If 'common', return only common tables between all images.
                If 'all', return all tables. Defaults to 'common'.
        """
        images = tuple(self.get_images(acquisition=acquisition).values())
        return list_image_tables(
            images=images,
            filter_types=filter_types,
            mode=mode,
        )

    async def list_image_tables_async(
        self,
        acquisition: int | None = None,
        filter_types: str | None = None,
        mode: Literal["common", "all"] = "common",
    ) -> list[str]:
        """List all image tables in the image asynchronously.

        Args:
            acquisition (int | None): The acquisition id to filter the images.
            filter_types (str | None): The type of tables to filter. If None,
                return all tables. Defaults to None.
            mode (Literal["common", "all"]): The mode to use for listing the tables.
                If 'common', return only common tables between all images.
                If 'all', return all tables. Defaults to 'common'.
        """
        images = await self.get_images_async(acquisition=acquisition)
        images = tuple(images.values())
        return await list_image_tables_async(
            images=images,
            filter_types=filter_types,
            mode=mode,
        )

    def concatenate_image_tables(
        self,
        name: str,
        acquisition: int | None = None,
        strict: bool = True,
        index_key: str | None = None,
        mode: Literal["eager", "lazy"] = "eager",
    ) -> Table:
        """Concatenate tables from all images in the plate.

        Args:
            name: The name of the table to concatenate.
            index_key: The key to use for the index of the concatenated table.
            acquisition: The acquisition id to filter the images.
            strict: If True, raise an error if the table is not found in the image.
            index_key: If a string is provided, a new index column will be created
                new_index_pattern = {row}_{column}_{path_in_well}_{label}
            mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
                if 'eager', the table will be loaded into memory.
                if 'lazy', the table will be loaded as a lazy frame.
        """
        images = self.get_images(acquisition=acquisition)
        extras = _build_extras(tuple(images.keys()))
        return concatenate_image_tables(
            images=tuple(images.values()),
            extras=extras,
            name=name,
            index_key=index_key,
            strict=strict,
            mode=mode,
        )

    def concatenate_image_tables_as(
        self,
        name: str,
        table_cls: type[TableType],
        acquisition: int | None = None,
        index_key: str | None = None,
        strict: bool = True,
        mode: Literal["eager", "lazy"] = "eager",
    ) -> TableType:
        """Concatenate tables from all images in the plate as a specific type.

        Args:
            name: The name of the table to concatenate.
            table_cls: The type of the table to concatenate.
            index_key: The key to use for the index of the concatenated table.
            acquisition: The acquisition id to filter the images.
            index_key: If a string is provided, a new index column will be created
                new_index_pattern = {row}_{column}_{path_in_well}_{label}
            strict: If True, raise an error if the table is not found in the image.
            mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
                if 'eager', the table will be loaded into memory.
                if 'lazy', the table will be loaded as a lazy frame.
        """
        images = self.get_images(acquisition=acquisition)
        extras = _build_extras(tuple(images.keys()))
        return concatenate_image_tables_as(
            images=tuple(images.values()),
            extras=extras,
            name=name,
            table_cls=table_cls,
            index_key=index_key,
            strict=strict,
            mode=mode,
        )

    async def concatenate_image_tables_async(
        self,
        name: str,
        acquisition: int | None = None,
        index_key: str | None = None,
        strict: bool = True,
        mode: Literal["eager", "lazy"] = "eager",
    ) -> Table:
        """Concatenate tables from all images in the plate asynchronously.

        Args:
            name: The name of the table to concatenate.
            index_key: The key to use for the index of the concatenated table.
            acquisition: The acquisition id to filter the images.
            index_key: If a string is provided, a new index column will be created
                new_index_pattern = {row}_{column}_{path_in_well}_{label}
            strict: If True, raise an error if the table is not found in the image.
            mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
                if 'eager', the table will be loaded into memory.
                if 'lazy', the table will be loaded as a lazy frame.
        """
        images = await self.get_images_async(acquisition=acquisition)
        extras = _build_extras(tuple(images.keys()))
        return await concatenate_image_tables_async(
            images=tuple(images.values()),
            extras=extras,
            name=name,
            index_key=index_key,
            strict=strict,
            mode=mode,
        )

    async def concatenate_image_tables_as_async(
        self,
        name: str,
        table_cls: type[TableType],
        acquisition: int | None = None,
        index_key: str | None = None,
        strict: bool = True,
        mode: Literal["eager", "lazy"] = "eager",
    ) -> TableType:
        """Concatenate tables from all images in the plate as a specific type.

        Args:
            name: The name of the table to concatenate.
            table_cls: The type of the table to concatenate.
            index_key: The key to use for the index of the concatenated table.
            acquisition: The acquisition id to filter the images.
            index_key: If a string is provided, a new index column will be created
                new_index_pattern = {row}_{column}_{path_in_well}_{label}
            strict: If True, raise an error if the table is not found in the image.
            mode: The mode to use for concatenation. Can be 'eager' or 'lazy'.
                if 'eager', the table will be loaded into memory.
                if 'lazy', the table will be loaded as a lazy frame.
        """
        images = await self.get_images_async(acquisition=acquisition)
        extras = _build_extras(tuple(images.keys()))
        return await concatenate_image_tables_as_async(
            images=tuple(images.values()),
            extras=extras,
            name=name,
            table_cls=table_cls,
            index_key=index_key,
            strict=strict,
            mode=mode,
        )


def open_ome_zarr_plate(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Open an OME-Zarr plate.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The
            access mode for the image. Defaults to "r+".
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return OmeZarrPlate(group_handler)


def _create_empty_plate_from_meta(
    store: StoreOrGroup,
    meta: NgioPlateMeta,
    version: NgffVersions = "0.4",
    overwrite: bool = False,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr plate from metadata."""
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(
        store=store, cache=True, mode=mode, parallel_safe=False
    )
    meta_handler = get_plate_meta_handler(group_handler, version=version)
    meta_handler.write_meta(meta)
    return group_handler


def create_empty_plate(
    store: StoreOrGroup,
    name: str,
    images: list[ImageInWellPath] | None = None,
    version: NgffVersions = "0.4",
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Initialize and create an empty OME-Zarr plate."""
    plate_meta = NgioPlateMeta.default_init(
        name=name,
        version=version,
    )
    group_handler = _create_empty_plate_from_meta(
        store=store,
        meta=plate_meta,
        version=version,
        overwrite=overwrite,
    )

    if images is not None:
        plate = OmeZarrPlate(group_handler)
        for image in images:
            plate.add_image(
                row=image.row,
                column=image.column,
                image_path=image.path,
                acquisition_id=image.acquisition_id,
                acquisition_name=image.acquisition_name,
            )
    return open_ome_zarr_plate(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )


def derive_ome_zarr_plate(
    ome_zarr_plate: OmeZarrPlate,
    store: StoreOrGroup,
    plate_name: str | None = None,
    version: NgffVersions = "0.4",
    keep_acquisitions: bool = False,
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrPlate:
    """Derive a new OME-Zarr plate from an existing one.

    Args:
        ome_zarr_plate (OmeZarrPlate): The existing OME-Zarr plate.
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        plate_name (str | None): The name of the new plate.
        version (NgffVersion): The version of the new plate.
        keep_acquisitions (bool): Whether to keep the acquisitions in the new plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        overwrite (bool): Whether to overwrite the existing plate.
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    if plate_name is None:
        plate_name = ome_zarr_plate.meta.plate.name

    new_meta = ome_zarr_plate.meta.derive(
        name=plate_name,
        version=version,
        keep_acquisitions=keep_acquisitions,
    )
    _ = _create_empty_plate_from_meta(
        store=store,
        meta=new_meta,
        overwrite=overwrite,
        version=version,
    )
    return open_ome_zarr_plate(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )


def open_ome_zarr_well(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    parallel_safe: bool = True,
) -> OmeZarrWell:
    """Open an OME-Zarr well.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the plate.
        cache (bool): Whether to use a cache for the zarr group metadata.
        mode (AccessModeLiteral): The access mode for the image. Defaults to "r+".
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=cache, mode=mode, parallel_safe=parallel_safe
    )
    return OmeZarrWell(group_handler)


def create_empty_well(
    store: StoreOrGroup,
    version: NgffVersions = "0.4",
    cache: bool = False,
    overwrite: bool = False,
    parallel_safe: bool = True,
) -> OmeZarrWell:
    """Create an empty OME-Zarr well.

    Args:
        store (StoreOrGroup): The Zarr store or group that stores the well.
        version (NgffVersion): The version of the new well.
        cache (bool): Whether to use a cache for the zarr group metadata.
        overwrite (bool): Whether to overwrite the existing well.
        parallel_safe (bool): Whether the group handler is parallel safe.
    """
    group_handler = ZarrGroupHandler(
        store=store, cache=True, mode="w" if overwrite else "w-", parallel_safe=False
    )
    meta_handler = get_well_meta_handler(group_handler, version=version)
    meta = NgioWellMeta.default_init()
    meta_handler.write_meta(meta)

    return open_ome_zarr_well(
        store=store,
        cache=cache,
        mode="r+",
        parallel_safe=parallel_safe,
    )
