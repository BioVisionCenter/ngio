"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Sequence
from typing import Generic, Literal, TypeVar

import dask.array as da
import numpy as np
import zarr

from ngio.common import (
    ArrayLike,
    Dimensions,
    Roi,
    RoiPixels,
    SlicingInputType,
    TransformProtocol,
    build_dask_getter,
    build_dask_setter,
    build_numpy_getter,
    build_numpy_setter,
    build_roi_dask_getter,
    build_roi_dask_setter,
    build_roi_numpy_getter,
    build_roi_numpy_setter,
    consolidate_pyramid,
)
from ngio.ome_zarr_meta import (
    AxesMapper,
    Dataset,
    ImageMetaHandler,
    LabelMetaHandler,
    PixelSize,
)
from ngio.tables import RoiTable
from ngio.utils import NgioFileExistsError, ZarrGroupHandler

_image_handler = TypeVar("_image_handler", ImageMetaHandler, LabelMetaHandler)


class AbstractImage(Generic[_image_handler]):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: _image_handler,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.

        """
        self._path = path
        self._group_handler = group_handler
        self._meta_handler = meta_handler

        self._dataset = self._meta_handler.meta.get_dataset(path=path)
        self._pixel_size = self._dataset.pixel_size

        try:
            self._zarr_array = self._group_handler.get_array(self._dataset.path)
        except NgioFileExistsError as e:
            raise NgioFileExistsError(f"Could not find the dataset at {path}.") from e

        self._dimensions = Dimensions(
            shape=self._zarr_array.shape, axes_mapper=self._dataset.axes_mapper
        )

        self._axes_mapper = self._dataset.axes_mapper

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"Image(path={self.path}, {self.dimensions})"

    @property
    def meta_handler(self) -> _image_handler:
        """Return the metadata."""
        return self._meta_handler

    @property
    def zarr_array(self) -> zarr.Array:
        """Return the Zarr array."""
        return self._zarr_array

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image."""
        return self.zarr_array.shape

    @property
    def dtype(self) -> str:
        """Return the dtype of the image."""
        return str(self.zarr_array.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        """Return the chunks of the image."""
        return self.zarr_array.chunks

    @property
    def dimensions(self) -> Dimensions:
        """Return the dimensions of the image."""
        return self._dimensions

    @property
    def axes_mapper(self) -> AxesMapper:
        """Return the axes mapper of the image."""
        return self._axes_mapper

    @property
    def is_3d(self) -> bool:
        """Return True if the image is 3D."""
        return self.dimensions.is_3d

    @property
    def is_2d(self) -> bool:
        """Return True if the image is 2D."""
        return self.dimensions.is_2d

    @property
    def is_time_series(self) -> bool:
        """Return True if the image is a time series."""
        return self.dimensions.is_time_series

    @property
    def is_2d_time_series(self) -> bool:
        """Return True if the image is a 2D time series."""
        return self.dimensions.is_2d_time_series

    @property
    def is_3d_time_series(self) -> bool:
        """Return True if the image is a 3D time series."""
        return self.dimensions.is_3d_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return True if the image is multichannel."""
        return self.dimensions.is_multi_channels

    @property
    def space_unit(self) -> str | None:
        """Return the space unit of the image."""
        return self.meta_handler.meta.space_unit

    @property
    def time_unit(self) -> str | None:
        """Return the time unit of the image."""
        return self.meta_handler.meta.time_unit

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size of the image."""
        return self._pixel_size

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self._dataset

    @property
    def path(self) -> str:
        """Return the path of the image."""
        return self._dataset.path

    def has_axis(self, axis: str) -> bool:
        """Return True if the image has the given axis."""
        self.axes_mapper.get_index("x")
        return self.dimensions.has_axis(axis)

    def _get_as_numpy(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Get the image as a numpy array.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        numpy_getter = build_numpy_getter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
        )
        return numpy_getter()

    def _get_roi_as_numpy(
        self,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> np.ndarray:
        """Get the image as a numpy array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        numpy_roi_getter = build_roi_numpy_getter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            roi=roi,
            pixel_size=self.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
        )
        return numpy_roi_getter()

    def _get_as_dask(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get the image as a dask array.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.
        """
        dask_getter = build_dask_getter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
        )
        return dask_getter()

    def _get_roi_as_dask(
        self,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> da.Array:
        """Get the image as a dask array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slicing_kwargs: The slices to get the array.
        """
        roi_dask_getter = build_roi_dask_getter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            roi=roi,
            pixel_size=self.pixel_size,
            axes_order=axes_order,
            transforms=transforms,
            slicing_dict=slicing_kwargs,
        )
        return roi_dask_getter()

    def _get_array(
        self,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slicing_kwargs: SlicingInputType,
    ) -> ArrayLike:
        """Get a slice of the image.

        Args:
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return.
                Can be "dask", "numpy".
            **slicing_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        if mode == "numpy":
            return self._get_as_numpy(
                axes_order=axes_order, transforms=transforms, **slicing_kwargs
            )
        elif mode == "dask":
            return self._get_as_dask(
                axes_order=axes_order, transforms=transforms, **slicing_kwargs
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from 'numpy', 'dask', or 'delayed'."
            )

    def _get_roi(
        self,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slice_kwargs: SlicingInputType,
    ) -> ArrayLike:
        """Get a slice of the image.

        Args:
            roi: The region of interest to get the array.
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The mode to return the array.
                Can be "dask", "numpy".
            **slice_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        if mode == "numpy":
            return self._get_roi_as_numpy(
                roi=roi, axes_order=axes_order, transforms=transforms, **slice_kwargs
            )
        elif mode == "dask":
            return self._get_roi_as_dask(
                roi=roi, axes_order=axes_order, transforms=transforms, **slice_kwargs
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from 'numpy', 'dask', or 'delayed'."
            )

    def _set_array(
        self,
        patch: ArrayLike,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set a slice of the image.

        Args:
            patch: The patch to set.
            axes_order: The order of the axes to set the patch.
            transforms: The transforms to apply to the patch.
            **slicing_kwargs: The slices to set the patch.

        """
        if isinstance(patch, np.ndarray):
            numpy_setter = build_numpy_setter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            numpy_setter(patch)

        elif isinstance(patch, da.Array):
            dask_setter = build_dask_setter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            dask_setter(patch)
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Supported types are: "
                "numpy.ndarray, dask.array.Array."
            )

    def _set_roi(
        self,
        roi: Roi | RoiPixels,
        patch: ArrayLike,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        **slicing_kwargs: SlicingInputType,
    ) -> None:
        """Set a slice of the image.

        Args:
            roi: The region of interest to set the patch.
            patch: The patch to set.
            axes_order: The order of the axes to set the patch.
            transforms: The transforms to apply to the patch.
            **slicing_kwargs: The slices to set the patch.

        """
        if isinstance(patch, np.ndarray):
            roi_numpy_setter = build_roi_numpy_setter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                roi=roi,
                pixel_size=self.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            roi_numpy_setter(patch)

        elif isinstance(patch, da.Array):
            roi_dask_setter = build_roi_dask_setter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                roi=roi,
                pixel_size=self.pixel_size,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            roi_dask_setter(patch)
        else:
            raise TypeError(
                f"Unsupported patch type: {type(patch)}. "
                "Supported types are: "
                "numpy.ndarray, dask.array.Array."
            )

    def _consolidate(
        self,
        order: Literal[0, 1, 2] = 1,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the image on disk.

        Args:
            order: The order of the consolidation.
            mode: The mode of the consolidation.
        """
        consolidate_image(image=self, order=order, mode=mode)

    def build_image_roi_table(self, name: str = "image") -> RoiTable:
        """Build the ROI table for an image."""
        return build_image_roi_table(image=self, name=name)


def consolidate_image(
    image: AbstractImage,
    order: Literal[0, 1, 2] = 1,
    mode: Literal["dask", "numpy", "coarsen"] = "dask",
) -> None:
    """Consolidate the image on disk."""
    target_paths = image._meta_handler.meta.paths
    targets = [
        image._group_handler.get_array(path)
        for path in target_paths
        if path != image.path
    ]
    consolidate_pyramid(
        source=image.zarr_array, targets=targets, order=order, mode=mode
    )


def build_image_roi_table(image: AbstractImage, name: str = "image") -> RoiTable:
    """Build the ROI table for an image."""
    dim_x = image.dimensions.get("x")
    dim_y = image.dimensions.get("y")
    assert dim_x is not None and dim_y is not None
    dim_z = image.dimensions.get("z")
    z = None if dim_z is None else 0
    dim_t = image.dimensions.get("t")
    t = None if dim_t is None else 0
    image_roi = RoiPixels(
        name=name,
        x=0,
        y=0,
        z=z,
        t=t,
        x_length=dim_x,
        y_length=dim_y,
        z_length=dim_z,
        t_length=dim_t,
    )
    return RoiTable(rois=[image_roi.to_roi(pixel_size=image.pixel_size)])
