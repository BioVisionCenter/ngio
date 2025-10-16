"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Sequence
from typing import Generic, Literal, TypeVar

import dask.array as da
import numpy as np
import zarr

from ngio.common import (
    Dimensions,
    InterpolationOrder,
    Roi,
    RoiPixels,
    consolidate_pyramid,
)
from ngio.io_pipes import (
    DaskGetter,
    DaskRoiGetter,
    DaskRoiSetter,
    DaskSetter,
    NumpyGetter,
    NumpyRoiGetter,
    NumpyRoiSetter,
    NumpySetter,
    SlicingInputType,
    TransformProtocol,
)
from ngio.ome_zarr_meta import (
    AxesHandler,
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

        try:
            self._zarr_array = self._group_handler.get_array(self._path)
        except NgioFileExistsError as e:
            raise NgioFileExistsError(f"Could not find the dataset at {path}.") from e

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"Image(path={self.path}, {self.dimensions})"

    @property
    def path(self) -> str:
        """Return the path of the image."""
        return self._path

    @property
    def meta_handler(self) -> _image_handler:
        """Return the metadata."""
        return self._meta_handler

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self.meta_handler.meta.get_dataset(path=self.path)

    @property
    def dimensions(self) -> Dimensions:
        """Return the dimensions of the image."""
        return Dimensions(
            shape=self.zarr_array.shape,
            chunks=self.zarr_array.chunks,
            dataset=self.dataset,
        )

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel size of the image."""
        return self.dataset.pixel_size

    @property
    def axes_handler(self) -> AxesHandler:
        """Return the axes handler of the image."""
        return self.dataset.axes_handler

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the axes of the image."""
        return self.dimensions.axes

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
        return self.axes_handler.space_unit

    @property
    def time_unit(self) -> str | None:
        """Return the time unit of the image."""
        return self.axes_handler.time_unit

    def has_axis(self, axis: str) -> bool:
        """Return True if the image has the given axis."""
        return self.axes_handler.has_axis(axis)

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
        numpy_getter = NumpyGetter(
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
        numpy_roi_getter = NumpyRoiGetter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            roi=roi,
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
        dask_getter = DaskGetter(
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
        roi_dask_getter = DaskRoiGetter(
            zarr_array=self.zarr_array,
            dimensions=self.dimensions,
            roi=roi,
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
    ) -> np.ndarray | da.Array:
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
                f"Unsupported mode: {mode}. Supported modes are: numpy, dask."
            )

    def _get_roi(
        self,
        roi: Roi | RoiPixels,
        axes_order: Sequence[str] | None = None,
        transforms: Sequence[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slice_kwargs: SlicingInputType,
    ) -> np.ndarray | da.Array:
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
                f"Unsupported mode: {mode}. Supported modes are: numpy, dask."
            )

    def _set_array(
        self,
        patch: np.ndarray | da.Array,
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
            numpy_setter = NumpySetter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            numpy_setter(patch)

        elif isinstance(patch, da.Array):
            dask_setter = DaskSetter(
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
        patch: np.ndarray | da.Array,
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
            roi_numpy_setter = NumpyRoiSetter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                roi=roi,
                axes_order=axes_order,
                transforms=transforms,
                slicing_dict=slicing_kwargs,
            )
            roi_numpy_setter(patch)

        elif isinstance(patch, da.Array):
            roi_dask_setter = DaskRoiSetter(
                zarr_array=self.zarr_array,
                dimensions=self.dimensions,
                roi=roi,
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
        order: InterpolationOrder = "linear",
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the image on disk.

        Args:
            order: The order of the consolidation.
            mode: The mode of the consolidation.
        """
        consolidate_image(image=self, order=order, mode=mode)

    def roi(self, name: str | None = "image") -> Roi:
        """Return the ROI covering the entire image."""
        dim_x = self.dimensions.get("x")
        dim_y = self.dimensions.get("y")
        assert dim_x is not None and dim_y is not None
        dim_z = self.dimensions.get("z")
        z = None if dim_z is None else 0
        dim_t = self.dimensions.get("t")
        t = None if dim_t is None else 0
        roi_px = RoiPixels(
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
        return roi_px.to_roi(pixel_size=self.pixel_size)

    def build_image_roi_table(self, name: str | None = "image") -> RoiTable:
        """Build the ROI table containing the ROI covering the entire image."""
        return RoiTable(rois=[self.roi(name=name)])

    def require_dimensions_match(
        self,
        other: "AbstractImage",
        allow_singleton: bool = False,
    ) -> None:
        """Assert that two images have matching spatial dimensions.

        Args:
            other: The other image to compare to.
            allow_singleton: If True, allow singleton dimensions to be
                compatible with non-singleton dimensions.

        Raises:
            NgioValueError: If the images do not have compatible dimensions.
        """
        self.dimensions.require_dimensions_match(
            other.dimensions, allow_singleton=allow_singleton
        )

    def check_if_dimensions_match(
        self,
        other: "AbstractImage",
        allow_singleton: bool = False,
    ) -> bool:
        """Check if two images have matching spatial dimensions.

        Args:
            other: The other image to compare to.
            allow_singleton: If True, allow singleton dimensions to be
                compatible with non-singleton dimensions.

        Returns:
            bool: True if the images have matching dimensions, False otherwise.
        """
        return self.dimensions.check_if_dimensions_match(
            other.dimensions, allow_singleton=allow_singleton
        )

    def require_axes_match(
        self,
        other: "AbstractImage",
    ) -> None:
        """Assert that two images have compatible axes.

        Args:
            other: The other image to compare to.

        Raises:
            NgioValueError: If the images do not have compatible axes.
        """
        self.dimensions.require_axes_match(other.dimensions)

    def check_if_axes_match(
        self,
        other: "AbstractImage",
    ) -> bool:
        """Check if two images have compatible axes.

        Args:
            other: The other image to compare to.

        Returns:
            bool: True if the images have compatible axes, False otherwise.

        """
        return self.dimensions.check_if_axes_match(other.dimensions)

    def require_rescalable(
        self,
        other: "AbstractImage",
    ) -> None:
        """Assert that two images can be rescaled to each other.

        For this to be true, the images must have the same axes, and
        the pixel sizes must be compatible (i.e. one can be scaled to the other).

        Args:
            other: The other image to compare to.

        Raises:
            NgioValueError: If the images cannot be scaled to each other.
        """
        self.dimensions.require_rescalable(other.dimensions)

    def check_if_rescalable(
        self,
        other: "AbstractImage",
    ) -> bool:
        """Check if two images can be rescaled to each other.

        For this to be true, the images must have the same axes, and
        the pixel sizes must be compatible (i.e. one can be scaled to the other).

        Args:
            other: The other image to compare to.

        Returns:
            bool: True if the images can be rescaled to each other, False otherwise.
        """
        return self.dimensions.check_if_rescalable(other.dimensions)


def consolidate_image(
    image: AbstractImage,
    order: InterpolationOrder = "linear",
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
