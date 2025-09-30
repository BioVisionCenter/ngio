"""Generic class to handle Image-like data in a OME-NGFF file."""

import math
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
from ngio.utils._errors import NgioValueError

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
            shape=self._zarr_array.shape, axes_handler=self._dataset.axes_handler
        )
        self._axes_mapper = self._dataset.axes_handler

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
    def axes_mapper(self) -> AxesHandler:
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
                pixel_size=self.pixel_size,
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
        order: InterpolationOrder = "linear",
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the image on disk.

        Args:
            order: The order of the consolidation.
            mode: The mode of the consolidation.
        """
        consolidate_image(image=self, order=order, mode=mode)

    def build_image_roi_table(self, name: str | None = "image") -> RoiTable:
        """Build the ROI table for an image."""
        return build_image_roi_table(image=self, name=name)

    def assert_dimensions_match(
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
        assert_dimensions_match(
            image1=self, image2=other, allow_singleton=allow_singleton
        )

    def assert_axes_match(
        self,
        other: "AbstractImage",
    ) -> None:
        """Assert that two images have compatible axes.

        Args:
            other: The other image to compare to.

        Raises:
            NgioValueError: If the images do not have compatible axes.
        """
        assert_axes_match(image1=self, image2=other)

    def assert_can_be_rescaled(
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
        assert_can_be_rescaled(image1=self, image2=other)


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


def build_image_roi_table(image: AbstractImage, name: str | None = "image") -> RoiTable:
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


def assert_dimensions_match(
    image1: AbstractImage,
    image2: AbstractImage,
    allow_singleton: bool = False,
) -> None:
    """Assert that two images have matching spatial dimensions.

    Args:
        image1: The first image.
        image2: The second image.
        allow_singleton: If True, allow singleton dimensions to be
            compatible with non-singleton dimensions.

    Raises:
        NgioValueError: If the images do not have compatible dimensions.
    """
    image1.dimensions.assert_dimensions_match(
        other=image2.dimensions, allow_singleton=allow_singleton
    )


def assert_axes_match(
    image1: AbstractImage,
    image2: AbstractImage,
) -> None:
    """Assert that two images have compatible axes.

    Args:
        image1: The first image.
        image2: The second image.

    Raises:
        NgioValueError: If the images do not have compatible axes.
    """
    image1.dimensions.assert_axes_match(other=image2.dimensions)


def _are_compatible(shape1: int, shape2: int, scaling: float) -> bool:
    """Check if shape2 is consistent with shape1 given pixel sizes.

    Since we only deal with shape discrepancies due to rounding, we
    shape1, needs to be larger than shape2.
    """
    if shape1 < shape2:
        return _are_compatible(shape2, shape1, 1 / scaling)
    expected_shape2 = shape1 * scaling
    expected_shape2_floor = math.floor(expected_shape2)
    expected_shape2_ceil = math.ceil(expected_shape2)
    return shape2 in {expected_shape2_floor, expected_shape2_ceil}


def assert_can_be_rescaled(
    image1: AbstractImage,
    image2: AbstractImage,
) -> None:
    """Assert that two images can be rescaled to each other.

    For this to be true, the images must have the same axes, and
    the pixel sizes must be compatible (i.e. one can be scaled to the other).

    Args:
        image1: The first image.
        image2: The second image.

    Raises:
        NgioValueError: If the images cannot be scaled to each other.
    """
    assert_axes_match(image1=image1, image2=image2)
    for ax1 in image1.dimensions.axes_handler.axes:
        if ax1.axis_type == "channel":
            continue
        ax2 = image2.dimensions.axes_handler.get_axis(ax1.name)
        assert ax2 is not None, "Axes do not match."
        px1 = image1.pixel_size.get(ax1.name, default=1.0)
        px2 = image2.pixel_size.get(ax2.name, default=1.0)
        shape1 = image1.dimensions.get(ax1.name, default=1)
        shape2 = image2.dimensions.get(ax2.name, default=1)
        scale = px1 / px2
        if not _are_compatible(
            shape1=shape1,
            shape2=shape2,
            scaling=scale,
        ):
            raise NgioValueError(
                f"Image1 with shape {image1.shape}, "
                f"and pixel size {image1.pixel_size}, "
                f"cannot be rescaled to "
                f"Image2 with shape {image2.shape}, "
                f"and pixel size {image2.pixel_size}. "
            )
