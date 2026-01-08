"""Generic class to handle Image-like data in a OME-NGFF file."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from zarr.core.array import CompressorLike

from ngio.common import (
    Dimensions,
    InterpolationOrder,
    Roi,
    consolidate_pyramid,
)
from ngio.common._pyramid import ChunksLike, ShardsLike, shapes_from_scaling_factors
from ngio.images._create_utils import (
    _image_or_label_meta,
    init_image_like_from_shapes,
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
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    Channel,
    NgffVersions,
    NgioLabelMeta,
)
from ngio.tables import RoiTable
from ngio.utils import (
    NgioFileExistsError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)
from ngio.utils._zarr_utils import find_dimension_separator


class AbstractImage(ABC):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | LabelMetaHandler,
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
    @abstractmethod
    def meta_handler(self) -> ImageMetaHandler | LabelMetaHandler:
        """Return the metadata."""
        pass

    @property
    @abstractmethod
    def meta(self) -> NgioImageMeta | NgioLabelMeta:
        """Return the metadata."""
        pass

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self.meta_handler.get_meta().get_dataset(path=self.path)

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
        roi: Roi,
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
        roi: Roi,
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
        roi: Roi,
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
        roi: Roi,
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
        slices = {}
        for ax_name in ["t", "z", "y", "x"]:
            axis_size = self.dimensions.get(ax_name, default=None)
            if axis_size is None:
                continue
            slices[ax_name] = slice(0, axis_size)
        roi_px = Roi.from_values(name=name, slices=slices, space="pixel")
        return roi_px.to_world(pixel_size=self.pixel_size)

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
    target_paths = image.meta_handler.get_meta().paths
    targets = [
        image._group_handler.get_array(path)
        for path in target_paths
        if path != image.path
    ]
    consolidate_pyramid(
        source=image.zarr_array, targets=targets, order=order, mode=mode
    )


def _shapes_from_ref_image(
    ref_image: AbstractImage,
) -> list[tuple[int, ...]]:
    """Rebuild base shape based on a new shape."""
    paths = ref_image.meta.paths
    index_path = paths.index(ref_image.path)
    sub_paths = paths[index_path:]
    group_handler = ref_image._group_handler
    shapes = []
    for path in sub_paths:
        zarr_array = group_handler.get_array(path)
        shapes.append(zarr_array.shape)
    if len(shapes) == len(paths):
        return shapes
    missing_levels = len(paths) - len(shapes)
    extended_shapes = shapes_from_scaling_factors(
        base_shape=shapes[-1],
        scaling_factors=ref_image.meta.scaling_factor(),
        num_levels=missing_levels + 1,
    )
    shapes.extend(extended_shapes[1:])
    return shapes


def _shapes_from_new_shape(
    ref_image: AbstractImage,
    shape: Sequence[int],
) -> list[tuple[int, ...]]:
    """Rebuild pyramid shapes based on a new base shape."""
    if len(shape) != len(ref_image.shape):
        raise NgioValueError(
            "The shape of the new image does not match the reference image."
            f"Got shape {shape} for reference shape {ref_image.shape}."
        )
    base_shape = tuple(shape)
    scaling_factors = ref_image.meta.scaling_factor()
    num_levels = len(ref_image.meta.paths)
    return shapes_from_scaling_factors(
        base_shape=base_shape,
        scaling_factors=scaling_factors,
        num_levels=num_levels,
    )


def _compute_pyramid_shapes(
    ref_image: AbstractImage,
    shape: Sequence[int] | None,
) -> list[tuple[int, ...]]:
    """Rebuild pyramid shapes based on a new base shape."""
    if shape is None:
        return _shapes_from_ref_image(ref_image=ref_image)
    return _shapes_from_new_shape(ref_image=ref_image, shape=shape)


def _check_chunks_and_shards_compatibility(
    ref_shape: tuple[int, ...],
    chunks: ChunksLike,
    shards: ShardsLike | None,
) -> None:
    """Check if the chunks and shards are compatible with the reference shape.

    Args:
        ref_shape: The reference shape.
        chunks: The chunks to check.
        shards: The shards to check.
    """
    if chunks != "auto":
        if len(chunks) != len(ref_shape):
            raise NgioValueError(
                "The length of the chunks must be the same as the number of dimensions."
            )
    if shards is not None and shards != "auto":
        if len(shards) != len(ref_shape):
            raise NgioValueError(
                "The length of the shards must be the same as the number of dimensions."
            )


def _apply_channel_policy(
    ref_image: AbstractImage,
    channels_policy: Literal["squeeze", "same", "singleton"] | int,
    shapes: list[tuple[int, ...]],
    axes: tuple[str, ...],
    chunks: ChunksLike,
    shards: ShardsLike | None,
) -> tuple[list[tuple[int, ...]], tuple[str, ...], ChunksLike, ShardsLike | None]:
    """Apply the channel policy to the shapes and axes.

    Args:
        ref_image: The reference image.
        channels_policy: The channels policy to apply.
        shapes: The shapes of the pyramid levels.
        axes: The axes of the image.
        chunks: The chunks of the image.
        shards: The shards of the image.

    Returns:
        The new shapes and axes after applying the channel policy.
    """
    if channels_policy == "same":
        return shapes, axes, chunks, shards

    if channels_policy == "singleton":
        # Treat 'singleton' as setting channel size to 1
        channels_policy = 1

    channel_index = ref_image.axes_handler.get_index("c")
    if channel_index is None:
        if channels_policy == "squeeze":
            return shapes, axes, chunks, shards
        raise NgioValueError(
            f"Cannot apply channel policy {channels_policy=} to an image "
            "without channels axis."
        )
    if channels_policy == "squeeze":
        new_shapes = []
        for shape in shapes:
            new_shape = shape[:channel_index] + shape[channel_index + 1 :]
            new_shapes.append(new_shape)
        new_axes = axes[:channel_index] + axes[channel_index + 1 :]
        if chunks == "auto":
            new_chunks: ChunksLike = "auto"
        else:
            new_chunks = chunks[:channel_index] + chunks[channel_index + 1 :]
        if shards == "auto" or shards is None:
            new_shards: ShardsLike | None = shards
        else:
            new_shards = shards[:channel_index] + shards[channel_index + 1 :]
        return new_shapes, new_axes, new_chunks, new_shards
    elif isinstance(channels_policy, int):
        new_shapes = []
        for shape in shapes:
            new_shape = (
                *shape[:channel_index],
                channels_policy,
                *shape[channel_index + 1 :],
            )
            new_shapes.append(new_shape)
        return new_shapes, axes, chunks, shards
    else:
        raise NgioValueError(
            f"Invalid channels policy: {channels_policy}. "
            "Must be 'squeeze', 'same', or an integer."
        )


def _check_channels_meta_compatibility(
    meta_type: type[_image_or_label_meta],
    ref_image: AbstractImage,
    channels_meta: Sequence[str | Channel] | None,
) -> Sequence[str | Channel] | None:
    """Check if the channels metadata is compatible with the reference image.

    Args:
        meta_type: The metadata type.
        ref_image: The reference image.
        channels_meta: The channels metadata to check.

    Returns:
        The channels metadata if compatible, None otherwise.
    """
    if issubclass(meta_type, NgioLabelMeta):
        if channels_meta is not None:
            raise NgioValueError("Cannot set channels_meta for a label image.")
        return None
    if channels_meta is not None:
        return channels_meta
    assert isinstance(ref_image.meta, NgioImageMeta)
    ref_meta = ref_image.meta
    index_c = ref_meta.axes_handler.get_index("c")
    if index_c is None:
        return None

    # If the channels number does not match, return None
    # Else return the channels metadata from the reference image
    ref_shape = ref_image.shape
    ref_num_channels = ref_shape[index_c] if index_c is not None else 1
    channels_ = ref_meta.channels_meta.channels if ref_meta.channels_meta else []
    # Reset to None if number of channels do not match
    channels_meta_ = channels_ if ref_num_channels == len(channels_) else None
    return channels_meta_


def abstract_derive(
    *,
    ref_image: AbstractImage,
    meta_type: type[_image_or_label_meta],
    store: StoreOrGroup,
    overwrite: bool = False,
    # Metadata parameters
    shape: Sequence[int] | None = None,
    pixelsize: float | tuple[float, float] | None = None,
    z_spacing: float | None = None,
    time_spacing: float | None = None,
    name: str | None = None,
    channels_policy: Literal["squeeze", "same", "singleton"] | int = "same",
    channels_meta: Sequence[str | Channel] | None = None,
    ngff_version: NgffVersions | None = None,
    # Zarr Array parameters
    chunks: ChunksLike | None = None,
    shards: ShardsLike | None = None,
    dtype: str | None = None,
    dimension_separator: Literal[".", "/"] | None = None,
    compressors: CompressorLike | None = None,
    extra_array_kwargs: Mapping[str, Any] | None = None,
    # Deprecated arguments
    labels: Sequence[str] | None = None,
    pixel_size: PixelSize | None = None,
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr image from an existing image.

    If a kwarg is not provided, the value from the reference image will be used.

    Args:
        ref_image (AbstractImage): The reference image to derive from.
        meta_type (type[_image_or_label_meta]): The metadata type to use.
        store (StoreOrGroup): The Zarr store or group to create the image in.
        overwrite (bool): Whether to overwrite an existing image.
        shape (Sequence[int] | None): The shape of the new image.
        pixelsize (float | tuple[float, float] | None): The pixel size of the new image.
        z_spacing (float | None): The z spacing of the new image.
        time_spacing (float | None): The time spacing of the new image.
        axes_names (Sequence[str] | None): The axes names of the new image.
        name (str | None): The name of the new image.
        channels_policy (Literal["squeeze", "same", "singleton"] | int):
            Possible policies:
            - If "squeeze", the channels axis will be removed (no matter its size).
            - If "same", the channels axis will be kept as is (if it exists).
            - If "singleton", the channels axis will be set to size 1.
            - If an integer is provided, the channels axis will be changed to have that
                size.
        channels_meta (Sequence[str | Channel] | None): The channels metadata
            of the new image.
        ngff_version (NgffVersions | None): The NGFF version to use.
        chunks (ChunksLike | None): The chunk shape of the new image.
        shards (ShardsLike | None): The shard shape of the new image.
        dtype (str | None): The data type of the new image.
        dimension_separator (DIMENSION_SEPARATOR | None): The separator to use for
            dimensions.
        compressors (CompressorLike | None): The compressors to use.
        extra_array_kwargs (Mapping[str, Any] | None): Extra arguments to pass to
            the zarr array creation.
        labels (Sequence[str] | None): The labels of the new image.
            This argument is DEPRECATED please use channels_meta instead.
        pixel_size (PixelSize | None): The pixel size of the new image.
            This argument is DEPRECATED please use pixelsize, z_spacing,
            and time_spacing instead.

    Returns:
        ImagesContainer: The new derived image.

    """
    # TODO: remove in ngio 0.6
    if labels is not None:
        warnings.warn(
            "The 'labels' argument is deprecated and will be removed in "
            "a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        channels_meta = list(labels)
    if pixel_size is not None:
        warnings.warn(
            "The 'pixel_size' argument is deprecated and will be removed in "
            "a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        pixelsize_ = (pixel_size.y, pixel_size.x)
        z_spacing_ = pixel_size.z
        time_spacing_ = pixel_size.t
    else:
        if pixelsize is None:
            pixelsize_ = (ref_image.pixel_size.y, ref_image.pixel_size.x)
        else:
            pixelsize_ = pixelsize

        if z_spacing is None:
            z_spacing_ = ref_image.pixel_size.z
        else:
            z_spacing_ = z_spacing

        if time_spacing is None:
            time_spacing_ = ref_image.pixel_size.t
        else:
            time_spacing_ = time_spacing
    ref_meta = ref_image.meta

    shapes = _compute_pyramid_shapes(
        shape=shape,
        ref_image=ref_image,
    )
    ref_shape = next(iter(shapes))

    if pixelsize is None:
        pixelsize = (ref_image.pixel_size.y, ref_image.pixel_size.x)

    if z_spacing is None:
        z_spacing = ref_image.pixel_size.z

    if time_spacing is None:
        time_spacing = ref_image.pixel_size.t

    if name is None:
        name = ref_meta.name

    if dtype is None:
        dtype = ref_image.dtype

    if dimension_separator is None:
        dimension_separator = find_dimension_separator(ref_image.zarr_array)

    if compressors is None:
        compressors = ref_image.zarr_array.compressors  # type: ignore

    if chunks is None:
        chunks = ref_image.zarr_array.chunks
    if shards is None:
        shards = ref_image.zarr_array.shards

    _check_chunks_and_shards_compatibility(
        ref_shape=ref_shape,
        chunks=chunks,
        shards=shards,
    )

    if ngff_version is None:
        ngff_version = ref_meta.version

    shapes, axes, chunks, shards = _apply_channel_policy(
        ref_image=ref_image,
        channels_policy=channels_policy,
        shapes=shapes,
        axes=ref_image.axes,
        chunks=chunks,
        shards=shards,
    )
    channels_meta_ = _check_channels_meta_compatibility(
        meta_type=meta_type,
        ref_image=ref_image,
        channels_meta=channels_meta,
    )

    handler = init_image_like_from_shapes(
        store=store,
        meta_type=meta_type,
        shapes=shapes,
        pixelsize=pixelsize_,
        z_spacing=z_spacing_,
        time_spacing=time_spacing_,
        levels=ref_meta.paths,
        time_unit=ref_image.time_unit,
        space_unit=ref_image.space_unit,
        axes_names=axes,
        name=name,
        channels_meta=channels_meta_,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressors=compressors,
        overwrite=overwrite,
        ngff_version=ngff_version,
        extra_array_kwargs=extra_array_kwargs,
    )
    return handler
