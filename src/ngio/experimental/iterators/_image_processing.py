from collections.abc import Callable, Generator, Sequence

import dask.array as da
import numpy as np

from ngio.common import (
    Roi,
    TransformProtocol,
    build_roi_dask_getter,
    build_roi_dask_setter,
    build_roi_numpy_getter,
    build_roi_numpy_setter,
)
from ngio.experimental.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.images import Image
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.utils._errors import NgioValidationError


class ImageProcessingIterator(AbstractIteratorBuilder):
    """Base class for iterators over ROIs."""

    def __init__(
        self,
        input_image: Image,
        output_image: Image,
        input_channel_selection: ChannelSlicingInputType = None,
        output_channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        output_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (Image): The input image to be used as input for the
                segmentation.
            output_image (Image): The image where the ROIs will be written.
            input_channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the input image.
            output_channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the output image.
            axes_order (Sequence[str] | None): Optional axes order for the
                segmentation.
            input_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the input image.
            output_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the output label.
        """
        self._input = input_image
        self._output = output_image
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table().rois()

        # Set iteration parameters
        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input,
            channel_selection=input_channel_selection,
            slicing_dict={},
        )
        self._output_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._output,
            channel_selection=output_channel_selection,
            slicing_dict={},
        )
        self._input_channel_selection = input_channel_selection
        self._output_channel_selection = output_channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

        # Check compatibility between input and output images
        if not self._input.dimensions.is_compatible_with(self._output.dimensions):
            raise NgioValidationError(
                "Input image and output label have incompatible dimensions. "
                f"Input: {self._input.dimensions}, Output: {self._output.dimensions}."
            )

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "output_image": self._output,
            "input_channel_selection": self._input_channel_selection,
            "output_channel_selection": self._output_channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "output_transforms": self._output_transforms,
        }

    def build_numpy_getter(self, roi: Roi):
        return build_roi_numpy_getter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            pixel_size=self._input.pixel_size,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi):
        return build_roi_numpy_setter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            pixel_size=self._output.pixel_size,
            roi=roi,
            slicing_dict=self._output_slicing_kwargs,
        )

    def build_dask_getter(self, roi: Roi):
        return build_roi_dask_getter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            pixel_size=self._input.pixel_size,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi):
        return build_roi_dask_setter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            pixel_size=self._output.pixel_size,
            roi=roi,
            slicing_dict=self._output_slicing_kwargs,
        )

    def post_consolidate(self):
        self._output.consolidate()

    def iter_as_numpy(
        self,
    ) -> Generator[tuple[np.ndarray, Callable[[np.ndarray], None]]]:
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Returns:
            Generator[tuple[da.Array, DaskWriter]]: An iterator the input
                image as Dask arrays and a writer to write the output
                to the label image.
        """
        return super().iter_as_numpy()

    def map_as_numpy(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a transformation function to the ROI pixels."""
        return super().map_as_numpy(func)

    def iter_as_dask(self) -> Generator[tuple[da.Array, Callable[[da.Array], None]]]:
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Returns:
            Generator[tuple[da.Array, DaskWriter]]: An iterator the input
                image as Dask arrays and a writer to write the output
                to the label image.
        """
        return super().iter_as_dask()

    def map_as_dask(self, func: Callable[[da.Array], da.Array]) -> None:
        """Apply a transformation function to the ROI pixels."""
        return super().map_as_dask(func)
