from collections.abc import Callable, Generator, Sequence

import dask.array as da
import numpy as np

from ngio.common import (
    Roi,
    TransformProtocol,
    build_roi_dask_getter,
    build_roi_numpy_getter,
)
from ngio.experimental.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.images import Image, Label
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.utils import NgioValueError


class FeatureExtractorIterator(AbstractIteratorBuilder):
    """Base class for iterators over ROIs."""

    def __init__(
        self,
        input_image: Image,
        input_label: Label,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        label_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (Image): The input image to be used as input for the
                segmentation.
            input_label (Label): The input label with the segmentation masks.
            channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the segmentation.
            axes_order (Sequence[str] | None): Optional axes order for the
                segmentation.
            input_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the input image.
            label_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the output label.
        """
        self._input = input_image
        self._input_label = input_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        # Set iteration parameters
        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._label_transforms = label_transforms

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "input_label": self._input_label,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "label_transforms": self._label_transforms,
        }

    def assert_axes_match(self) -> None:
        """Check that the input image and label have the same axes.

        Besides the channel axis, all axes must be present in both images.

        Raises:
            NgioValueError: If the axes do not match.
        """
        for axis in self._input.dimensions._axes_mapper.axes:
            if axis.axis_type == "channel":
                continue
            l_axis = self._input_label.dimensions._axes_mapper.get_axis(axis.name)
            if l_axis is None:
                raise NgioValueError(
                    f"Input image and label must have the same axes. "
                    f"Axis {axis.name} is missing in the label."
                )

    def assert_dimensions_match(self, allow_singleton: bool = False) -> None:
        """Check that the input image and label have compatible dimensions.

        Besides the channel axis, all axes must have the same dimension in
        both images.

        Args:
            allow_singleton: Whether to allow singleton dimensions to be
                different. For example, if the input image has shape
                (5, 100, 100) and the label has shape (1, 100, 100).

        Raises:
            NgioValueError: If the dimensions do not match.
        """
        self.assert_axes_match()
        for axis in self._input.dimensions._axes_mapper.axes:
            l_axis = self._input_label.dimensions._axes_mapper.get_axis(axis.name)
            assert l_axis is not None  # already checked in assert_axes_match
            i_dim = self._input.dimensions.get(axis.name)
            l_dim = self._input_label.dimensions.get(axis.name)
            if i_dim != l_dim:
                if allow_singleton and (i_dim == 1 or l_dim == 1):
                    continue
                raise NgioValueError(
                    f"Input image and label must have the same dimensions. "
                    f"Axis {axis.name} has dimension {i_dim} in the "
                    f"input image and {l_dim} in the label."
                )

    def build_numpy_getter(self, roi: Roi):
        data_getter = build_roi_numpy_getter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            pixel_size=self._input.pixel_size,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = build_roi_numpy_getter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            pixel_size=self._input_label.pixel_size,
            roi=roi,
            remove_channel_selection=True,
        )
        return lambda: (data_getter(), label_getter(), roi)

    def build_numpy_setter(self, roi: Roi):
        return None

    def build_dask_getter(self, roi: Roi):
        data_getter = build_roi_dask_getter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            pixel_size=self._input.pixel_size,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = build_roi_dask_getter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            pixel_size=self._input_label.pixel_size,
            roi=roi,
            remove_channel_selection=True,
        )
        return lambda: (data_getter(), label_getter(), roi)

    def build_dask_setter(self, roi: Roi):
        return None

    def post_consolidate(self):
        pass

    def iter_as_numpy(self) -> Generator[tuple[np.ndarray, np.ndarray, Roi]]:  # type: ignore (non compatible override)
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Returns:
            Generator[tuple[da.Array, DaskWriter]]: An iterator the input
                image as Dask arrays and a writer to write the output
                to the label image.
        """
        for (data, label, roi), _ in super().iter_as_numpy():
            yield data, label, roi

    def map_as_numpy(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a transformation function to the ROI pixels."""
        raise NotImplementedError("Numpy mapping not implemented for this iterator.")

    def iter_as_dask(self) -> Generator[tuple[da.Array, da.Array, Roi]]:  # type: ignore (non compatible override)
        """Create an iterator over the pixels of the ROIs as Dask arrays.

        Returns:
            Generator[tuple[da.Array, DaskWriter]]: An iterator the input
                image as Dask arrays and a writer to write the output
                to the label image.
        """
        for (data, label, roi), _ in super().iter_as_dask():
            yield data, label, roi

    def map_as_dask(self, func: Callable[[da.Array], da.Array]) -> None:
        """Apply a transformation function to the ROI pixels."""
        raise NotImplementedError("Dask mapping not implemented for this iterator.")
