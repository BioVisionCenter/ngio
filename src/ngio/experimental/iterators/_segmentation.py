from collections.abc import Sequence

import dask.array as da
import numpy as np

from ngio.common import Roi
from ngio.experimental.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.images import Image, Label
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.images._masked_image import MaskedImage
from ngio.io_pipes import (
    DaskGetterMasked,
    DaskRoiGetter,
    DaskRoiSetter,
    DaskSetterMasked,
    NumpyGetterMasked,
    NumpyRoiGetter,
    NumpyRoiSetter,
    NumpySetterMasked,
    TransformProtocol,
)
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol


class SegmentationIterator(AbstractIteratorBuilder[np.ndarray, da.Array]):
    """Base class for iterators over ROIs."""

    def __init__(
        self,
        input_image: Image,
        output_label: Label,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        output_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (Image): The input image to be used as input for the
                segmentation.
            output_label (Label): The label image where the ROIs will be written.
            channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the segmentation.
            axes_order (Sequence[str] | None): Optional axes order for the
                segmentation.
            input_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the input image.
            output_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the output label.
        """
        self._input = input_image
        self._output = output_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        # Set iteration parameters
        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

        self._input.require_dimensions_match(self._output, allow_singleton=False)

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "output_label": self._output,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "output_transforms": self._output_transforms,
        }

    def build_numpy_getter(self, roi: Roi) -> DataGetterProtocol[np.ndarray]:
        return NumpyRoiGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi) -> DataSetterProtocol[np.ndarray]:
        return NumpyRoiSetter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def build_dask_getter(self, roi: Roi) -> DataGetterProtocol[da.Array]:
        return DaskRoiGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi) -> DataSetterProtocol[da.Array]:
        return DaskRoiSetter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def post_consolidate(self):
        self._output.consolidate()


class MaskedSegmentationIterator(SegmentationIterator):
    """Base class for iterators over ROIs."""

    def __init__(
        self,
        input_image: MaskedImage,
        output_label: Label,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        output_transforms: Sequence[TransformProtocol] | None = None,
    ) -> None:
        """Initialize the iterator with a ROI table and input/output images.

        Args:
            input_image (MaskedImage): The input image to be used as input for the
                segmentation.
            output_label (Label): The label image where the ROIs will be written.
            channel_selection (ChannelSlicingInputType): Optional
                selection of channels to use for the segmentation.
            axes_order (Sequence[str] | None): Optional axes order for the
                segmentation.
            input_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the input image.
            output_transforms (Sequence[TransformProtocol] | None): Optional
                transforms to apply to the output label.
        """
        self._input = input_image
        self._output = output_label

        self._ref_image = input_image
        self._set_rois(input_image._masking_roi_table.rois())

        # Set iteration parameters
        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

        # Check compatibility between input and output images
        # if not self._input.dimensions.is_compatible_with(self._output.dimensions):
        #    raise NgioValidationError(
        #        "Input image and output label have incompatible dimensions. "
        #        f"Input: {self._input.dimensions}, Output: {self._output.dimensions}."
        #    )

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "output_label": self._output,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "output_transforms": self._output_transforms,
        }

    def build_numpy_getter(self, roi: Roi):
        return NumpyGetterMasked(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi):
        return NumpySetterMasked(
            roi=roi,
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def build_dask_getter(self, roi: Roi):
        return DaskGetterMasked(
            roi=roi,
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi):
        return DaskSetterMasked(
            roi=roi,
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def post_consolidate(self):
        self._output.consolidate()
