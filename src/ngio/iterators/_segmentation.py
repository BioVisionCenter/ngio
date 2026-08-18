from collections.abc import Sequence

import dask.array as da
import numpy as np

from ngio.common import Roi
from ngio.common._pyramid import ConsolidationMode
from ngio.images import Image, Label
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.images._masked_image import MaskedImage
from ngio.io_pipes import (
    DaskGetter,
    DaskSetter,
    NumpyGetter,
    NumpySetter,
    TransformProtocol,
)
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._mask_transform import BaseMaskTransform
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder


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
        consolidation_mode: ConsolidationMode | None = None,
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
            consolidation_mode: How to build the output pyramid after
                iteration, see `Label.consolidate`. Defaults to `None`.
        """
        self._input = input_image
        self._output = output_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()
        self._consolidation_mode = consolidation_mode

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
            "consolidation_mode": self._consolidation_mode,
        }

    @property
    def output_image(self) -> Label:
        """The label this iterator writes to."""
        return self._output

    def build_numpy_getter(self, roi: Roi) -> DataGetterProtocol[np.ndarray]:
        return NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi) -> DataSetterProtocol[np.ndarray]:
        return NumpySetter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def build_dask_getter(self, roi: Roi) -> DataGetterProtocol[da.Array]:
        return DaskGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi) -> DataSetterProtocol[da.Array]:
        return DaskSetter(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._output_transforms,
            remove_channel_selection=True,
        )

    def post_consolidate(self):
        self._output.consolidate(mode=self._consolidation_mode)


class MaskedSegmentationIterator(SegmentationIterator):
    """Base class for iterators over ROIs."""

    # Narrows the base class's `Image`: this iterator needs the masking label
    # and ROI table that only a `MaskedImage` carries.
    _input: MaskedImage

    def __init__(
        self,
        input_image: MaskedImage,
        output_label: Label,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        output_transforms: Sequence[TransformProtocol] | None = None,
        consolidation_mode: ConsolidationMode | None = None,
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
            consolidation_mode: How to build the output pyramid after
                iteration, see `Label.consolidate`. Defaults to `None`.
        """
        self._input = input_image
        self._output = output_label

        self._ref_image = input_image
        self._set_rois(input_image._masking_roi_table.rois())
        self._consolidation_mode = consolidation_mode

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
            "consolidation_mode": self._consolidation_mode,
        }

    def _input_transforms_with_mask(self) -> Sequence[TransformProtocol]:
        """The input transforms plus a terminal mask over the input image."""
        mask_transform = BaseMaskTransform(
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            target_dimensions=self._input.dimensions,
            set_transforms=self._input_transforms,
        )
        return [*(self._input_transforms or []), mask_transform]

    def _output_transforms_with_mask(self) -> Sequence[TransformProtocol]:
        """The output transforms plus a terminal mask over the output label."""
        mask_transform = BaseMaskTransform(
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            target_dimensions=self._output.dimensions,
            set_transforms=self._output_transforms,
        )
        return [*(self._output_transforms or []), mask_transform]

    def build_numpy_getter(self, roi: Roi):
        return NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=roi,
            axes_order=self._axes_order,
            transforms=self._input_transforms_with_mask(),
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi):
        return NumpySetter(
            roi=roi,
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms_with_mask(),
            remove_channel_selection=True,
        )

    def build_dask_getter(self, roi: Roi):
        return DaskGetter(
            roi=roi,
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms_with_mask(),
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi):
        return DaskSetter(
            roi=roi,
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            axes_order=self._axes_order,
            transforms=self._output_transforms_with_mask(),
            remove_channel_selection=True,
        )
