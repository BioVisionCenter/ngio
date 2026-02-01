from collections.abc import Sequence
from typing import TypeAlias

import dask.array as da
import numpy as np

from ngio.common import Roi
from ngio.experimental.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.images import Image, Label
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.io_pipes import (
    DaskRoiGetter,
    DataGetter,
    NumpyRoiGetter,
    TransformProtocol,
)

NumpyPipeType: TypeAlias = tuple[np.ndarray, np.ndarray, Roi]
DaskPipeType: TypeAlias = tuple[da.Array, da.Array, Roi]


class NumpyFeatureGetter(DataGetter[NumpyPipeType]):
    def __init__(
        self,
        image_getter: NumpyRoiGetter,
        label_getter: NumpyRoiGetter,
    ) -> None:
        self._image_getter = image_getter
        self._label_getter = label_getter
        super().__init__(
            zarr_array=self._image_getter.zarr_array,
            slicing_ops=self._image_getter.slicing_ops,
            axes_ops=self._image_getter.axes_ops,
            transforms=self._image_getter.transforms,
            roi=self._image_getter.roi,
        )

    def get(self) -> NumpyPipeType:
        return self._image_getter(), self._label_getter(), self.roi

    @property
    def image(self) -> np.ndarray:
        return self._image_getter()

    @property
    def label(self) -> np.ndarray:
        return self._label_getter()


class DaskFeatureGetter(DataGetter[DaskPipeType]):
    def __init__(
        self,
        image_getter: DaskRoiGetter,
        label_getter: DaskRoiGetter,
    ) -> None:
        self._image_getter = image_getter
        self._label_getter = label_getter
        super().__init__(
            zarr_array=self._image_getter.zarr_array,
            slicing_ops=self._image_getter.slicing_ops,
            axes_ops=self._image_getter.axes_ops,
            transforms=self._image_getter.transforms,
            roi=self._image_getter.roi,
        )

    def get(self) -> DaskPipeType:
        return self._image_getter(), self._label_getter(), self.roi

    @property
    def image(self) -> da.Array:
        return self._image_getter()

    @property
    def label(self) -> da.Array:
        return self._label_getter()


class FeatureExtractorIterator(AbstractIteratorBuilder[NumpyPipeType, DaskPipeType]):
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

        self._input.require_axes_match(self._input_label)
        self._input.require_rescalable(self._input_label)

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

    def build_numpy_getter(self, roi: Roi) -> NumpyFeatureGetter:
        data_getter = NumpyRoiGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = NumpyRoiGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=roi,
            remove_channel_selection=True,
        )
        return NumpyFeatureGetter(data_getter, label_getter)

    def build_dask_getter(self, roi: Roi) -> DaskFeatureGetter:
        data_getter = DaskRoiGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            roi=roi,
            slicing_dict=self._input_slicing_kwargs,
        )
        label_getter = DaskRoiGetter(
            zarr_array=self._input_label.zarr_array,
            dimensions=self._input_label.dimensions,
            axes_order=self._axes_order,
            transforms=self._label_transforms,
            roi=roi,
            remove_channel_selection=True,
        )
        return DaskFeatureGetter(data_getter, label_getter)

    def build_numpy_setter(self, roi: Roi) -> None:
        return None

    def build_dask_setter(self, roi: Roi) -> None:
        return None

    def post_consolidate(self):
        pass

    def iter_as_numpy(self):  # type: ignore[override]
        """Create an iterator over the pixels of the ROIs."""
        return self.iter(lazy=False, data_mode="numpy", iterator_mode="readonly")

    def iter_as_dask(self):  # type: ignore[override]
        """Create an iterator over the pixels of the ROIs."""
        return self.iter(lazy=False, data_mode="dask", iterator_mode="readonly")
