from collections.abc import Callable, Sequence

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
from ngio.io_pipes._mask_transform import BaseMaskMerge, BaseMaskTransform
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.iterators._mappers import MapperProtocol
from ngio.iterators._stitch import StitchConfig, StitchingSetter, StitchPlan
from ngio.utils import NgioValueError


class SegmentationIterator(AbstractIteratorBuilder[np.ndarray, da.Array]):
    """Segment an image region by region into a label.

    Reads each region from the input image and writes the function's label
    patch to the output label. With `stitch=True` (and a halo) objects split
    by tile boundaries are resolved into one id after the map; see
    `StitchConfig`.
    """

    # Class-level defaults so subclasses that write their own `__init__` — the
    # masked iterator does — are simply not stitching, rather than missing an
    # attribute the inherited `finalize` reads. Stitching needs a tile
    # grid, which a masking ROI table does not have.
    _stitch: StitchConfig | None = None
    _stitch_plan: StitchPlan | None = None

    def __init__(
        self,
        input_image: Image,
        output_label: Label,
        *,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        output_transforms: Sequence[TransformProtocol] | None = None,
        consolidation_mode: ConsolidationMode | None = None,
        stitch: StitchConfig | bool = False,
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
            stitch: Resolve objects split across tile boundaries into one id
                after the map. Requires a halo — see `with_halo` — because the
                evidence is overlap between neighbouring tiles' predictions.
                `True` uses the `StitchConfig` defaults.
        """
        self._input = input_image
        self._output = output_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()
        self._consolidation_mode = consolidation_mode
        self._stitch = StitchConfig() if stitch is True else stitch or None
        self._stitch_plan: StitchPlan | None = None

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
            "stitch": self._stitch or False,
        }

    @property
    def output_image(self) -> Label:
        """The label this iterator writes to."""
        return self._output

    def build_numpy_getter(self, roi: Roi) -> DataGetterProtocol[np.ndarray]:
        return NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=self._read_roi(roi),
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def _stitching_plan(self) -> StitchPlan:
        """The stitch plan, built once the full ROI list is known."""
        if self._stitch_plan is None:
            assert self._stitch is not None
            self._stitch_plan = StitchPlan(
                config=self._stitch,
                output=self._output,
                rois=self.rois,
                ref_image=self._ref_image,
                halo=self.halo,
                read_roi=self._read_roi,
            )
        return self._stitch_plan

    def _wrap_for_stitch(
        self, setter: DataSetterProtocol[np.ndarray], roi: Roi
    ) -> DataSetterProtocol[np.ndarray]:
        """Put the stitch wrapper outside the halo crop, so it sees the band."""
        if self._stitch is None:
            return setter
        return StitchingSetter(setter, self._stitching_plan(), roi)

    def build_numpy_setter(self, roi: Roi) -> DataSetterProtocol[np.ndarray]:
        return self._wrap_for_stitch(
            self._wrap_setter(
                NumpySetter(
                    zarr_array=self._output.zarr_array,
                    dimensions=self._output.dimensions,
                    roi=roi,
                    axes_order=self._axes_order,
                    transforms=self._output_transforms,
                    remove_channel_selection=True,
                ),
                roi,
            ),
            roi,
        )

    def build_dask_getter(self, roi: Roi) -> DataGetterProtocol[da.Array]:
        return DaskGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=self._read_roi(roi),
            axes_order=self._axes_order,
            transforms=self._input_transforms,
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi) -> DataSetterProtocol[da.Array]:
        if self._stitch is not None:
            # Without this, the dask write path would skip the id offsets and
            # the band banking, and the resolve afterwards would compact the
            # colliding tile-local ids into silently wrong labels.
            raise NgioValueError(
                "Stitching is only supported on the numpy path: the dask "
                "setters do not offset ids or bank halo bands, so the resolve "
                "would corrupt the labels. Use map()/iter(data_mode='numpy'), "
                "or build the iterator with stitch=False."
            )
        return self._wrap_setter(
            DaskSetter(
                zarr_array=self._output.zarr_array,
                dimensions=self._output.dimensions,
                roi=roi,
                axes_order=self._axes_order,
                transforms=self._output_transforms,
                remove_channel_selection=True,
            ),
            roi,
        )

    def map(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        mapper: MapperProtocol[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """See `AbstractIteratorBuilder.map`; also cleans up on failure.

        A failed run cannot be resolved, so the stitch scratch arrays are
        deleted rather than left as a stray `_ngio_stitch` group beside the
        resolution levels. The already-written tiles stay: re-running the map
        recreates the scratch and produces the same ids (the offsets are
        derived, not counted).
        """
        if self._stitch is None:
            return super().map(func, mapper=mapper)
        try:
            return super().map(func, mapper=mapper)
        except BaseException:
            if self._stitch_plan is not None:
                self._stitch_plan.cleanup()
                self._stitch_plan = None
            raise

    def finalize(self):
        # The relabel has to precede consolidation: every pyramid level is
        # derived from level 0, so stitching after would leave them disagreeing.
        if self._stitch is not None:
            self._stitching_plan().resolve()
            self._stitch_plan = None
        self._output.consolidate(mode=self._consolidation_mode)


class MaskedSegmentationIterator(SegmentationIterator):
    """Segment each object of a masking ROI table, inside its own mask.

    Regions come from the masking table's per-object bounding boxes; reads
    are masked to the object (outside pixels filled) and writes protect
    everything outside it (`MaskMerge`). There is no `stitch` option here:
    stitching needs a regular tile grid, which per-object bounding boxes do
    not form — ids across objects are the caller's to keep unique, e.g. with
    `output_transforms=[UniqueLabelsTransform(block_size)]`, where the ROI's
    own label supplies the block index.
    """

    # Narrows the base class's `Image`: this iterator needs the masking label
    # and ROI table that only a `MaskedImage` carries.
    _input: MaskedImage

    def __init__(
        self,
        input_image: MaskedImage,
        output_label: Label,
        *,
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

    def _input_transforms_with_mask(self) -> Sequence[TransformProtocol]:
        """The input transforms plus a terminal mask over the input image."""
        mask_transform = BaseMaskTransform(
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            target_dimensions=self._input.dimensions,
        )
        return [*(self._input_transforms or []), mask_transform]

    def _output_mask_merge(self) -> BaseMaskMerge:
        """The mask protecting everything outside the ROI's own label."""
        return BaseMaskMerge(
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            target_dimensions=self._output.dimensions,
        )

    def build_numpy_getter(self, roi: Roi):
        return NumpyGetter(
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            roi=self._read_roi(roi),
            axes_order=self._axes_order,
            transforms=self._input_transforms_with_mask(),
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_numpy_setter(self, roi: Roi):
        return self._wrap_setter(
            NumpySetter(
                roi=roi,
                zarr_array=self._output.zarr_array,
                dimensions=self._output.dimensions,
                axes_order=self._axes_order,
                transforms=self._output_transforms,
                merge=self._output_mask_merge(),
                remove_channel_selection=True,
            ),
            roi,
        )

    def build_dask_getter(self, roi: Roi):
        return DaskGetter(
            roi=self._read_roi(roi),
            zarr_array=self._input.zarr_array,
            dimensions=self._input.dimensions,
            axes_order=self._axes_order,
            transforms=self._input_transforms_with_mask(),
            slicing_dict=self._input_slicing_kwargs,
        )

    def build_dask_setter(self, roi: Roi):
        return self._wrap_setter(
            DaskSetter(
                roi=roi,
                zarr_array=self._output.zarr_array,
                dimensions=self._output.dimensions,
                axes_order=self._axes_order,
                transforms=self._output_transforms,
                merge=self._output_mask_merge(),
                remove_channel_selection=True,
            ),
            roi,
        )
