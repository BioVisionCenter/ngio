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
from ngio.io_pipes._io_pipe_ops import IoPipeContext, setup_io_pipe
from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.io_pipes._mask_transform import BaseMaskMerge, BaseMaskTransform
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.iterators._mappers import MapperProtocol
from ngio.iterators._stitch import (
    ScratchBanks,
    StitchConfig,
    StitchingSetter,
    StitchPlan,
    read_scratch_attrs,
)
from ngio.transforms._unique_labels import UniqueLabelsTransform
from ngio.utils import NgioValueError


def _require_no_unique_labels_with_stitch(
    stitch: StitchConfig | None,
    output_transforms: Sequence[TransformProtocol] | None,
) -> None:
    """Refuse combining the stitch's id blocks with `UniqueLabelsTransform`.

    Both offset ids: the stitch outside the chain, the transform inside it.
    The bank would hold singly-offset ids while the disk holds doubly-offset
    ones, so the resolve would relabel garbage. The stitch's per-tile blocks
    plus the final compaction already make ids unique everywhere.
    """
    if stitch is None:
        return
    if any(
        isinstance(transform, UniqueLabelsTransform)
        for transform in output_transforms or []
    ):
        raise NgioValueError(
            "stitch= and UniqueLabelsTransform cannot be combined: both "
            "offset the ids, and the doubly-offset labels on disk would no "
            "longer match the banked predictions. Stitching already keeps "
            "ids unique (per-tile blocks, compacted at finalize) — drop the "
            "transform."
        )


class _MaskedBankMask:
    """Zeroes a banked patch outside its object's mask.

    A named class rather than a closure so it pickles: `ProcessMapper`
    ships the setters (and this with them) to the workers.
    """

    def __init__(self, mask_transform: BaseMaskTransform, ctx: IoPipeContext) -> None:
        self._mask_transform = mask_transform
        self._ctx = ctx

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        masked = self._mask_transform.on_get(patch, self._ctx)
        return np.asarray(masked)


class SegmentationIterator(AbstractIteratorBuilder[np.ndarray, da.Array]):
    """Segment an image region by region into a label.

    Reads each region from the input image and writes the function's label
    patch to the output label. With `stitch=True` objects split across region
    boundaries are resolved into one id after the map — any ROI list works:
    grids with a halo, overlapping FOV layouts, ragged tables. See
    `StitchConfig`.
    """

    # Class-level defaults so subclasses that write their own `__init__` are
    # simply not stitching, rather than missing an attribute the inherited
    # `finalize` reads.
    _stitch: StitchConfig | None = None
    _stitch_plan: StitchPlan | None = None
    # The masked iterator restricts stitch comparisons to same-object tiles:
    # an object cannot span two masks, so cross-label pairs can never merge.
    _stitch_same_label_only: bool = False

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
            stitch: Resolve objects split across region boundaries into one
                id after the map. Needs a halo (`with_halo`) or overlapping
                ROIs — the evidence is overlap between neighbouring
                predictions. `True` uses the `StitchConfig` defaults.
        """
        self._input = input_image
        self._output = output_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()
        self._consolidation_mode = consolidation_mode
        self._stitch = StitchConfig() if stitch is True else stitch or None
        self._stitch_plan: StitchPlan | None = None
        _require_no_unique_labels_with_stitch(self._stitch, output_transforms)

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
                halo=self.halo,
                read_roi=self._read_roi,
                scratch_factory=self._scratch_factory,
                same_label_only=self._stitch_same_label_only,
            )
        return self._stitch_plan

    def _stitch_bank_transform(
        self, roi: Roi
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        """Hook reshaping what a tile banks; the masked iterator masks it."""
        return None

    def _scratch_matches(self, attrs: dict) -> bool:
        """Whether existing scratch attrs describe this iterator's plan."""
        n_jobs = attrs.get("n_jobs")
        return isinstance(n_jobs, int) and attrs.get(
            "fingerprint"
        ) == self._plan_fingerprint(n_jobs)

    def _require_scratch_match(self, attrs: dict, n_jobs: int) -> None:
        if attrs.get("n_jobs") != n_jobs or attrs.get(
            "fingerprint"
        ) != self._plan_fingerprint(n_jobs):
            raise NgioValueError(
                "The stitch scratch was prepared for a different plan: the "
                "tiling, halo, stitch config, or n_jobs changed since "
                "`prepare_jobs`. Re-run `prepare_jobs(n_jobs)` with the "
                "current iterator, then resubmit the jobs."
            )

    def _scratch_factory(self) -> ScratchBanks:
        """Resolve the scratch banks for the current role.

        A partition slice must find the prepared scratch root (and it must
        match this plan) — jobs never create shared state; each job creates
        only its own tiles' bank arrays as it writes them. The unrestricted
        iterator opens a matching prepared root when one exists (the gather
        step, or a resumed run) and otherwise creates a fresh one, which is
        the standalone `map` behaviour.
        """
        assert self._stitch is not None
        store = self._stitch.scratch_store
        attrs = read_scratch_attrs(self._output, store)
        if self._partition is not None:
            _, _, n_jobs = self._partition
            if attrs is None:
                raise NgioValueError(
                    "This job's stitch scratch does not exist: run "
                    "`prepare_jobs(n_jobs)` once before submitting the jobs."
                )
            self._require_scratch_match(attrs, n_jobs=n_jobs)
            return ScratchBanks.open(self._output, store)
        if attrs is not None and self._scratch_matches(attrs):
            return ScratchBanks.open(self._output, store)
        if attrs is not None and "fingerprint" in attrs:
            # A prepared distributed scratch that does not match this plan:
            # wiping it would destroy every job's banked prediction. Only
            # `prepare_jobs` may start over.
            raise NgioValueError(
                "A prepared stitch scratch exists but was made for a "
                "different plan: the tiling, halo, stitch config, or n_jobs "
                "changed. Refusing to wipe the jobs' banked predictions — "
                "rebuild the iterator to match, or re-run "
                "`prepare_jobs(n_jobs)` to deliberately start over."
            )
        return ScratchBanks.create(self._output, store)

    def _prepare_distributed(self, n_jobs: int) -> None:
        """Create the scratch root, wiping any stale one first.

        Root attributes only — each job creates its own tiles' bank arrays
        (sibling creation touches no shared metadata, so it is race-free).
        """
        if self._stitch is None:
            return
        self._stitching_plan()  # runs the plan's validations
        ScratchBanks.create(
            self._output,
            self._stitch.scratch_store,
            extra_attrs={
                "fingerprint": self._plan_fingerprint(n_jobs),
                "n_jobs": n_jobs,
            },
        )

    def _validate_job_context(self, n_jobs: int) -> None:
        """`for_job` on a stitched iterator needs the prepared scratch."""
        if self._stitch is None:
            return
        attrs = read_scratch_attrs(self._output, self._stitch.scratch_store)
        if attrs is None:
            raise NgioValueError(
                "A distributed stitched run needs `prepare_jobs(n_jobs)` "
                "before `for_job`: the scratch root must exist, race-free, "
                "before any job banks into it."
            )
        self._require_scratch_match(attrs, n_jobs=n_jobs)

    def _fingerprint_extras(self) -> tuple[str, ...]:
        if self._stitch is None:
            return ()
        return (
            f"stitch.block_size={self._stitch.block_size}",
            f"stitch.iou_threshold={self._stitch.iou_threshold}",
            f"stitch.compact={self._stitch.compact}",
        )

    def _wrap_for_stitch(
        self, setter: DataSetterProtocol[np.ndarray], roi: Roi
    ) -> DataSetterProtocol[np.ndarray]:
        """Put the stitch wrapper outside the halo crop, so it sees the grown patch."""
        if self._stitch is None:
            return setter
        return StitchingSetter(
            setter,
            self._stitching_plan(),
            roi,
            bank_transform=self._stitch_bank_transform(roi),
        )

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

    def _require_stitchless_dask(self) -> None:
        """The dask setters neither offset ids nor bank; raising beats corrupting."""
        if self._stitch is not None:
            raise NgioValueError(
                "Stitching is only supported on the numpy path: the dask "
                "setters do not offset ids or bank predictions, so the "
                "resolve would corrupt the labels. Use "
                "map()/iter(data_mode='numpy'), or build the iterator with "
                "stitch=False."
            )

    def build_dask_setter(self, roi: Roi) -> DataSetterProtocol[da.Array]:
        self._require_stitchless_dask()
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
        *,
        mapper: MapperProtocol[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """See `AbstractIteratorBuilder.map`; also cleans up on failure.

        A failed standalone run cannot be resolved, so the stitch scratch
        arrays are deleted rather than left as a stray `_ngio_stitch` group
        beside the resolution levels. A *partition slice* never cleans up:
        the scratch holds the banks every other job wrote, and one failed
        job must not destroy them — re-running that job is idempotent (its
        banks rewrite, the id offsets are derived, not counted). The
        already-written tiles stay in both cases.
        """
        if self._stitch is None or self._partition is not None:
            return super().map(func, mapper=mapper)
        try:
            return super().map(func, mapper=mapper)
        except BaseException:
            if self._stitch_plan is not None:
                self._stitch_plan.cleanup()
                self._stitch_plan = None
            raise

    def finalize(self):
        self._require_unrestricted_finalize()
        # The relabel has to precede consolidation: every pyramid level is
        # derived from level 0, so stitching after would leave them disagreeing.
        if self._stitch is not None:
            # The stitch resolve relabels level 0 wherever the union-find
            # reached, not just under the written ROIs -- only a full rebuild
            # is guaranteed consistent after it.
            self._stitching_plan().resolve()
            self._stitch_plan = None
            self._output.consolidate(mode=self._consolidation_mode)
        else:
            self._output.consolidate(
                mode=self._consolidation_mode, regions=self._touched_write_regions()
            )


class MaskedSegmentationIterator(SegmentationIterator):
    """Segment each object of a masking ROI table, inside its own mask.

    Regions come from the masking table's per-object bounding boxes; reads
    are masked to the object (outside pixels filled) and writes protect
    everything outside it (`MaskMerge`) — overlapping bounding boxes are
    safe, because each write only touches its own object's pixels. With
    `stitch=True` and a tiling (`by_grid` + `with_halo`), sub-objects split
    by a tile boundary *within one mask* merge into one id; tiles of
    different masks are never compared (an object cannot span two masks),
    and ids come out unique and dense across every object — no
    `UniqueLabelsTransform` needed (combining it with `stitch` raises).
    Mind `block_size`: the id ceiling scales with objects times tiles per
    object.
    """

    # Narrows the base class's `Image`: this iterator needs the masking label
    # and ROI table that only a `MaskedImage` carries.
    _input: MaskedImage
    # Only same-object tiles are compared at resolve: cross-mask predictions
    # are disjoint by construction, so comparing them is pure wasted IO.
    _stitch_same_label_only: bool = True

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
        stitch: StitchConfig | bool = False,
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
            stitch: Merge sub-objects split by a tile boundary within one
                mask into one id after the map — for masked objects tiled
                with `by_grid` + `with_halo`. `True` uses the `StitchConfig`
                defaults.
        """
        self._input = input_image
        self._output = output_label

        self._ref_image = input_image
        self._set_rois(input_image._masking_roi_table.rois())
        self._consolidation_mode = consolidation_mode
        self._stitch = StitchConfig() if stitch is True else stitch or None
        self._stitch_plan = None
        _require_no_unique_labels_with_stitch(self._stitch, output_transforms)

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

    def _stitch_bank_transform(
        self, roi: Roi
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        """Bank only what is writable: zero the patch outside its own mask.

        An object spanning the mask boundary inside the patch carries one id
        for its real (in-mask) and fill-fed (cropped-at-write) parts;
        unmasked banks could union two written objects through those
        extensions. The context uses the *grown* ROI, which keeps the
        object's `label`, so the mask machinery selects the right object over
        the whole banked region.
        """
        mask_transform = BaseMaskTransform(
            label_zarr_array=self._input._label.zarr_array,
            label_dimensions=self._input._label.dimensions,
            axes_order=self._axes_order,
            target_dimensions=self._output.dimensions,
            fill_value=0,
        )
        ctx = setup_io_pipe(
            zarr_array=self._output.zarr_array,
            dimensions=self._output.dimensions,
            axes_order=self._axes_order,
            remove_channel_selection=True,
            roi=self._read_roi(roi),
        )
        return _MaskedBankMask(mask_transform, ctx)

    def build_numpy_setter(self, roi: Roi):
        return self._wrap_for_stitch(
            self._wrap_setter(
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
        self._require_stitchless_dask()
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
