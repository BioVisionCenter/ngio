from collections.abc import Callable, Sequence
from typing import Self

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
from ngio.io_pipes._merge_policy import MergeInput, resolve_merge
from ngio.io_pipes._ops_slices_utils import check_if_regions_overlap
from ngio.iterators._abstract_iterator import OverlapPolicy, WritingIteratorBuilder
from ngio.iterators._mappers import MapperProtocol, WriteOrder, validate_write_order
from ngio.iterators._stitch import (
    IouSeamMatcher,
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


class SegmentationIterator(WritingIteratorBuilder[np.ndarray, da.Array, None]):
    """Segment an image region by region into a label.

    Reads each region from the input image and writes the function's label
    patch to the output label. With `with_stitch(...)` objects split across
    region boundaries are resolved into one id at the gather — any ROI list
    works: grids with a halo, overlapping FOV layouts, ragged tables. See
    `StitchConfig`. Regions whose write footprints overlap need a declared
    resolution — `with_stitch(...)` or `on_overlap(...)` — or the writing
    verbs refuse.
    """

    # Class-level defaults so subclasses that write their own `__init__` are
    # simply not stitching, rather than missing an attribute the inherited
    # `finalize` reads.
    _stitch: StitchConfig | None = None
    _stitch_plan: StitchPlan | None = None
    _on_overlap: OverlapPolicy | None = None
    _write_order: WriteOrder = "any"
    # `_stitch_plan` is derived per-ROI-list, so deliberately not carried.
    _chain_state_attrs: tuple[str, ...] = ("_stitch", "_on_overlap", "_write_order")
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
    ) -> None:
        """Segment `input_image` region by region into `output_label`.

        Args:
            input_image: The image to segment.
            output_label: The label the segmentation is written to.
            channel_selection: Restrict the image reads to these channels.
            axes_order: Axes order of the patches handed to the function.
            input_transforms: Transforms applied to each image patch.
            output_transforms: Transforms applied to each predicted patch
                before the write.
            consolidation_mode: How to build the output pyramid after
                iteration, see `Label.consolidate`.
        """
        self._input = input_image
        self._output = output_label
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()
        self._consolidation_mode = consolidation_mode

        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

        self._input.require_dimensions_match(self._output, allow_singleton=False)

    def _get_init_kwargs(self) -> dict:
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

    def with_stitch(self, config: StitchConfig | None = None) -> Self:
        """Declare the stitch: split objects become one id at the gather.

        Objects cut by a region boundary are resolved into one id when the
        run finalizes — any ROI list works: grids with a halo, overlapping
        FOV layouts, ragged tables. Needs a halo (`with_halo`) or
        overlapping ROIs: the evidence is overlap between neighbouring
        predictions. Declare before `for_job`; `None` uses the
        `StitchConfig` defaults. Refuses when `on_overlap` is declared —
        a merge policy would make the disk diverge from the banked
        predictions, so the resolve would relabel garbage.

        Args:
            config: The stitch configuration, `None` for the defaults.
        """
        resolved = config if config is not None else StitchConfig()
        if self._on_overlap is not None:
            raise NgioValueError(
                "with_stitch and on_overlap cannot be combined: a merge "
                "policy would make the written labels diverge from the "
                "banked predictions the resolve compares. The stitch "
                "already owns the contested pixels (deterministic write "
                "order) — drop the `on_overlap` declaration."
            )
        _require_no_unique_labels_with_stitch(resolved, self._output_transforms)
        new_instance = self._new_from_rois(self.rois)
        new_instance._stitch = resolved
        return new_instance

    def on_overlap(
        self, policy: OverlapPolicy, *, write_order: WriteOrder = "any"
    ) -> Self:
        """Declare how contested write pixels are resolved.

        `"last"` is last-writer-wins — an explicit acknowledgment, no merge
        is performed. Any `merge=` policy (`"max"`/`"min"`/`"sum"` are
        order-independent; `"keep_nonzero"`; a merge function; a
        `MergePolicy`) combines each write with what is on disk — note it
        applies to *every* write, so pre-existing content participates
        too. Declare before `for_job`. Refuses when `with_stitch` is
        declared (the stitch owns the contested pixels).

        Args:
            policy: `"last"`, or anything the write path's `merge=` takes.
            write_order: Who wins a contested pixel. `"any"` (the default)
                is schedule-defined — deterministic per version, fastest.
                `"roi"` makes the later ROI win — reproducible across
                versions and bit-identical to the manual `iter` loop, at up
                to 2.5x cost on parallel overlapping tilings. Irrelevant
                under an order-independent merge.
        """
        if policy != "last":
            resolve_merge(policy)  # refuse an invalid rule at declaration
        if self._stitch is not None:
            raise NgioValueError(
                "on_overlap and with_stitch cannot be combined: the stitch "
                "already owns the contested pixels (deterministic write "
                "order), and a merge policy would make the written labels "
                "diverge from the banked predictions. Drop one of the two."
            )
        new_instance = self._new_from_rois(self.rois)
        new_instance._on_overlap = policy
        new_instance._write_order = validate_write_order(write_order)
        return new_instance

    def _units_write_order(self) -> WriteOrder:
        """The declared write order; a stitch's config takes precedence."""
        if self._stitch is not None:
            return self._stitch.write_order
        return self._write_order

    def _overlap_merge(self) -> MergeInput | None:
        """The `merge=` the setters carry; `None` for undeclared or "last"."""
        return None if self._on_overlap in (None, "last") else self._on_overlap

    def _validate_write_plan(self) -> None:
        """Refuse undeclared overlapping write footprints.

        Without a declared resolution the later tile silently overwrites
        the earlier one's labels — deterministic, but almost never the
        intent for a segmentation. Measured pixel-exactly on the setters'
        write regions (halo margins are cropped before the write, so a
        halo never triggers this; sharing a chunk without sharing pixels
        does not either).
        """
        if self._stitch is not None or self._on_overlap is not None:
            return
        if len(self.rois) < 2:
            return
        slicing_tuples = (
            self.build_numpy_setter(roi).slicing_ops.normalized_slicing_tuple
            for roi in self.rois
        )
        if check_if_regions_overlap(slicing_tuples):
            raise NgioValueError(
                "Segmentation regions overlap where they write: without a "
                "declared resolution, the later tile silently overwrites "
                "the earlier one's labels. Declare it: `.with_stitch(...)` "
                "to merge objects split across the seams, or "
                "`.on_overlap('last')` (or a merge rule) to accept a "
                "deterministic overwrite."
            )

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
        # The default matcher's repr pins its threshold, so a drift between
        # phases still refuses; a custom matcher, like the measurement
        # function, is not fingerprintable — only its type is recorded.
        matcher = self._stitch.matcher()
        matcher_term = (
            f"stitch.matcher={matcher!r}"
            if isinstance(matcher, IouSeamMatcher)
            else f"stitch.matcher={type(matcher).__qualname__}"
        )
        return (
            f"stitch.block_size={self._stitch.block_size}",
            matcher_term,
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
                    merge=self._overlap_merge(),
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
                "map()/iter(data_mode='numpy'), or drop the `with_stitch` "
                "declaration."
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
                merge=self._overlap_merge(),
            ),
            roi,
        )

    def map(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        mapper: MapperProtocol[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """See `WritingIteratorBuilder.map`; also cleans up on failure.

        A failed standalone run cannot be resolved, so the stitch scratch
        arrays are deleted rather than left as a stray `_ngio_stitch` group
        beside the resolution levels. Cleanup happens only when this run
        *created* the scratch: a partition slice, a resumed run, or the
        gather step opened a prepared root that holds the banks every other
        job wrote, and one failure must not destroy them — re-running is
        idempotent (banks rewrite, the id offsets are derived, not counted).
        The already-written tiles stay in every case.
        """
        if self._stitch is None:
            return super().map(func, mapper=mapper)
        # Build the plan (and let its validation warnings and errors fire)
        # before any tile runs: lazily it would surface mid-run, from a
        # worker, after some tiles have already written.
        plan = self._stitching_plan()
        # Resolve the scratch in the parent, before any unit is pickled or
        # run — never as a side effect of pickling.
        _ = plan.banks
        if self._partition is not None:
            return super().map(func, mapper=mapper)
        try:
            return super().map(func, mapper=mapper)
        except BaseException:
            if self._stitch_plan is not None and self._stitch_plan.created_banks:
                self._stitch_plan.cleanup()
                self._stitch_plan = None
            raise

    def segment(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        mapper: MapperProtocol[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Segment every region and write the labels; the topic verb for `map`.

        A serial run finalizes automatically (the stitch resolve included).
        On a `for_job` slice it segments only this job's share; the gather is
        the unrestricted iterator's `finalize()`, once, after all jobs.

        Args:
            func: The segmentation model. Under a parallel mapper it runs on
                worker threads (or processes) and must be safe there.
            mapper: How the units are scheduled; see `map`.
        """
        self.map(func, mapper=mapper)

    def finalize(self) -> None:
        """Resolve the stitch (when configured), then consolidate the pyramid."""
        self._require_unrestricted_finalize()
        # The relabel has to precede consolidation: every pyramid level is
        # derived from level 0, so stitching after would leave them disagreeing.
        if self._stitch is not None:
            # The stitch resolve relabels level 0 wherever the union-find
            # reached, not just under the written ROIs — only a full rebuild
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
    `with_stitch(...)` and a tiling (`by_grid` + `with_halo`), sub-objects
    split by a tile boundary *within one mask* merge into one id; tiles of
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

    def _units_write_order(self) -> WriteOrder:
        """Always `"any"`: mask-protected writes never contest a pixel."""
        return "any"

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
        """Segment each masked object's box from `input_image` into `output_label`.

        The ROIs come from `input_image`'s masking ROI table: one per object,
        pixels outside the object masked on read and protected on write.

        Args:
            input_image: The masked image to segment; its masking table
                supplies the per-object ROIs.
            output_label: The label the segmentation is written to.
            channel_selection: Restrict the image reads to these channels.
            axes_order: Axes order of the patches handed to the function.
            input_transforms: Transforms applied to each image patch, before
                the mask fill.
            output_transforms: Transforms applied to each predicted patch
                before the write.
            consolidation_mode: How to build the output pyramid after
                iteration, see `Label.consolidate`.
        """
        self._input = input_image
        self._output = output_label

        self._ref_image = input_image
        self._set_rois(input_image._masking_roi_table.rois())
        self._consolidation_mode = consolidation_mode

        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

        self._input.require_dimensions_match(self._output, allow_singleton=False)

    def _get_init_kwargs(self) -> dict:
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

    def on_overlap(
        self, policy: OverlapPolicy, *, write_order: WriteOrder = "roi"
    ) -> Self:
        """Refused: masked writes never contest.

        Each write touches only its own object's pixels (`MaskMerge`
        protects the rest), so there is nothing to declare — and the one
        `merge=` slot already carries the mask protection.
        """
        raise NgioValueError(
            "MaskedSegmentationIterator takes no overlap policy: writes are "
            "mask-protected, so overlapping bounding boxes never contest a "
            "pixel — there is nothing to declare."
        )

    def _validate_write_plan(self) -> None:
        """Masked writes are exempt: bounding boxes overlap by design.

        `MaskMerge` protects everything outside each write's own object,
        so overlapping boxes never contest a pixel.
        """

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
