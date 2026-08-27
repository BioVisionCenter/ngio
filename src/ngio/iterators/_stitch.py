"""Stitching a tiled segmentation: id blocks, per-tile banks, seam resolve.

Works on any ROI list — regular grids with a halo, overlapping-FOV microscope
layouts, ragged ROI tables. Each tile offsets its ids into its own block and
*banks* its full grown prediction into its own scratch array (one per tile,
global origin recorded in the attrs), so banking is conflict-free in any
topology: no two tiles ever share a scratch write. The resolve step then
compares every pair of banks that overlap in pixel space, unions the ids that
agree (IoU over the shared pixels), and relabels the output in place.

Where two tiles wrote the same *output* pixels and their objects do not
match, `StitchConfig.write_order` picks the seam owner: schedule-defined
by default (`"any"`), the later tile under `"roi"`. Unification itself is
exact either way — the banks are per-tile and order-free.
"""

import threading
import uuid
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import zarr

from ngio.common._label_ops import (
    check_offset_fits,
    chunk_selections,
    offset_labels,
    overlap_iou,
    relabel,
    relabel_sequential,
)
from ngio.common._roi import Roi
from ngio.common._union_find import UnionFind
from ngio.images._abstract_image import AbstractImage
from ngio.io_pipes._io_pipes_types import DataSetterProtocol
from ngio.io_pipes._ops_axes import AxesOps, set_as_numpy_axes_ops
from ngio.io_pipes._ops_slices import SlicingOps
from ngio.iterators._mappers import WriteOrder, validate_write_order
from ngio.iterators._stitch_geometry import (
    SpanBox,
    intersection_box,
    labels_compatible,
    sweep_pairs,
    tile_extents,
    touching_unstitched_axes,
)
from ngio.utils import (
    NgioFileNotFoundError,
    NgioStore,
    NgioUserWarning,
    NgioValueError,
    StoreOrGroup,
    open_group_wrapper,
)
from ngio.utils._warnings import stacklevel_of_first_caller

_SCRATCH_GROUP = "_ngio_stitch"
_LAYOUT = "per-roi/1"


class SeamMatcherProtocol(Protocol):
    """Decides which ids of two overlapping banked predictions are one object.

    Called only at the resolve (the gather step, on the calling thread — no
    pickling constraints), once per overlapping tile pair, with the two
    tiles' banked label patches reconstructed over their shared region:
    aligned, same shape, in the label's own axes order. Ids are as they
    appear in the patches (each tile's block-offset ids). Return the
    `(id_in_a, id_in_b)` pairs that are the same object; the framework owns
    which tile pairs are compared and over which regions, the banking and
    scratch lifecycle, the union-find, and the global relabel.

    Must be deterministic — a pure function of the two patches — or a
    distributed gather is no longer bit-identical to a serial run.
    Background `0` is never pairable, and every returned id must appear in
    its patch; violations raise at the resolve.
    """

    def __call__(
        self, patch_a: np.ndarray, patch_b: np.ndarray
    ) -> list[tuple[int, int]]:
        """Return the `(id_in_a, id_in_b)` same-object pairs."""
        ...


@dataclass(frozen=True)
class IouSeamMatcher:
    """The default seam criterion: IoU over the shared pixels, thresholded.

    `overlap_iou` counts only pixels labelled on *both* sides, so background
    (and a masked tile's outside) never enters a pair and can never
    displace a written id.
    """

    iou_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Validate the threshold."""
        if not 0.0 < self.iou_threshold <= 1.0:
            raise NgioValueError(
                f"iou_threshold must be in (0, 1], got {self.iou_threshold}. "
                "Zero would merge any pair that shares a single pixel."
            )

    def __call__(
        self, patch_a: np.ndarray, patch_b: np.ndarray
    ) -> list[tuple[int, int]]:
        """The pairs whose IoU over the shared pixels meets the threshold."""
        return [
            pair
            for pair, score in overlap_iou(patch_a, patch_b).items()
            if score >= self.iou_threshold
        ]


def _validated_seam_pairs(
    pairs: list[tuple[int, int]], left: np.ndarray, right: np.ndarray
) -> list[tuple[int, int]]:
    """Enforce the seam-matcher invariants before anything is unioned.

    Unioning background with a written id would corrupt the label globally,
    and a foreign id would relabel an object the pair never saw.
    """
    left_ids = set(np.unique(left).tolist())
    right_ids = set(np.unique(right).tolist())
    for mine, theirs in pairs:
        if mine == 0 or theirs == 0:
            raise NgioValueError(
                f"The seam matcher paired background: ({mine}, {theirs}). "
                "`0` is not an object and can never merge."
            )
        if mine not in left_ids or theirs not in right_ids:
            raise NgioValueError(
                f"The seam matcher returned ({mine}, {theirs}), but the id "
                "is absent from its patch: every pair must reference ids "
                "that appear in the patches it was given."
            )
    return pairs


@dataclass(frozen=True, kw_only=True)
class StitchConfig:
    """How to stitch a tiled segmentation.

    Attributes:
        block_size: Ids reserved per tile. Must exceed the largest label a
            single tile can produce, or ids spill into the next tile's block.
        iou_threshold: How much two tiles' predictions must agree over their
            shared pixels before their ids are called the same object. Low
            values merge eagerly; the default errs towards leaving an object
            split, which is fixable downstream, rather than merging two that
            are not. Configures the default `IouSeamMatcher`; ignored when
            `seam_matcher` is set.
        seam_matcher: Replaces the default pair criterion — see
            `SeamMatcherProtocol`. Like the measurement function, a custom
            matcher is not part of the plan fingerprint: declare the
            identical one on every phase of a distributed run.
        compact: Renumber the surviving ids to a dense `1..N` at the end.
            The renumbering walks the **whole** label, so objects outside the
            iterated ROIs get new ids too — anything keyed by the old ids
            (feature tables, masking ROI tables) is invalidated. A run that
            renumbers such pre-existing objects warns. Use `compact=False`
            and reconcile ids yourself to keep external references valid.
        scratch_store: Where to keep the per-tile banks while the map runs.
            `None` puts them in a transient group inside the output label,
            which works under every mapper. A `MemoryStore` avoids touching
            the output store — but an in-memory scratch cannot cross a
            process boundary, so `ProcessMapper` refuses it.
        write_order: Who owns the seams of *unmatched* objects — object
            unification is exact either way. `"any"` (the default) is
            schedule-defined; `"roi"` gives them to the later tile,
            reproducibly, at up to 2.5x cost on parallel overlapping
            tilings.
    """

    block_size: int = 10_000
    iou_threshold: float = 0.3
    seam_matcher: SeamMatcherProtocol | None = None
    compact: bool = True
    scratch_store: StoreOrGroup | None = None
    write_order: WriteOrder = "any"

    def __post_init__(self) -> None:
        """Validate the configuration."""
        validate_write_order(self.write_order)
        if self.block_size <= 0:
            raise NgioValueError(f"block_size must be > 0, got {self.block_size}.")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise NgioValueError(
                f"iou_threshold must be in (0, 1], got {self.iou_threshold}. "
                "Zero would merge any pair that shares a single pixel."
            )

    def matcher(self) -> SeamMatcherProtocol:
        """The seam matcher the resolve runs; the IoU default unless swapped."""
        return self.seam_matcher or IouSeamMatcher(self.iou_threshold)


@dataclass(frozen=True)
class _TileWork:
    """What one tile contributes: its id block and its pixel-space extents."""

    index: int
    offset: int
    core: SpanBox
    grown: SpanBox


def _scratch_group(
    output: AbstractImage, scratch_store: StoreOrGroup | None
) -> zarr.Group:
    """The scratch group, created inside the label or opened from the store."""
    if scratch_store is None:
        return output._group_handler.get_group(_SCRATCH_GROUP, create_mode=True)
    return open_group_wrapper(scratch_store, mode="a")


def read_scratch_attrs(
    output: AbstractImage, scratch_store: StoreOrGroup | None
) -> dict[str, Any] | None:
    """The scratch group's attributes, or `None` when no scratch exists."""
    try:
        if scratch_store is None:
            group = output._group_handler.get_group(_SCRATCH_GROUP)
        else:
            group = open_group_wrapper(scratch_store, mode="r")
    except NgioFileNotFoundError:
        # Only "no scratch" may answer `None` — the caller reacts by creating
        # a fresh scratch, which would wipe banks other jobs already wrote,
        # so an unreadable scratch must raise instead.
        return None
    return dict(group.attrs)


def _delete_scratch(output: AbstractImage, scratch_store: StoreOrGroup | None) -> None:
    """Remove the scratch group, if it is still there.

    Only the group ngio created inside the label is deleted. A store the
    caller supplied is theirs, so its arrays are emptied of meaning but the
    store itself is left for them to dispose of.
    """
    if scratch_store is not None:
        return
    try:
        output._group_handler.delete_group(_SCRATCH_GROUP)
    except (KeyError, FileNotFoundError):
        pass


class ScratchBanks:
    """The per-tile scratch arrays under one root group.

    One array per tile, named `roi_{index:06d}`, shaped to the tile's grown
    extents, with the tile's global origin, id offset, and the run's id in
    the attrs. The `run_id` is the staleness guard: a caller-supplied
    `scratch_store` is never wiped, so a leftover bank from an earlier run
    must be recognisable — a bank whose `run_id` does not match the root's is
    treated as missing, never silently read.
    """

    def __init__(self, group: zarr.Group, run_id: str, label_array: zarr.Array):
        self._group = group
        self._run_id = run_id
        self._label_array = label_array
        #: Whether this handle *created* the root (vs opened a prepared one) —
        #: an error path may clean up what it created, never what it opened.
        self.created = False

    @classmethod
    def create(
        cls,
        output: AbstractImage,
        scratch_store: StoreOrGroup | None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> "ScratchBanks":
        """Create the scratch root fresh, wiping any leftovers first.

        Creation is the one moment a run may destroy scratch state: stale or
        corrupted banks from a crashed, interrupted, or superseded run are
        wiped here (or, on a caller's store, invalidated by the new
        `run_id`), so nothing older than this call can ever reach a resolve.
        """
        _delete_scratch(output, scratch_store)
        group = _scratch_group(output, scratch_store)
        run_id = uuid.uuid4().hex
        # `resolving` is cleared explicitly: on a caller-supplied store the
        # delete above is a no-op and `attrs.update` merges, so a marker left
        # by a crashed resolve would otherwise refuse this fresh run forever.
        group.attrs.update(
            {
                "layout": _LAYOUT,
                "run_id": run_id,
                "resolving": False,
                **(extra_attrs or {}),
            }
        )
        banks = cls(group, run_id, output.zarr_array)
        banks.created = True
        return banks

    @classmethod
    def open(
        cls, output: AbstractImage, scratch_store: StoreOrGroup | None
    ) -> "ScratchBanks":
        """Open an existing scratch root without touching its content.

        The job and gather steps of a distributed run come through here: the
        banks other jobs wrote must survive, so nothing is created or reset.
        """
        attrs = read_scratch_attrs(output, scratch_store)
        if attrs is None:
            raise NgioValueError(
                "The stitch scratch does not exist: a distributed stitched "
                "run needs `prepare_jobs(n_jobs)` to create it before any "
                "job runs."
            )
        if attrs.get("layout") != _LAYOUT or not isinstance(attrs.get("run_id"), str):
            raise NgioValueError(
                "The stitch scratch was written by an incompatible ngio "
                f"version (layout {attrs.get('layout')!r}); re-run "
                "`prepare_jobs(n_jobs)` to recreate it."
            )
        if attrs.get("resolving"):
            raise NgioValueError(
                "A previous `finalize()` was interrupted while compacting the "
                "label in place, so the label may hold a mix of renumbered "
                "and original ids. Refusing to reuse these banks: resolving "
                "against a half-compacted label silently splits objects. "
                "Re-run `prepare_jobs(n_jobs)` and the jobs to regenerate the "
                "label, then finalize again."
            )
        group = _scratch_group(output, scratch_store)
        return cls(group, attrs["run_id"], output.zarr_array)

    @property
    def is_memory(self) -> bool:
        """Whether the banks live on an in-memory store."""
        return NgioStore.ensure(self._group.store).store_type == "memory"

    @staticmethod
    def _key(index: int) -> str:
        return f"roi_{index:06d}"

    def write(self, work: _TileWork, disk_patch: np.ndarray) -> None:
        """Bank one tile's (offset, disk-oriented) grown prediction.

        Creates the tile's array — single-writer by construction, and
        `overwrite=True` keeps a re-run of the same tile idempotent.
        """
        shape = tuple(stop - start for start, stop in work.grown)
        if tuple(disk_patch.shape) != shape:
            raise NgioValueError(
                f"Tile {work.index} banked a patch of shape "
                f"{tuple(disk_patch.shape)} where its grown region is "
                f"{shape}. The function must return the grown region it was "
                "handed, unrescaled."
            )
        chunks = tuple(
            min(chunk, size)
            for chunk, size in zip(self._label_array.chunks, shape, strict=True)
        )
        array = self._group.create_array(
            name=self._key(work.index),
            shape=shape,
            dtype=self._label_array.dtype,
            chunks=chunks,
            overwrite=True,
        )
        array[...] = disk_patch
        # The attrs stamp is the commit marker, so it lands after the payload:
        # a write killed mid-payload leaves an attrs-less array, which
        # `_open_valid` reads as missing instead of as a valid zero bank.
        array.attrs.update(
            {
                "index": work.index,
                "offset": work.offset,
                "origin": [start for start, _ in work.grown],
                "run_id": self._run_id,
            }
        )

    def _open_valid(self, work: _TileWork) -> zarr.Array | None:
        """The tile's bank, or `None` when absent, stale, or mismatched.

        Deliberately uncached: a cached handle would keep answering after
        another run wiped the scratch, reading back zeros where the `run_id`
        guard should have said "missing".
        """
        try:
            member = self._group[self._key(work.index)]
        except KeyError:
            return None
        if not isinstance(member, zarr.Array):
            return None
        attrs = dict(member.attrs)
        if (
            attrs.get("run_id") != self._run_id
            or attrs.get("offset") != work.offset
            or attrs.get("origin") != [start for start, _ in work.grown]
        ):
            return None
        return member

    def missing(self, works: Sequence[_TileWork]) -> list[int]:
        """Indices of tiles with no valid bank."""
        return [work.index for work in works if self._open_valid(work) is None]

    def set_resolving(self, resolving: bool) -> None:
        """Mark (or unmark) the in-place compaction walk as underway.

        Stamped before `relabel_sequential` and cleared after it: the walk is
        the one non-idempotent phase, so a scratch found with the marker set
        means a crashed resolve left the label half-compacted.
        """
        self._group.attrs.update({"resolving": resolving})

    def read(self, work: _TileWork, box: SpanBox) -> np.ndarray:
        """The tile's banked prediction over `box` (global pixel coords)."""
        array = self._open_valid(work)
        if array is None:
            raise NgioValueError(
                f"Tile {work.index} has no valid bank; the run is incomplete."
            )
        selection = tuple(
            slice(start - origin, stop - origin)
            for (start, stop), (origin, _) in zip(box, work.grown, strict=True)
        )
        return np.asarray(array[selection])


class StitchPlan:
    """Everything the stitch needs, derived once from the iterator's ROIs.

    Built lazily on first use, because it needs the whole ROI list — a single
    tile does not carry enough information to know its overlaps.
    """

    def __init__(
        self,
        *,
        config: StitchConfig,
        output: AbstractImage,
        rois: Sequence[Roi],
        halo: dict[str, int],
        read_roi: Any,
        scratch_factory: Callable[[], ScratchBanks] | None = None,
        same_label_only: bool = False,
    ) -> None:
        """Place every tile, reserve its id block, and find the overlapping pairs.

        The scratch is opened (or created) on first access rather than here:
        the plan's geometry is needed before any storage should be touched,
        and in a distributed run which banks to use (create fresh vs open the
        prepared root) is the iterator's decision, injected via
        `scratch_factory`. With no factory, first access creates a fresh
        scratch — the standalone `map` behaviour.
        """
        self._config = config
        self._output = output
        self._halo = halo
        self._array_axes = tuple(output.dimensions.axes_handler.axes_names)

        names = [roi.get_name() for roi in rois]
        seen: dict[str, int] = {}
        for index, name in enumerate(names):
            other = seen.setdefault(name, index)
            if other != index:
                raise NgioValueError(
                    f"Two tiles share the name {name!r}; the stitch plan is "
                    "keyed by name, so duplicates would silently shadow each "
                    "other. Give the ROIs unique names."
                )

        info = np.iinfo(output.zarr_array.dtype)
        top_id = (len(rois) + 1) * config.block_size - 1
        if top_id > int(info.max):
            raise NgioValueError(
                f"{len(rois)} tiles at block_size={config.block_size} reach "
                f"id {top_id}, past what {output.zarr_array.dtype} can hold "
                f"({int(info.max)}). Lower block_size or widen the label "
                "dtype."
            )

        extents = tile_extents(rois, output, read_roi, self._array_axes)
        self._works = [
            _TileWork(
                index=extent.index,
                offset=(extent.index + 1) * config.block_size,
                core=extent.core,
                grown=extent.grown,
            )
            for extent in extents
        ]
        self._work_by_name = dict(zip(names, self._works, strict=True))
        self._pairs = [
            (i, j)
            for i, j in sweep_pairs([extent.grown for extent in extents])
            if labels_compatible(extents[i], extents[j], same_label_only)
        ]

        if len(rois) > 1 and not self._pairs:
            # Under the label restriction, one tile per object is a normal
            # configuration (nothing was split), not a misconfiguration.
            singleton_objects = same_label_only and all(
                extent.label is not None for extent in extents
            )
            if singleton_objects:
                counts: dict[int, int] = {}
                for extent in extents:
                    assert extent.label is not None
                    counts[extent.label] = counts.get(extent.label, 0) + 1
                singleton_objects = all(count < 2 for count in counts.values())
            if singleton_objects:
                warnings.warn(
                    "Stitching is a per-object no-op: every object fits one "
                    "tile, so nothing was split. Ids are still offset per tile"
                    + (" and compacted at the end." if config.compact else "."),
                    NgioUserWarning,
                    stacklevel=stacklevel_of_first_caller(),
                )
            else:
                core_pairs = [
                    (i, j)
                    for i, j in sweep_pairs([extent.core for extent in extents])
                    if labels_compatible(extents[i], extents[j], same_label_only)
                ]
                if not halo and not core_pairs:
                    raise NgioValueError(
                        "Stitching needs a halo or overlapping ROIs: with "
                        "neither, no two tiles ever predict the same pixel and "
                        "there is no agreement to measure. Call "
                        "`with_halo(...)`, or tile with overlap."
                    )
                warnings.warn(
                    "Stitching found no overlapping tiles: nothing will be "
                    "merged. Ids are still offset per tile"
                    + (" and compacted at the end." if config.compact else "."),
                    NgioUserWarning,
                    stacklevel=stacklevel_of_first_caller(),
                )

        unhaloed = [
            axis for axis, name in enumerate(self._array_axes) if name not in halo
        ]
        for axis in touching_unstitched_axes(extents, unhaloed, same_label_only):
            warnings.warn(
                f"Tiles touch along '{self._array_axes[axis]}', which has no "
                "halo: seams along it will not be stitched. Add the axis to "
                "`with_halo(...)` to stitch across it.",
                NgioUserWarning,
                stacklevel=stacklevel_of_first_caller(),
            )

        self._scratch_factory = scratch_factory
        self._banks: ScratchBanks | None = None
        self._scratch_lock: threading.Lock | None = threading.Lock()

    @property
    def banks(self) -> ScratchBanks:
        """The scratch banks, resolved on first access (thread-safe)."""
        if self._banks is None:
            lock = self._scratch_lock
            # `__getstate__` nulls the lock, but `__setstate__` always
            # restores one; only a hand-built state could get here.
            assert lock is not None
            with lock:
                if self._banks is None:
                    if self._scratch_factory is None:
                        self._banks = ScratchBanks.create(
                            self._output, self._config.scratch_store
                        )
                    else:
                        self._banks = self._scratch_factory()
        return self._banks

    def __getstate__(self) -> dict:
        """Ship the resolved scratch reference across a process boundary.

        The parent must resolve the scratch before pickling (the writing
        verbs do) — resolving lazily here would make pickling write to the
        store. The unpicklable factory and process-local lock are dropped,
        as is everything a worker does not use: the output image (its
        handler carries a file lock) and the plan geometry (`_works`/
        `_pairs` would dominate `ProcessMapper` IPC). A worker only banks;
        `resolve`/`cleanup` refuse on a worker copy.
        """
        if self._banks is None:
            raise NgioValueError(
                "This stitch plan's scratch is not resolved yet: pickling "
                "would have to create it as a side effect. Run the plan "
                "through a writing verb (which resolves it up front), or "
                "touch `plan.banks` before pickling."
            )
        state = self.__dict__.copy()
        state["_scratch_factory"] = None
        state["_scratch_lock"] = None
        state["_output"] = None
        state["_works"] = []
        state["_pairs"] = []
        state["_work_by_name"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._scratch_lock = threading.Lock()

    def work_for(self, roi: Roi) -> _TileWork:
        """The tile record for `roi`."""
        work = self._work_by_name.get(roi.get_name())
        if work is None:
            raise NgioValueError(
                f"Tile {roi.get_name()!r} is not part of the stitch plan; the "
                "ROI list changed after the plan was built."
            )
        return work

    def bank(self, work: _TileWork, disk_patch: np.ndarray) -> None:
        """Bank one tile's offset prediction (disk-oriented)."""
        self.banks.write(work, disk_patch)

    def _require_output(self) -> AbstractImage:
        if self._output is None:
            raise NgioValueError(
                "This stitch plan was shipped to a worker and only banks "
                "there; resolve and cleanup belong to the parent's plan."
            )
        return self._output

    def resolve(self) -> None:
        """Compare every overlapping pair of banks, union, relabel in place."""
        output = self._require_output()
        attrs = read_scratch_attrs(output, self._config.scratch_store)
        if self._banks is None and attrs is None:
            # Never *create* a scratch on the resolve path: with nothing
            # banked there is nothing to resolve, and creation would leave a
            # stray `_ngio_stitch` group beside the label.
            raise NgioValueError(
                "Nothing to resolve: no stitched map has banked predictions "
                "here — the run never mapped, or it already finalized."
            )
        if attrs is not None and attrs.get("resolving"):
            # Checked before `self.banks`: the standalone factory path would
            # otherwise wipe the crashed run's evidence before this fires.
            raise NgioValueError(
                "A previous `finalize()` was interrupted while compacting the "
                "label in place, so the label may hold a mix of renumbered "
                "and original ids, and resolving again would silently split "
                "objects. Re-run the writing verb (or `prepare_jobs(n_jobs)` "
                "and the jobs) to regenerate the label, then finalize again."
            )
        banks = self.banks
        missing = banks.missing(self._works)
        if missing:
            if banks.created:
                # This handle just wiped/created the scratch, so nothing in
                # it is another job's work; remove it rather than leave an
                # empty group beside the label.
                self.cleanup()
            raise NgioValueError(
                f"The stitched run is incomplete: tile(s) {missing} never "
                "banked a prediction (or banked one from a different run). "
                "Run the missing jobs, then finalize again."
            )

        matcher = self._config.matcher()
        union = UnionFind[int]()
        for i, j in self._pairs:
            box = intersection_box(self._works[i].grown, self._works[j].grown)
            assert box is not None  # pairs come from the same boxes
            left = banks.read(self._works[i], box)
            right = banks.read(self._works[j], box)
            for mine, theirs in _validated_seam_pairs(
                matcher(left, right), left, right
            ):
                union.union(mine, theirs)

        mapping = union.resolve()
        label_array = output.zarr_array
        if self._config.compact:
            banks.set_resolving(True)
            # One pass: canonicalize and renumber together. Collecting the ids
            # first would mean reading the whole label twice.
            dense = relabel_sequential(label_array, mapping)
            banks.set_resolving(False)
            # Ids this run wrote all carry a tile's offset, so anything outside
            # the offset range pre-existed the run — and just got a new id.
            # (An original id that happens to fall inside the range is missed;
            # the common case, small ids in an existing label, is caught.)
            block = self._config.block_size
            top = (len(self._works) + 1) * block
            foreign = sum(1 for root in dense if not block < root < top)
            if foreign:
                warnings.warn(
                    f"compact=True renumbered {foreign} object id(s) this run "
                    "never touched: the renumbering walks the whole label, so "
                    "anything keyed by the old ids (feature tables, masking "
                    "ROI tables) is now stale. Use `compact=False` to keep "
                    "ids outside the iterated ROIs unchanged.",
                    NgioUserWarning,
                    stacklevel=stacklevel_of_first_caller(),
                )
        else:
            _apply_relabel(label_array, mapping)
        self.cleanup()

    @property
    def created_banks(self) -> bool:
        """Whether this run created the scratch root (vs opened a prepared one).

        The predicate for a failure-path cleanup: a run that only *opened* the
        scratch — a job, a resumed run, the gather step — must never delete it,
        because it holds the banks every other job wrote.
        """
        return self._banks is not None and self._banks.created

    def cleanup(self) -> None:
        """Delete the scratch group ngio created."""
        _delete_scratch(self._require_output(), self._config.scratch_store)


def _apply_relabel(label_array: zarr.Array, mapping: dict[int, int]) -> None:
    """Rewrite ids in place, one chunk at a time."""
    if not mapping:
        return
    for selection in chunk_selections(label_array):
        block = np.asarray(label_array[selection])
        remapped = relabel(block, mapping)
        if not np.array_equal(remapped, block):
            label_array[selection] = remapped


class StitchingSetter:
    """Offsets a tile's ids, banks its grown prediction, then writes as usual.

    Sits *outside* the halo crop so it sees the grown patch — the halo
    context is the whole point, and the crop is what throws it away. The id
    offset is applied here rather than by a transform for the same reason: a
    transform on the setter's chain would only ever see the cropped core, and
    then the bank and the core would disagree about what an object is called.
    """

    def __init__(
        self,
        setter: DataSetterProtocol[np.ndarray],
        plan: StitchPlan,
        roi: Roi,
        bank_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Wrap `setter` for the tile covering `roi`.

        `bank_transform` reshapes what is *banked*, never what is written —
        the masked iterator uses it to zero outside-mask pixels, so the
        banked opinion is exactly the writable one. Must be picklable (a
        named callable, not a lambda): `ProcessMapper` pickles the setters.
        """
        self._setter = setter
        self._plan = plan
        self._work = plan.work_for(roi)
        self._bank_transform = bank_transform

    def __repr__(self) -> str:
        return f"StitchingSetter({self._setter!r}, offset={self._work.offset})"

    @property
    def zarr_array(self) -> zarr.Array:
        """The wrapped setter's target array."""
        return self._setter.zarr_array

    @property
    def slicing_ops(self) -> SlicingOps:
        """The wrapped setter's slicing — the core, as the mappers must see it."""
        return self._setter.slicing_ops

    @property
    def axes_ops(self) -> AxesOps:
        """The wrapped setter's axes operations."""
        return self._setter.axes_ops

    @property
    def transforms(self):
        """The wrapped setter's transform chain."""
        return self._setter.transforms

    @property
    def roi(self) -> Roi:
        """The core ROI being written."""
        return self._setter.roi

    def __getstate__(self) -> dict:
        """Refuse to cross a process boundary with an in-memory scratch.

        `ProcessMapper` pickles the units, and a `MemoryStore` pickles by
        *value*: each worker would bank into a private copy and the parent
        would resolve against nothing. The mappers' own store check only
        inspects the setter's target array, so it cannot see this — but
        pickling is exactly the moment it would go wrong, so the refusal
        belongs here.
        """
        if self._plan.banks.is_memory:
            raise NgioValueError(
                "A stitch scratch on an in-memory store cannot be used with "
                "ProcessMapper: each worker would bank into its own private "
                "copy and the predictions would be lost. Pass a persistent "
                "`scratch_store` to StitchConfig (or leave it unset to use "
                "the output's own store), or map with ThreadedMapper."
            )
        return self.__dict__

    def __call__(self, patch: np.ndarray) -> None:
        return self.set(patch)

    def set(self, patch: np.ndarray) -> None:
        """Offset the grown patch, hand it on, then bank it.

        Core write first: a run killed between the two leaves the tile
        written but unbanked, which `resolve()` refuses loudly by name.
        Banking first left the opposite window — a valid-looking bank and a
        tile missing from the output, silently absorbed by the gather.
        Re-running the tile is then the recovery, with the caveats any job
        re-run already has (order-dependent `merge=` rules re-apply, and
        seam ownership under `write_order="any"` may shift).
        """
        check_offset_fits(patch, self._work.offset, self._plan._config.block_size)
        offset = offset_labels(patch, self._work.offset)
        banked = (
            offset if self._bank_transform is None else self._bank_transform(offset)
        )
        self._setter.set(offset)
        self._plan.bank(
            self._work,
            set_as_numpy_axes_ops(array=banked, axes_ops=self._setter.axes_ops),
        )
