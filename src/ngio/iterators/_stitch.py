"""Stitching a tiled segmentation into one consistent labelling.

Each tile is segmented independently, so an object crossing a tile boundary
comes out as two objects with two ids. Stitching resolves those into one.

The evidence is **overlap between two independent predictions**, not adjacency
across the cut. That distinction is the whole point: two objects that merely
touch at the boundary are adjacent but do not overlap, and adjacency would merge
them. Getting overlap evidence requires a halo — each tile reads past its own
edge, so it has an opinion about the strip its neighbour owns — and it requires
keeping that opinion, since the tile writes only its core. The scratch arrays
here are where it is kept: one per haloed axis, holding the forward band each
tile predicts inside the next tile's core.

The map phase stays disjoint and parallel throughout; all the reconciling
happens afterwards, in `resolve`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import zarr

from ngio.common._label_ops import (
    chunk_selections,
    check_offset_fits,
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
from ngio.iterators._stitch_geometry import (
    TilePlacement,
    forward_bands,
    neighbours_along,
    shared_band,
    tile_placements,
)
from ngio.utils import NgioStore, NgioValueError, StoreOrGroup, open_group_wrapper

_SCRATCH_GROUP = "_ngio_stitch"


@dataclass(frozen=True)
class StitchConfig:
    """How to stitch a tiled segmentation.

    Attributes:
        block_size: Ids reserved per tile. Must exceed the largest label a
            single tile can produce, or ids spill into the next tile's block.
        min_iou: How much two tiles' predictions must agree over the shared band
            before their ids are called the same object. Low values merge
            eagerly; the default errs towards leaving an object split, which is
            fixable downstream, rather than merging two that are not.
        compact: Renumber the surviving ids to a dense `1..N` at the end.
        scratch_store: Where to keep the halo bands while the map runs. `None`
            puts them in a transient group inside the output label, which works
            under every mapper. A `MemoryStore` is often the better choice —
            labels compress well, so the bands are small, and nothing touches
            the output store — but an in-memory scratch cannot cross a process
            boundary, so `ProcessMapper` refuses it.
    """

    block_size: int = 10_000
    min_iou: float = 0.3
    compact: bool = True
    scratch_store: StoreOrGroup | None = None

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.block_size <= 0:
            raise NgioValueError(f"block_size must be > 0, got {self.block_size}.")
        if not 0.0 < self.min_iou <= 1.0:
            raise NgioValueError(
                f"min_iou must be in (0, 1], got {self.min_iou}. Zero would "
                "merge any pair that shares a single pixel."
            )


@dataclass
class _TileWork:
    """What one tile contributes: its id block, and the bands it predicts."""

    placement: TilePlacement
    offset: int
    grown_origin: dict[str, int]
    bands: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)


class StitchPlan:
    """Everything the stitch needs, derived once from the iterator's ROIs.

    Built lazily on first use, because it needs the whole ROI list — a single
    tile does not carry enough information to place itself on the grid.
    """

    def __init__(
        self,
        *,
        config: StitchConfig,
        output: AbstractImage,
        rois: Sequence[Roi],
        ref_image: AbstractImage,
        halo: dict[str, int],
        read_roi: Any,
    ) -> None:
        """Place every tile, reserve its id block, and open the scratch arrays."""
        if not halo:
            raise NgioValueError(
                "Stitching needs a halo: without one each tile only sees its "
                "own core, so neighbouring tiles never predict the same pixels "
                "and there is no overlap to measure. Call `with_halo(...)` "
                "before `map`."
            )
        self._config = config
        self._output = output
        # Ordered by the array's own axes, not by the halo dict's insertion
        # order: `with_halo` builds its dict x-first, while a (y, x) label is
        # indexed y-first, and a selection built in the wrong order silently
        # reads the transpose of the intended band.
        self._array_axes = tuple(output.dimensions.axes_handler.axes_names)
        self._axes = tuple(name for name in self._array_axes if name in halo)
        self._halo = halo
        self._placements = tile_placements(rois, ref_image, self._axes)

        # Two tiles in one cell means the haloed axes do not distinguish the
        # tiles (e.g. z-tiled with a halo only on y and x). The neighbour map
        # is keyed by cell, so the duplicates would silently shadow each other
        # and whole seams would go unstitched.
        seen_cells: dict[tuple[int, ...], int] = {}
        for placement in self._placements:
            other = seen_cells.setdefault(placement.grid, placement.index)
            if other != placement.index:
                raise NgioValueError(
                    f"Tiles {rois[other].get_name()!r} and "
                    f"{rois[placement.index].get_name()!r} occupy the same "
                    f"stitch grid cell over the haloed axes {self._axes}: they "
                    "differ only along an axis with no halo. Stitching needs "
                    "the halo to cover every tiled axis — add the missing axis "
                    "to `with_halo(...)`."
                )

        self._work: dict[str, _TileWork] = {}
        for placement in self._placements:
            roi = rois[placement.index]
            if roi.get_name() in self._work:
                raise NgioValueError(
                    f"Two tiles share the name {roi.get_name()!r}; the stitch "
                    "plan is keyed by name, so duplicates would silently "
                    "shadow each other. Give the ROIs unique names."
                )
            grown = read_roi(roi).to_pixel(pixel_size=ref_image.pixel_size)
            grown_origin = {}
            for axis_name in self._axes:
                roi_slice = grown.get(axis_name)
                assert roi_slice is not None and roi_slice.start is not None
                grown_origin[axis_name] = int(roi_slice.start)
            self._work[roi.get_name()] = _TileWork(
                placement=placement,
                offset=(placement.index + 1) * config.block_size,
                grown_origin=grown_origin,
            )

        for axis, axis_name in enumerate(self._axes):
            for index, band in forward_bands(
                self._placements, axis, halo[axis_name], self._axes
            ).items():
                self._work[rois[index].get_name()].bands[axis_name] = band

        self._scratch = _open_scratch(output, self._axes, config.scratch_store)

    @property
    def axes(self) -> tuple[str, ...]:
        """The haloed axes, in the order the placements use."""
        return self._axes

    def work_for(self, roi: Roi) -> _TileWork:
        """The tile record for `roi`."""
        work = self._work.get(roi.get_name())
        if work is None:
            raise NgioValueError(
                f"Tile {roi.get_name()!r} is not part of the stitch plan; the "
                "ROI list changed after the plan was built."
            )
        return work

    def scratch_for(self, axis_name: str) -> zarr.Array:
        """The scratch array holding forward bands along `axis_name`."""
        return self._scratch[axis_name]

    def resolve(self) -> None:
        """Scan the seams, union the agreeing ids, and relabel in place."""
        union = UnionFind[int]()
        label_array = self._output.zarr_array

        for axis, axis_name in enumerate(self._axes):
            scratch = self._scratch[axis_name]
            for lower, upper in neighbours_along(self._placements, axis):
                band = shared_band(
                    lower, upper, axis, self._halo[axis_name], self._axes
                )
                if band is None:
                    continue
                selection = _band_selection(band, self._array_axes)
                predicted = np.asarray(scratch[selection])
                written = np.asarray(label_array[selection])
                for (mine, theirs), score in overlap_iou(predicted, written).items():
                    if score >= self._config.min_iou:
                        union.union(mine, theirs)

        mapping = union.resolve()
        if self._config.compact:
            # One pass: canonicalize and renumber together. Collecting the ids
            # first would mean reading the whole label twice.
            relabel_sequential(label_array, mapping)
        else:
            _apply_relabel(label_array, mapping)
        self.cleanup()

    def cleanup(self) -> None:
        """Delete the scratch arrays ngio created."""
        _delete_scratch(self._output, self._config.scratch_store)


def _band_selection(
    band: dict[str, tuple[int, int]], array_axes: Sequence[str]
) -> tuple[slice, ...]:
    """Index tuple for `band`, positioned by axis name over the array's axes."""
    return tuple(
        slice(*band[axis_name]) if axis_name in band else slice(None)
        for axis_name in array_axes
    )


def _apply_relabel(label_array: zarr.Array, mapping: dict[int, int]) -> None:
    """Rewrite ids in place, one chunk at a time."""
    if not mapping:
        return
    for selection in chunk_selections(label_array):
        block = np.asarray(label_array[selection])
        remapped = relabel(block, mapping)
        if not np.array_equal(remapped, block):
            label_array[selection] = remapped


def _open_scratch(
    output: AbstractImage,
    axes: Sequence[str],
    scratch_store: StoreOrGroup | None,
) -> dict[str, zarr.Array]:
    """Create one scratch array per haloed axis, shaped like the output.

    With no `scratch_store` the arrays live in a transient group inside the
    output label — same store, so it works under every mapper, at the cost of a
    non-standard group sitting beside the resolution levels until the stitch
    resolves.
    """
    if scratch_store is None:
        group = output._group_handler.create_group(_SCRATCH_GROUP, overwrite=True)
    else:
        group = open_group_wrapper(scratch_store, mode="a")
    array = output.zarr_array
    return {
        axis_name: group.create_array(
            name=axis_name,
            shape=array.shape,
            dtype=array.dtype,
            chunks=array.chunks,
            overwrite=True,
        )
        for axis_name in axes
    }


def _delete_scratch(
    output: AbstractImage, scratch_store: StoreOrGroup | None
) -> None:
    """Remove the scratch arrays, if they are still there.

    Only the group ngio created inside the label is deleted. A store the caller
    supplied is theirs, so its arrays are emptied of meaning but the store
    itself is left for them to dispose of.
    """
    if scratch_store is not None:
        return
    try:
        output._group_handler.delete_group(_SCRATCH_GROUP)
    except (KeyError, FileNotFoundError):
        pass


class StitchingSetter:
    """Offsets a tile's ids, banks its halo bands, then writes as usual.

    Sits *outside* the halo crop so it sees the grown patch — the halo band is
    the whole point, and the crop is what throws it away. The id offset is
    applied here rather than by a transform for the same reason: a transform on
    the setter's chain would only ever see the cropped core, and then the band
    and the core would disagree about what an object is called.
    """

    def __init__(
        self,
        setter: DataSetterProtocol[np.ndarray],
        plan: StitchPlan,
        roi: Roi,
    ) -> None:
        """Wrap `setter` for the tile covering `roi`."""
        self._setter = setter
        self._plan = plan
        self._work = plan.work_for(roi)

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
    def transforms(self):  # noqa: ANN201
        """The wrapped setter's transform chain."""
        return self._setter.transforms

    @property
    def roi(self) -> Roi:
        """The core ROI being written."""
        return self._setter.roi

    def __getstate__(self) -> dict:
        """Refuse to cross a process boundary with an in-memory scratch.

        `ProcessMapper` pickles the units, and a `MemoryStore` pickles by
        *value*: each worker would bank its bands into a private copy and the
        parent would resolve against nothing. The mappers' own store check only
        inspects the setter's target array, so it cannot see this — but pickling
        is exactly the moment it would go wrong, so the refusal belongs here.
        """
        for axis_name, array in self._plan._scratch.items():
            if NgioStore.ensure(array.store).store_type == "memory":
                raise NgioValueError(
                    "A stitch scratch on an in-memory store cannot be used with "
                    f"ProcessMapper: the '{axis_name}' bands would be banked into "
                    "each worker's private copy and lost. Pass a persistent "
                    "`scratch_store` to StitchConfig (or leave it unset to use "
                    "the output's own store), or map with ThreadedMapper."
                )
        return self.__dict__

    def __call__(self, patch: np.ndarray) -> None:
        return self.set(patch)

    def set(self, patch: np.ndarray) -> None:
        """Offset the grown patch, bank its bands, then hand it on."""
        check_offset_fits(patch, self._work.offset, self._plan._config.block_size)
        offset = offset_labels(patch, self._work.offset)
        self._bank_bands(offset)
        self._setter.set(offset)

    def _bank_bands(self, patch: np.ndarray) -> None:
        """Copy this tile's forward bands into the scratch arrays.

        Safe to do concurrently with every other tile: a tile's forward band
        lies inside the *next* tile's core and is clipped to it, so no two tiles
        ever write the same scratch pixel. The mappers' write-conflict check
        only inspects the setter's own array, so this disjointness is an
        argument rather than something they verify.
        """
        output_axes = self._setter.axes_ops.output_axes
        core_selection = list(self._setter.slicing_ops.normalized_slicing_tuple)
        on_disk_axes = self._setter.slicing_ops.on_disk_axes

        for axis_name, band in self._work.bands.items():
            patch_selection = []
            for patch_axis in output_axes:
                if patch_axis in band:
                    origin = self._work.grown_origin[patch_axis]
                    start, stop = band[patch_axis]
                    patch_selection.append(slice(start - origin, stop - origin))
                else:
                    patch_selection.append(slice(None))
            band_patch = patch[tuple(patch_selection)]

            destination = list(core_selection)
            for band_axis, (start, stop) in band.items():
                destination[on_disk_axes.index(band_axis)] = slice(start, stop)

            # Indexed through `Any`: the selection mixes slices with the ints
            # and index lists the core slicing may carry, which zarr's stubs
            # only describe one shape at a time.
            self._plan.scratch_for(axis_name)[cast("Any", tuple(destination))] = (
                set_as_numpy_axes_ops(array=band_patch, axes_ops=self._setter.axes_ops)
            )
