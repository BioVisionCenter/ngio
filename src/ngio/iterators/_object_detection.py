"""Object detection over tiles: bounding boxes in, one deduplicated ROI table out.

A detector (a YOLO model, a maxima finder) runs on one tile at a time and
reports its detections as `Roi` objects in the tile's own pixel coordinates.
This iterator owns everything around that: it tiles the image (with an
optional `with_halo` read margin, so objects on a tile's edge are seen whole
by at least one tile), anchors each tile's boxes into the reference image's
world coordinates (`Roi.anchor` — the tile's ROI plus a patch-local box),
resolves the duplicates that haloed reads produce by non-maximum suppression,
and returns the survivors as a single `RoiTable`, renumbered `1..N`.
Nothing is written: storing the table is the caller's `add_table` call.

The duplicates are the price of the halo, and NMS is the standard cure: two
tiles that both saw an object near their shared border both report it, the two
boxes overlap almost entirely, and the lower-scoring one is dropped. Per-tile
NMS inside the detector (YOLO does its own) composes cleanly with this
cross-tile pass — it removes intra-tile duplicates, this removes inter-tile
ones.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar, cast

import dask.array as da
import numpy as np
import pandas as pd

from ngio.common import Roi
from ngio.images import Image
from ngio.images._image import (
    ChannelSlicingInputType,
    add_channel_selection_to_slicing_dict,
)
from ngio.io_pipes import DaskGetter, DataGetter, NumpyGetter, TransformProtocol
from ngio.iterators._abstract_iterator import AbstractIteratorBuilder
from ngio.iterators._mappers import MapperProtocol
from ngio.iterators._partials import (
    INDEX_COLUMN,
    delete_partials_root,
    merge_partial_frames,
    prepare_partials,
    validate_partials_context,
    write_partial,
)
from ngio.tables import RoiTable
from ngio.utils import NgioValueError, deprecated

NumpyPipeType: TypeAlias = tuple[np.ndarray, Roi]
DaskPipeType: TypeAlias = tuple[da.Array, Roi]

T = TypeVar("T", np.ndarray, da.Array)

_BBOX_AXES = ("x", "y", "z")


@dataclass(frozen=True, kw_only=True)
class NmsConfig:
    """How duplicate detections are suppressed.

    Attributes:
        iou_threshold: Two boxes overlapping at or above this
            intersection-over-union are the same object; the lower-scoring
            one is suppressed. Values near 1 keep almost everything, values
            near 0 merge aggressively; 0 itself would merge any pair that
            merely touches and is refused.
        score_column: Extra field on the returned Rois ranking the
            detections (and the resulting table's column), `"confidence"`
            by default. When the detector provides no such field the box
            volume ranks instead — bigger wins.
        max_detections_per_tile: A tile producing more detections than this
            is a runaway detector; it fails loudly instead of flooding the
            table.
    """

    iou_threshold: float = 0.5
    score_column: str = "confidence"
    max_detections_per_tile: int = 10_000

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not 0.0 < self.iou_threshold <= 1.0:
            raise NgioValueError(
                f"iou_threshold must be in (0, 1], got {self.iou_threshold}. "
                "Zero would suppress any pair of boxes that merely touch."
            )
        if self.max_detections_per_tile <= 0:
            raise NgioValueError(
                "max_detections_per_tile must be > 0, got "
                f"{self.max_detections_per_tile}."
            )


class DetectionGetter(DataGetter[tuple[T, Roi]], Generic[T]):
    """Pairs a tile's patch with the (haloed) ROI the patch covers.

    The ROI is what ties the patch-local answer back to the image: it names
    the tile and, anchored with a local box, yields the absolute region.
    """

    def __init__(self, data_getter: DataGetter[T]) -> None:
        self._data_getter = data_getter
        super().__init__(
            zarr_array=data_getter.zarr_array,
            slicing_ops=data_getter.slicing_ops,
            axes_ops=data_getter.axes_ops,
            transforms=data_getter.transforms,
            roi=data_getter.roi,
        )

    def get(self) -> tuple[T, Roi]:
        return self._data_getter(), self.roi


class _UnpackedDetectionFunc:
    """Adapt `func(patch)` to the unit payload tuple.

    The payload carries the tile's ROI for `iter_as_numpy`; the detection
    function itself sees only the patch. A class rather than a closure so it
    pickles by reference and the wrapped `func` can cross a `ProcessMapper`
    boundary.
    """

    def __init__(self, func: Callable[[np.ndarray], list[Roi]]) -> None:
        self._func = func

    def __call__(self, data: NumpyPipeType) -> list[Roi]:
        patch, _ = data
        return self._func(patch)


class _Detection:
    """One box, carried from the tile that found it to the final table."""

    __slots__ = ("bounds", "group", "roi", "row", "score", "tile_index")

    def __init__(
        self,
        *,
        tile_index: int,
        row: int,
        score: float,
        roi: Roi,
        bounds: dict[str, tuple[float, float]],
        group: tuple,
    ) -> None:
        self.tile_index = tile_index
        self.row = row
        self.score = score
        # The absolute, world-space ROI (extras riding on it), not yet
        # renumbered.
        self.roi = roi
        # Reference-image pixel coordinates, `[min, max)` per bbox axis.
        self.bounds = bounds
        # Extents along the axes the boxes do not pin (t, or z for a 2D
        # detector on a 3D image): only detections sharing them can be
        # duplicates of each other.
        self.group = group


def _bbox_iou(
    a: dict[str, tuple[float, float]], b: dict[str, tuple[float, float]]
) -> float:
    """Intersection-over-union of two boxes over their (shared) axes."""
    intersection = 1.0
    volume_a = 1.0
    volume_b = 1.0
    for axis_name, (a_min, a_max) in a.items():
        b_min, b_max = b[axis_name]
        intersection *= max(0.0, min(a_max, b_max) - max(a_min, b_min))
        volume_a *= a_max - a_min
        volume_b *= b_max - b_min
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def _validated_tile_rois(result: Any, tile_name: str) -> tuple[str, ...] | None:
    """Validate one tile's returned Rois; their pinned axes, `None` when empty.

    Everything a single tile can get wrong is refused here, with the tile's
    name in the message: a non-list result, a non-Roi item, a world-space Roi
    (the patch-local numbers would land the box in the wrong place silently),
    a `name` or `label` the iterator itself assigns, an unbounded or
    non-box axis, and boxes of mixed dimensionality within the tile.
    """
    if not isinstance(result, list):
        raise NgioValueError(
            f"A detection function must return a list of Roi objects, got "
            f"{type(result).__name__} for tile {tile_name!r}."
        )
    axes: tuple[str, ...] | None = None
    for row, roi in enumerate(result):
        if not isinstance(roi, Roi):
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: expected a Roi, got "
                f"{type(roi).__name__}."
            )
        if roi.space != "pixel":
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: the Roi is in "
                f"{roi.space!r} space. Detection boxes are patch-local: "
                "build them with `Roi.from_values(..., space='pixel')`; the "
                "iterator anchors them into world coordinates."
            )
        if roi.name is not None or roi.label is not None:
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: the Roi carries a `name` "
                "or `label`, which the iterator itself assigns (survivors "
                "are renumbered 1..N). Move e.g. a class label to an extra "
                "field: `Roi.from_values(..., class_id=3)`."
            )
        pinned = {roi_slice.axis_name for roi_slice in roi.slices}
        beyond_box = pinned.difference(_BBOX_AXES)
        if beyond_box:
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: the Roi pins axes "
                f"{sorted(beyond_box)}. A detection box may pin only x, y "
                "and optionally z; the tile's other axes are inherited from "
                "the tile itself."
            )
        if "x" not in pinned or "y" not in pinned:
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: detection boxes must pin "
                "x and y (z is optional)."
            )
        for roi_slice in roi.slices:
            if roi_slice.start is None or roi_slice.length is None:
                raise NgioValueError(
                    f"Tile {tile_name!r}, item {row}: axis "
                    f"{roi_slice.axis_name!r} is not fully bounded; a box "
                    "needs both a start and a length on every axis it pins."
                )
        roi_axes = tuple(axis for axis in _BBOX_AXES if axis in pinned)
        if axes is None:
            axes = roi_axes
        elif roi_axes != axes:
            raise NgioValueError(
                f"Tile {tile_name!r}, item {row}: the box pins axes "
                f"{roi_axes} but earlier boxes pinned {axes}; every "
                "detection must report the same box dimensionality."
            )
    return axes


def _rois_to_frame(rois: Sequence[Roi], tile_name: str) -> pd.DataFrame:
    """One tile's (validated) boxes as flat records, for the partial table."""
    records = []
    for roi in rois:
        record: dict[str, Any] = {}
        for roi_slice in roi.slices:
            record[f"{roi_slice.axis_name}_start"] = roi_slice.start
            record[f"{roi_slice.axis_name}_length"] = roi_slice.length
        extras = roi.model_extra or {}
        collisions = sorted(
            key for key in extras if key in record or key == INDEX_COLUMN
        )
        if collisions:
            raise NgioValueError(
                f"Tile {tile_name!r}: extra field(s) {collisions} on a "
                "returned Roi collide with the partial table's own columns; "
                "rename them."
            )
        record.update(extras)
        records.append(record)
    return pd.DataFrame(records)


def _frame_to_rois(frame: pd.DataFrame) -> list[Roi]:
    """Rebuild one tile's patch-local boxes from its partial records."""
    axes = [axis for axis in _BBOX_AXES if f"{axis}_start" in frame.columns]
    rois = []
    for record in frame.to_dict("records"):
        slices = {
            axis: (
                float(record.pop(f"{axis}_start")),
                float(record.pop(f"{axis}_length")),
            )
            for axis in axes
        }
        # Concatenating tiles with heterogeneous extra fields NaN-fills the
        # union of columns; those fill values are not detector output and a
        # serial `detect` never sees them.
        extras = {key: value for key, value in record.items() if not pd.isna(value)}
        rois.append(Roi.from_values(slices=slices, name=None, space="pixel", **extras))
    return rois


class ObjectDetectionIterator(AbstractIteratorBuilder[NumpyPipeType, DaskPipeType]):
    """Tile an image, detect per tile, return one deduplicated ROI table.

    A read-only iterator: nothing is written to the image, and the halo is a
    pure read margin — call `with_halo(...)` so an object cut by a tile edge
    is seen whole by the neighbouring tile; the duplicate detections that
    produces are resolved by NMS (`NmsConfig`). The product is a
    `RoiTable` of world-anchored boxes, returned by `detect` for the
    caller to store.
    """

    # The halo is this iterator's read margin: there is no write to crop it
    # from, and NMS is what reconciles the overlapping context.
    _allow_readonly_halo = True

    def __init__(
        self,
        input_image: Image,
        *,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        nms: NmsConfig | None = None,
    ) -> None:
        """Initialize the iterator over `input_image`.

        Args:
            input_image: The image the detector runs on.
            channel_selection: Optional selection of channels to read.
            axes_order: Optional axes order for the patches handed to the
                detector.
            input_transforms: Optional transforms applied to each patch.
            nms: How duplicate detections are suppressed; `None` uses the
                `NmsConfig` defaults.
        """
        self._input = input_image
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._nms = nms if nms is not None else NmsConfig()

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "nms": self._nms,
        }

    def build_numpy_getter(self, roi: Roi) -> DetectionGetter[np.ndarray]:
        return DetectionGetter(
            NumpyGetter(
                zarr_array=self._input.zarr_array,
                dimensions=self._input.dimensions,
                roi=self._read_roi(roi),
                axes_order=self._axes_order,
                transforms=self._input_transforms,
                slicing_dict=self._input_slicing_kwargs,
            )
        )

    def build_dask_getter(self, roi: Roi) -> DetectionGetter[da.Array]:
        return DetectionGetter(
            DaskGetter(
                zarr_array=self._input.zarr_array,
                dimensions=self._input.dimensions,
                roi=self._read_roi(roi),
                axes_order=self._axes_order,
                transforms=self._input_transforms,
                slicing_dict=self._input_slicing_kwargs,
            )
        )

    def build_numpy_setter(self, roi: Roi) -> None:
        return None

    def build_dask_setter(self, roi: Roi) -> None:
        return None

    def finalize(self) -> None:
        """A no-op: nothing is written, so there is nothing to consolidate.

        Unlike the writers' `finalize`, this stays a no-op on a `for_job`
        slice too — the read-only gather verb is `merge_partials`.
        """

    def iter_as_numpy(self):  # type: ignore[override]
        """Iterate `(patch, roi)` pairs over the (haloed) tiles."""
        return self._iter(lazy=False, data_mode="numpy", iterator_mode="readonly")

    def iter_batched(self, batch_size: int = 8):  # type: ignore[override]
        """Iterate `(patch, roi)` payloads in batches of `batch_size`."""
        self._require_valid_batch_size(batch_size)
        return self._iter_batched_readonly(batch_size)

    @deprecated(
        replacement="iter_as_numpy() (or Image.get_as_dask() for a lazy array)",
        removed_in="1.2",
    )
    def iter_as_dask(self):  # type: ignore[override]
        """Iterate `(patch, roi)` pairs over the (haloed) tiles.

        Deprecated: removed in ngio=1.2.
        """
        return self._iter(lazy=False, data_mode="dask", iterator_mode="readonly")

    def detect(
        self,
        func: Callable[[np.ndarray], list[Roi]],
        *,
        mapper: MapperProtocol | None = None,
    ) -> RoiTable:
        """Run the detector over every tile and return one table of objects.

        The per-tile detection fans out exactly like `reduce` — pass a
        `mapper` to parallelize it — and the reconciliation (coordinates, NMS,
        renumbering) happens once, at the end, on the calling thread. Nothing
        is written: the returned table is yours to store, e.g.
        `container.add_table(name, table)`.

        Args:
            func: `(patch) -> list[Roi]` with the boxes the detector found,
                built in the *patch's own* pixel coordinates:
                `Roi.from_values(slices={"x": (x0, width), "y": (y0,
                height)}, name=None, space="pixel", confidence=0.9)`.
                `x` and `y` are required, `z` optional (the same choice on
                every box); the tile's other axes are inherited from the
                tile, and the iterator anchors each box into the reference
                image's world coordinates. `space="pixel"` is required — a
                world-space Roi is refused, and so are `name` and `label`,
                which the iterator itself assigns (move a class label to an
                extra field, e.g. `class_id=3`). Every extra field — class,
                confidence, whatever the detector reports — rides along into
                the table. Under a parallel mapper the function runs on
                worker threads or processes and must be safe there.
            mapper: How the per-tile work is scheduled; `None` is serial.

        Returns:
            A `RoiTable` of the surviving detections, ids `1..N` in
            (tile, row) order, in the reference image's world coordinates.
            An image with nothing to detect gives an empty table.
        """
        if self._partition is not None:
            raise NgioValueError(
                "detect on a partition slice would run NMS over only this "
                "job's tiles — and greedy NMS is not hierarchical, so the "
                "result could differ from a serial run. Use "
                "`detect_to_partial(func)` per job and `merge_partials()` "
                "once, after all jobs."
            )
        results = self.reduce(_UnpackedDetectionFunc(func), mapper=mapper)
        detections = self._collect(results)
        kept = self._suppress(detections)
        return RoiTable(rois=self._renumbered(kept))

    def _partials_handler(self):
        """Partials live beside the input image's levels."""
        return self._input._group_handler

    def _prepare_distributed(self, n_jobs: int) -> None:
        prepare_partials(self, self._partials_handler(), n_jobs)

    def _validate_job_context(self, n_jobs: int) -> None:
        validate_partials_context(self, self._partials_handler(), n_jobs)

    def detect_to_partial(
        self,
        func: Callable[[np.ndarray], list[Roi]],
        *,
        mapper: MapperProtocol | None = None,
    ) -> None:
        """Detect over this job's tiles and store the raw boxes as a partial.

        The per-job step of a distributed run (see `prepare_jobs`): the
        detector's raw, **pre-NMS** output is stored tagged with its global
        tile index, so `merge_partials` can run the ONE global NMS and
        renumbering — bit-identical to a serial `detect` (per-job NMS
        followed by a merge NMS would not be: greedy NMS is not
        hierarchical). The tiles' boxes are validated here, so a malformed
        detector fails in the job rather than at the merge. Only callable on
        a `for_job` slice; re-running a job overwrites its own partial.
        """
        if self._partition is None:
            raise NgioValueError(
                "detect_to_partial is the per-job step of a distributed run: "
                "restrict the iterator first (`for_job(**args)`), or use "
                "detect() for a local run."
            )
        _, job_index, n_jobs = self._partition
        handler = self._partials_handler()
        validate_partials_context(self, handler, n_jobs)

        indices = self.partition_indices or []
        results = self.reduce(_UnpackedDetectionFunc(func), mapper=mapper)
        # Early, loud validation with the correct global tile indices; the
        # anchored output is discarded — the merge re-derives everything from
        # the raw boxes so the whole pipeline runs once, globally.
        self._collect(results, tile_indices=indices)

        frames = []
        for index, result in zip(indices, results, strict=True):
            if not result:
                continue
            frame = _rois_to_frame(result, self.rois[index].get_name())
            frame[INDEX_COLUMN] = index
            frames.append(frame)
        payload = pd.concat(frames, ignore_index=True) if frames else None
        write_partial(
            handler,
            job_index,
            payload,
            attrs={
                "job_index": job_index,
                "n_jobs": n_jobs,
                "fingerprint": self._plan_fingerprint(n_jobs),
            },
        )

    def merge_partials(self) -> RoiTable:
        """Merge a distributed run's partials into the one final ROI table.

        The consolidate step (see `prepare_jobs`): validates that every job
        of the prepared plan produced a matching partial — a half-finished
        run errors instead of returning a silently incomplete table — then
        rebuilds every tile's raw boxes and runs the full serial pipeline
        once, globally: anchoring, the cross-tile invariant checks, NMS, and
        the dense `1..N` renumbering. Bit-identical to a serial `detect` by
        construction (one exception: an extra field a detector sets to `NaN`
        does not survive the partial round-trip).
        On success the partials group is removed. Nothing is
        registered: the returned table is yours to store with `add_table`.
        """
        if self._partition is not None:
            raise NgioValueError(
                "merge_partials belongs to the unrestricted iterator: rebuild "
                "it without `for_job` for the consolidate step."
            )
        handler = self._partials_handler()
        merged = merge_partial_frames(self, handler)

        results: list[list[Roi]] = [[] for _ in self.rois]
        if merged is not None:
            for index, group in merged.groupby(INDEX_COLUMN, sort=True):
                results[int(cast("Any", index))] = _frame_to_rois(
                    group.drop(columns=[INDEX_COLUMN])
                )
        detections = self._collect(results)
        kept = self._suppress(detections)
        table = RoiTable(rois=self._renumbered(kept))
        delete_partials_root(handler)
        return table

    def _collect(
        self,
        results: list[list[Roi]],
        tile_indices: Sequence[int] | None = None,
    ) -> list[_Detection]:
        """Validate and anchor every tile's boxes into reference coordinates.

        `tile_indices` maps each result to its global position in `rois`;
        `None` means the results cover every ROI in order (the serial case).
        """
        pixel_size = self._ref_image.pixel_size
        detections: list[_Detection] = []
        bbox_axes: tuple[str, ...] | None = None
        tiles: list[tuple[int, list[Roi]]] = []
        scored = 0
        total = 0

        if tile_indices is None:
            tile_indices = range(len(results))
        for tile_index, result in zip(tile_indices, results, strict=True):
            tile_name = self.rois[tile_index].get_name()
            axes = _validated_tile_rois(result, tile_name)
            if axes is None:
                continue
            if bbox_axes is None:
                bbox_axes = axes
            elif axes != bbox_axes:
                raise NgioValueError(
                    f"Tile {tile_name!r} pins axes {axes} but earlier tiles "
                    f"pinned {bbox_axes}; every tile must report the same "
                    "box dimensionality."
                )
            if len(result) > self._nms.max_detections_per_tile:
                raise NgioValueError(
                    f"Tile {tile_name!r} produced {len(result)} detections, "
                    "more than NmsConfig.max_detections_per_tile "
                    f"({self._nms.max_detections_per_tile}) allows. Raise the "
                    "cap if the count is genuine; a runaway detector should "
                    "fail here rather than flood the table."
                )
            scored += sum(
                1 for roi in result if self._nms.score_column in (roi.model_extra or {})
            )
            total += len(result)
            tiles.append((tile_index, result))

        if scored and scored != total:
            raise NgioValueError(
                f"Some detections carry a {self._nms.score_column!r} extra "
                "field and some do not; NMS needs one consistent ranking. "
                "Report the score on every box, or on none (box volume ranks "
                "instead)."
            )

        for tile_index, rois in tiles:
            assert bbox_axes is not None
            tile_roi = self.rois[tile_index]
            read_region = self._read_roi(tile_roi)

            for row, local in enumerate(rois):
                absolute = read_region.anchor(local, pixel_size=pixel_size)

                absolute_px = absolute.to_pixel(pixel_size=pixel_size)
                bounds: dict[str, tuple[float, float]] = {}
                for axis_name in bbox_axes:
                    box_slice = absolute_px[axis_name]
                    assert box_slice.start is not None
                    assert box_slice.length is not None
                    bounds[axis_name] = (
                        box_slice.start,
                        box_slice.start + box_slice.length,
                    )
                extras = local.model_extra or {}
                if scored:
                    score = float(extras[self._nms.score_column])
                else:
                    # No confidence to rank by: the bigger box wins.
                    score = 1.0
                    for roi_slice in local.slices:
                        assert roi_slice.length is not None
                        score *= roi_slice.length
                detections.append(
                    _Detection(
                        tile_index=tile_index,
                        row=row,
                        score=score,
                        roi=absolute,
                        bounds=bounds,
                        group=self._group_key(absolute, bbox_axes),
                    )
                )
        return detections

    def _group_key(self, absolute: Roi, bbox_axes: tuple[str, ...]) -> tuple:
        """The detection's extents along the axes its box does not pin.

        Inherited from the tile by `Roi.anchor`. Two detections can only be
        duplicates of one another where their tiles cover the same slab of
        those axes: a 2D detector run frame by frame must not have NMS merge
        boxes from different time points.
        """
        key = []
        for axis_name in ("t", "z"):
            if axis_name in bbox_axes:
                continue
            roi_slice = absolute.get(axis_name)
            if roi_slice is not None and roi_slice.start is not None:
                key.append((axis_name, roi_slice.start, roi_slice.length))
        return tuple(key)

    def _suppress(self, detections: list[_Detection]) -> list[_Detection]:
        """Greedy NMS: highest score first, drop what overlaps a kept box."""
        # Deterministic under ties and any mapper: (tile, row) breaks them.
        ranked = sorted(detections, key=lambda d: (-d.score, d.tile_index, d.row))
        kept: list[_Detection] = []
        for detection in ranked:
            duplicate = any(
                other.group == detection.group
                and _bbox_iou(other.bounds, detection.bounds) >= self._nms.iou_threshold
                for other in kept
            )
            if not duplicate:
                kept.append(detection)
        return kept

    def _renumbered(self, kept: list[_Detection]) -> list[Roi]:
        """Dense ids `1..N` in (tile, row) order, stamped on the final ROIs."""
        return [
            detection.roi.model_copy(update={"label": label, "name": str(label)})
            for label, detection in enumerate(
                sorted(kept, key=lambda d: (d.tile_index, d.row)), start=1
            )
        ]
