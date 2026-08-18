"""Object detection over tiles: bounding boxes in, one masking ROI table out.

A detector (a YOLO model, a maxima finder) runs on one tile at a time and
numbers its detections from zero, in the tile's own pixel coordinates. This
iterator owns everything around that: it tiles the image (with an optional
padded read, so objects on a tile's edge are seen whole by at least one tile),
anchors each tile's boxes into the reference image's world coordinates
(`Roi.compose` — the tile's ROI plus a patch-local box), resolves the
duplicates that padded reads produce by non-maximum suppression, and returns
the survivors as a single `MaskingRoiTable`, renumbered `1..N`. Nothing is
written: storing the table is the caller's `add_table` call.

The duplicates are the price of the padding, and NMS is the standard cure: two
tiles that both saw an object near their shared border both report it, the two
boxes overlap almost entirely, and the lower-scoring one is dropped. Per-tile
NMS inside the detector (YOLO does its own) composes cleanly with this
cross-tile pass — it removes intra-tile duplicates, this removes inter-tile
ones.
"""

from collections.abc import Callable, Sequence
from typing import Generic, TypeAlias, TypeVar

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
from ngio.iterators._rois_utils import halo_roi
from ngio.tables import MaskingRoiTable
from ngio.utils import NgioValueError, deprecated

#: What a per-tile detection function may return: a DataFrame or a dict of
#: columns. Boxes are `[min, max)` in the tile patch's own pixel coordinates:
#: `x_min`/`x_max` and `y_min`/`y_max` are required, `z_min`/`z_max` optional
#: (both or neither). Every other column rides along into the table.
DetectionFuncResult: TypeAlias = pd.DataFrame | dict[str, list]

NumpyPipeType: TypeAlias = tuple[np.ndarray, Roi]
DaskPipeType: TypeAlias = tuple[da.Array, Roi]

T = TypeVar("T", np.ndarray, da.Array)

_BBOX_AXES = ("x", "y", "z")

#: Detection columns that would collide with the `Roi` fields the iterator
#: itself assigns; a detector reporting these must rename them.
_RESERVED_COLUMNS = frozenset({"name", "label", "slices", "space"})


class DetectionGetter(DataGetter[tuple[T, Roi]], Generic[T]):
    """Pairs a tile's patch with the (padded) ROI the patch covers.

    The ROI is what ties the patch-local answer back to the image: it names
    the tile and, composed with a local box, yields the absolute region.
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
    """Adapt `func(patch, roi)` to the unit payload tuple.

    A class rather than a closure so it pickles by reference and the wrapped
    `func` can cross a `ProcessMapper` boundary.
    """

    def __init__(
        self, func: Callable[[np.ndarray, Roi], DetectionFuncResult]
    ) -> None:
        self._func = func

    def __call__(self, data: NumpyPipeType) -> DetectionFuncResult:
        patch, roi = data
        return self._func(patch, roi)


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


def _as_dataframe(result: DetectionFuncResult, tile_name: str) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        return pd.DataFrame(result)
    raise NgioValueError(
        f"A detection function must return a DataFrame or a dict of columns, "
        f"got {type(result).__name__} for tile {tile_name!r}."
    )


def _bbox_axes_of(frame: pd.DataFrame, tile_name: str) -> tuple[str, ...]:
    """The axes the frame's columns pin, validating the column contract."""
    axes = []
    for axis_name in _BBOX_AXES:
        has_min = f"{axis_name}_min" in frame.columns
        has_max = f"{axis_name}_max" in frame.columns
        if has_min != has_max:
            raise NgioValueError(
                f"Tile {tile_name!r} carries '{axis_name}_min' or "
                f"'{axis_name}_max' but not both; a box needs both bounds."
            )
        if has_min:
            axes.append(axis_name)
    for required in ("x", "y"):
        if required not in axes:
            raise NgioValueError(
                f"Tile {tile_name!r} has no '{required}_min'/'{required}_max' "
                "columns: detection boxes must pin x and y (z is optional)."
            )
    reserved = _RESERVED_COLUMNS.intersection(frame.columns)
    if reserved:
        raise NgioValueError(
            f"Tile {tile_name!r} carries column(s) {sorted(reserved)}, which "
            "collide with the ROI fields the iterator assigns (the table's "
            "own ids among them). Rename them, e.g. 'label' -> 'class_id'."
        )
    return tuple(axes)


class ObjectDetectionIterator(AbstractIteratorBuilder[NumpyPipeType, DaskPipeType]):
    """Tile an image, detect per tile, return one deduplicated ROI table."""

    def __init__(
        self,
        input_image: Image,
        channel_selection: ChannelSlicingInputType = None,
        axes_order: Sequence[str] | None = None,
        input_transforms: Sequence[TransformProtocol] | None = None,
        padding_x: int = 0,
        padding_y: int = 0,
        padding_z: int = 0,
        block_size: int = 10_000,
        iou_threshold: float = 0.5,
        score_column: str = "confidence",
    ) -> None:
        """Initialize the iterator over `input_image`.

        Args:
            input_image: The image the detector runs on.
            channel_selection: Optional selection of channels to read.
            axes_order: Optional axes order for the patches handed to the
                detector.
            input_transforms: Optional transforms applied to each patch.
            padding_x: Extra pixels read on each side of a tile along x,
                clipped at the image borders. An object cut by a tile edge is
                seen whole by the neighbouring tile's padded read; the
                duplicate detections this produces are resolved by NMS.
            padding_y: As `padding_x`, along y.
            padding_z: As `padding_x`, along z.
            block_size: Ids reserved per tile: no single tile may produce
                this many detections. The final table renumbers to a dense
                `1..N` anyway, so the blocks are bookkeeping — but a tile
                overflowing its block is a runaway detector, and it fails
                loudly instead of flooding the table.
            iou_threshold: Two boxes overlapping at or above this
                intersection-over-union are the same object; the
                lower-scoring one is suppressed. Values near 1 keep almost
                everything, values near 0 merge aggressively; 0 itself would
                merge any pair that merely touches and is refused.
            score_column: Column ranking the detections for NMS, `"confidence"`
                by default. When the detector provides no such column the box
                volume ranks instead — bigger wins.
        """
        for name, value in (
            ("padding_x", padding_x),
            ("padding_y", padding_y),
            ("padding_z", padding_z),
        ):
            if value < 0:
                raise NgioValueError(f"{name} must be >= 0, got {value}.")
        if block_size <= 0:
            raise NgioValueError(f"block_size must be > 0, got {block_size}.")
        if not 0.0 < iou_threshold <= 1.0:
            raise NgioValueError(
                f"iou_threshold must be in (0, 1], got {iou_threshold}. Zero "
                "would suppress any pair of boxes that merely touch."
            )
        self._input = input_image
        self._ref_image = input_image
        self._rois = input_image.build_image_roi_table(name=None).rois()

        self._input_slicing_kwargs = add_channel_selection_to_slicing_dict(
            image=self._input, channel_selection=channel_selection, slicing_dict={}
        )
        self._channel_selection = channel_selection
        self._axes_order = axes_order
        self._input_transforms = input_transforms
        self._padding = {
            axis_name: value
            for axis_name, value in (
                ("x", padding_x),
                ("y", padding_y),
                ("z", padding_z),
            )
            if value > 0
        }
        self._block_size = block_size
        self._iou_threshold = iou_threshold
        self._score_column = score_column

    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        return {
            "input_image": self._input,
            "channel_selection": self._channel_selection,
            "axes_order": self._axes_order,
            "input_transforms": self._input_transforms,
            "padding_x": self._padding.get("x", 0),
            "padding_y": self._padding.get("y", 0),
            "padding_z": self._padding.get("z", 0),
            "block_size": self._block_size,
            "iou_threshold": self._iou_threshold,
            "score_column": self._score_column,
        }

    @property
    def padding(self) -> dict[str, int]:
        """The per-axis read padding, empty when there is none."""
        return dict(self._padding)

    def _padded_roi(self, roi: Roi) -> Roi:
        """The region a tile reads: grown by the padding, clipped at borders."""
        if not self._padding:
            return roi
        padded, _ = halo_roi(roi, self._ref_image, self._padding)
        return padded

    def build_numpy_getter(self, roi: Roi) -> DetectionGetter[np.ndarray]:
        return DetectionGetter(
            NumpyGetter(
                zarr_array=self._input.zarr_array,
                dimensions=self._input.dimensions,
                roi=self._padded_roi(roi),
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
                roi=self._padded_roi(roi),
                axes_order=self._axes_order,
                transforms=self._input_transforms,
                slicing_dict=self._input_slicing_kwargs,
            )
        )

    def build_numpy_setter(self, roi: Roi) -> None:
        return None

    def build_dask_setter(self, roi: Roi) -> None:
        return None

    def post_consolidate(self) -> None:
        pass

    def iter_as_numpy(self):  # type: ignore[override]
        """Iterate `(patch, roi)` pairs over the (padded) tiles."""
        return self._iter(lazy=False, data_mode="numpy", iterator_mode="readonly")

    @deprecated(
        replacement="iter_as_numpy() (or Image.get_as_dask() for a lazy array)",
        removed_in="1.2",
    )
    def iter_as_dask(self):  # type: ignore[override]
        """Iterate lazy `(patch, roi)` pairs over the (padded) tiles.

        Deprecated: removed in ngio=1.2.
        """
        return self._iter(lazy=False, data_mode="dask", iterator_mode="readonly")

    def detect(
        self,
        func: Callable[[np.ndarray, Roi], DetectionFuncResult],
        *,
        mapper: MapperProtocol | None = None,
    ) -> MaskingRoiTable:
        """Run the detector over every tile and return one table of objects.

        The per-tile detection fans out exactly like `reduce` — pass a
        `mapper` to parallelize it — and the reconciliation (coordinates, NMS,
        renumbering) happens once, at the end, on the calling thread. Nothing
        is written: the returned table is yours to store, e.g.
        `container.add_table(name, table)`.

        Args:
            func: `(patch, roi) -> DataFrame | dict[str, list]` with the
                boxes the detector found. `roi` is the (padded) region the
                patch covers — the same argument every iterator func gets to
                identify its region globally; a custom flow can anchor its
                own boxes with `roi.compose(...)`. The boxes are `[min, max)`
                in the *patch's own* pixel coordinates: `x_min`/`x_max` and
                `y_min`/`y_max` required, `z_min`/`z_max` optional (both or
                neither, and the same choice on every tile). Every other
                column — class, confidence, whatever the detector reports —
                rides along into the table. Under a parallel mapper the
                function runs on worker threads or processes and must be
                safe there.
            mapper: How the per-tile work is scheduled; `None` is serial.

        Returns:
            A `MaskingRoiTable` of the surviving detections, ids `1..N` in
            (tile, row) order, in the reference image's world coordinates.
            An image with nothing to detect gives an empty table.
        """
        results = self.reduce(_UnpackedDetectionFunc(func), mapper=mapper)
        detections = self._collect(results)
        kept = self._suppress(detections)
        return MaskingRoiTable(rois=self._renumbered(kept))

    def _collect(self, results: list[DetectionFuncResult]) -> list[_Detection]:
        """Validate and anchor every tile's boxes into reference coordinates."""
        pixel_size = self._ref_image.pixel_size
        detections: list[_Detection] = []
        bbox_axes: tuple[str, ...] | None = None
        frames: list[tuple[int, pd.DataFrame]] = []
        scored = 0

        for tile_index, result in enumerate(results):
            tile_name = self.rois[tile_index].get_name()
            frame = _as_dataframe(result, tile_name)
            if not len(frame):
                continue
            axes = _bbox_axes_of(frame, tile_name)
            if bbox_axes is None:
                bbox_axes = axes
            elif axes != bbox_axes:
                raise NgioValueError(
                    f"Tile {tile_name!r} pins axes {axes} but earlier tiles "
                    f"pinned {bbox_axes}; every tile must report the same "
                    "box dimensionality."
                )
            if len(frame) >= self._block_size:
                raise NgioValueError(
                    f"Tile {tile_name!r} produced {len(frame)} detections, "
                    f"which does not fit in a block of {self._block_size} "
                    "ids. Raise block_size above the largest count any "
                    "single tile can produce."
                )
            if self._score_column in frame.columns:
                scored += 1
            frames.append((tile_index, frame))

        if scored and scored != len(frames):
            raise NgioValueError(
                f"Some tiles carry a {self._score_column!r} column and some "
                "do not; NMS needs one consistent ranking. Report the score "
                "from every tile, or from none (box volume ranks instead)."
            )

        for tile_index, frame in frames:
            assert bbox_axes is not None
            tile_roi = self.rois[tile_index]
            tile_name = tile_roi.get_name()
            padded = self._padded_roi(tile_roi)

            for row, record in enumerate(frame.to_dict("records")):
                slices: dict[str, tuple[float, float]] = {}
                for axis_name in bbox_axes:
                    low = float(record.pop(f"{axis_name}_min"))
                    high = float(record.pop(f"{axis_name}_max"))
                    if high < low:
                        raise NgioValueError(
                            f"Tile {tile_name!r}, row {row}: "
                            f"{axis_name}_max ({high}) is below "
                            f"{axis_name}_min ({low})."
                        )
                    slices[axis_name] = (low, high - low)
                local = Roi.from_values(
                    slices=slices, name=None, space="pixel", **record
                )
                absolute = padded.compose(local, pixel_size=pixel_size)

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
                if self._score_column in frame.columns:
                    score = float(record[self._score_column])
                else:
                    # No confidence to rank by: the bigger box wins.
                    score = 1.0
                    for low, length in slices.values():
                        score *= length
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

        Inherited from the tile by `Roi.compose`. Two detections can only be
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
                and _bbox_iou(other.bounds, detection.bounds) >= self._iou_threshold
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
