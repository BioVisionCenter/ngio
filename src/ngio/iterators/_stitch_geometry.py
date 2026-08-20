"""Tile geometry for stitching: pixel-space extents and the pairs sharing pixels.

The stitch needs to know, for every tile, which pixels of the label it wrote
(its *core*) and which pixels it had an opinion about (its *grown* read, core
plus halo), and which pairs of tiles overlap anywhere in that opinion. All of
it is computed in the label's own pixel space over the label's own axes — no
grid is assumed, so any ROI list works: regular grids, overlapping FOV
layouts, ragged tables.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ngio.common._roi import Roi
from ngio.images._abstract_image import AbstractImage
from ngio.utils import NgioValueError

#: Per-axis half-open pixel interval `[start, stop)`, one entry per label axis.
SpanBox = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TileExtent:
    """One tile's pixel-space extents over the label's axes.

    Attributes:
        index: Position in the iterator's `rois` list.
        core: The written region, per label axis.
        grown: The read region (core plus halo), per label axis.
        label: The ROI's masking label, when it carries one — sub-tiles of a
            masked object inherit the object's label.
    """

    index: int
    core: SpanBox
    grown: SpanBox
    label: int | None = None


def _roi_spans(
    roi: Roi, label_image: AbstractImage, axes: Sequence[str], what: str
) -> SpanBox:
    """The ROI's `[start, stop)` per label axis; unpinned axes span fully."""
    roi_px = roi.to_pixel(pixel_size=label_image.pixel_size)
    spans = []
    for axis_name in axes:
        dim = label_image.dimensions.get(axis_name, default=1)
        roi_slice = roi_px.get(axis_name)
        if roi_slice is None or roi_slice.start is None or roi_slice.length is None:
            spans.append((0, int(dim)))
            continue
        start = int(roi_slice.start)
        stop = start + int(roi_slice.length)
        if stop <= start:
            raise NgioValueError(
                f"Tile {roi.get_name()!r}: the {what} region is empty along "
                f"'{axis_name}' ({start}..{stop})."
            )
        spans.append((start, stop))
    return tuple(spans)


def tile_extents(
    rois: Sequence[Roi],
    label_image: AbstractImage,
    read_roi: Callable[[Roi], Roi],
    axes: Sequence[str],
) -> list[TileExtent]:
    """Every tile's core and grown extents, in label pixel space.

    Args:
        rois: The tiles, in iterator order.
        label_image: The output label; supplies pixel size and axis extents.
        read_roi: Grows a core ROI by the halo (the iterator's `_read_roi`).
        axes: The label's on-disk axes, in order.
    """
    return [
        TileExtent(
            index=index,
            core=_roi_spans(roi, label_image, axes, "written"),
            grown=_roi_spans(read_roi(roi), label_image, axes, "read"),
            label=roi.label,
        )
        for index, roi in enumerate(rois)
    ]


def intersection_box(a: SpanBox, b: SpanBox) -> SpanBox | None:
    """The per-axis intersection of two boxes, or `None` when they are disjoint."""
    box = []
    for (a_start, a_stop), (b_start, b_stop) in zip(a, b, strict=True):
        start, stop = max(a_start, b_start), min(a_stop, b_stop)
        if stop <= start:
            return None
        box.append((start, stop))
    return tuple(box)


def sweep_pairs(spans: Sequence[SpanBox]) -> list[tuple[int, int]]:
    """All unordered index pairs whose boxes intersect on every axis.

    Sort-and-sweep on the axis with the most distinct starts, comparing each
    box only against those still open there — near-linear on tilings, worst
    case O(n²) when everything overlaps everything (in which case the output
    is that big anyway). Deterministic: pairs come back sorted.
    """
    if len(spans) < 2:
        return []
    n_axes = len(spans[0])
    sweep_axis = max(range(n_axes), key=lambda ax: len({box[ax][0] for box in spans}))
    order = sorted(range(len(spans)), key=lambda i: spans[i][sweep_axis][0])

    pairs: list[tuple[int, int]] = []
    active: list[int] = []
    for i in order:
        start = spans[i][sweep_axis][0]
        active = [j for j in active if spans[j][sweep_axis][1] > start]
        for j in active:
            if intersection_box(spans[i], spans[j]) is not None:
                pairs.append((min(i, j), max(i, j)))
        active.append(i)
    return sorted(pairs)


def labels_compatible(a: TileExtent, b: TileExtent, same_label_only: bool) -> bool:
    """Whether a pair of tiles may be compared, under the label restriction.

    With `same_label_only` (the masked iterator), only tiles of the same
    masked object are comparable — an object cannot span two masks, so
    cross-label comparisons could never merge anything. Unlabelled tiles are
    always comparable.
    """
    if not same_label_only:
        return True
    return a.label is None or b.label is None or a.label == b.label


def touching_unstitched_axes(
    extents: Sequence[TileExtent],
    unhaloed_axes: Sequence[int],
    same_label_only: bool = False,
) -> list[int]:
    """Axes along which two cores touch face-to-face with no halo to cross.

    Such a pair produces no grown-region overlap on that axis, so the seam
    between them is silently unstitched — worth a warning, not an error:
    ragged and per-region ROI lists legitimately contain such adjacency.
    Under `same_label_only`, only same-object pairs count: two *different*
    masked objects touching is not a seam anyone wants stitched.
    """
    touching: list[int] = []
    for axis in unhaloed_axes:
        # Extending every box by one pixel on `axis` turns face-contact into
        # overlap; any pair found this way that a plain sweep does not find
        # is exactly a touching pair.
        extended = [
            tuple(
                (start, stop + 1) if ax == axis else (start, stop)
                for ax, (start, stop) in enumerate(extent.core)
            )
            for extent in extents
        ]
        plain = set(sweep_pairs([extent.core for extent in extents]))
        if any(
            pair not in plain
            and labels_compatible(extents[pair[0]], extents[pair[1]], same_label_only)
            for pair in sweep_pairs(extended)
        ):
            touching.append(axis)
    return touching
