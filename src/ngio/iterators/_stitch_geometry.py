"""Tile geometry for stitching: grid indices, and the bands tiles share.

The seam scan needs to know two things about a list of tile ROIs: which grid
cell each one occupies, and where two neighbouring tiles' predictions cover the
same pixels. Both are computed from the ROI list as a whole, which is what makes
them robust — a single ROI does not carry enough information to place itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ngio.common._roi import Roi
from ngio.images._abstract_image import AbstractImage
from ngio.utils import NgioValueError


@dataclass(frozen=True)
class TilePlacement:
    """Where one tile sits, in pixels and in grid cells.

    Attributes:
        index: Position in the iterator's `rois` list.
        grid: The tile's cell per axis, in the order of `axes`.
        origin: The core region's first pixel per axis.
        stop: One past the core region's last pixel per axis.
    """

    index: int
    grid: tuple[int, ...]
    origin: tuple[int, ...]
    stop: tuple[int, ...]


def tile_placements(
    rois: Sequence[Roi], ref_image: AbstractImage, axes: Sequence[str]
) -> list[TilePlacement]:
    """Place each ROI on the tile grid by ranking its origin.

    The grid cell comes from *ranking* the distinct origins along each axis, not
    from dividing an origin by a tile size. Division is wrong: `grid()` clips the
    final tile against the parent ROI, so a 100-pixel axis tiled at 32 ends with
    `start=96, length=4`, and `96 // 4` is 24 where the answer is 3. Ranking
    needs the whole list, which is exactly what this function is given.

    Args:
        rois: The tiles, in iterator order.
        ref_image: The image the ROIs are expressed against; supplies pixel size.
        axes: The axes to place along, e.g. `("y", "x")`.

    Returns:
        One placement per ROI, in the same order.

    Raises:
        NgioValueError: If a ROI does not pin one of `axes`.
    """
    pixel_size = ref_image.pixel_size
    bounds: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for roi in rois:
        roi_px = roi.to_pixel(pixel_size=pixel_size)
        origin, stop = [], []
        for axis_name in axes:
            roi_slice = roi_px.get(axis_name)
            if roi_slice is None or roi_slice.start is None or roi_slice.length is None:
                raise NgioValueError(
                    f"Tile {roi.get_name()!r} does not pin axis '{axis_name}', so "
                    "it cannot be placed on the tile grid."
                )
            start = int(roi_slice.start)
            origin.append(start)
            stop.append(start + int(roi_slice.length))
        bounds.append((tuple(origin), tuple(stop)))

    ranks = [
        {start: rank for rank, start in enumerate(sorted({b[0][ax] for b in bounds}))}
        for ax in range(len(axes))
    ]
    return [
        TilePlacement(
            index=index,
            grid=tuple(ranks[ax][origin[ax]] for ax in range(len(axes))),
            origin=origin,
            stop=stop,
        )
        for index, (origin, stop) in enumerate(bounds)
    ]


def neighbours_along(
    placements: Sequence[TilePlacement], axis: int
) -> list[tuple[TilePlacement, TilePlacement]]:
    """Pairs of tiles that are adjacent along `axis`, lower cell first.

    Adjacent means one grid step apart on `axis` and identical on every other,
    so each pair shares a face rather than merely an edge or a corner.
    """
    by_cell = {placement.grid: placement for placement in placements}
    pairs = []
    for placement in placements:
        neighbour_cell = list(placement.grid)
        neighbour_cell[axis] += 1
        neighbour = by_cell.get(tuple(neighbour_cell))
        if neighbour is not None:
            pairs.append((placement, neighbour))
    return pairs


def shared_band(
    lower: TilePlacement,
    upper: TilePlacement,
    axis: int,
    halo: int,
    axes: Sequence[str],
) -> dict[str, tuple[int, int]] | None:
    """The pixels where `lower`'s halo reaches into `upper`'s core.

    That band is where the two tiles both have an opinion: `lower` predicted it
    from its grown read, and `upper` owns it and wrote it. Comparing the two is
    the whole basis of the stitch.

    Returns:
        Per-axis `(start, stop)` in pixels, or `None` when the halo does not
        reach — a zero halo, or tiles that do not actually touch.
    """
    if halo <= 0:
        return None
    reach = min(lower.stop[axis] + halo, upper.stop[axis])
    if reach <= upper.origin[axis]:
        return None

    band: dict[str, tuple[int, int]] = {}
    for ax, axis_name in enumerate(axes):
        if ax == axis:
            band[axis_name] = (upper.origin[axis], reach)
        else:
            start = max(lower.origin[ax], upper.origin[ax])
            stop = min(lower.stop[ax], upper.stop[ax])
            if stop <= start:
                return None
            band[axis_name] = (start, stop)
    return band


def forward_bands(
    placements: Sequence[TilePlacement], axis: int, halo: int, axes: Sequence[str]
) -> dict[int, dict[str, tuple[int, int]]]:
    """The band each tile contributes on `axis`, keyed by tile index.

    Only the *forward* band is kept — the one reaching into the next tile along
    the axis. That is enough to give every adjacent pair one comparison, and it
    makes the scratch layer for this axis collision-free by construction: tile
    `k` writes inside tile `k+1`'s core and nowhere else, and `shared_band`
    clips to that core, so no two tiles ever touch the same pixel. Tiles at the
    far edge contribute nothing.
    """
    bands: dict[int, dict[str, tuple[int, int]]] = {}
    for lower, upper in neighbours_along(placements, axis):
        band = shared_band(lower, upper, axis, halo, axes)
        if band is not None:
            bands[lower.index] = band
    return bands
