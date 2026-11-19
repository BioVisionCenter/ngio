import itertools
import math
from collections.abc import Iterator, Mapping
from typing import Literal

from ngio.common import Roi
from ngio.common._roi import _join_roi_names
from ngio.images._abstract_image import AbstractImage
from ngio.utils import NgioValueError

HaloMargins = dict[str, tuple[int, int]]
"""Per-axis pixels added on each side of a ROI when reading, keyed by axis name."""


def halo_roi(
    roi: Roi, ref_image: AbstractImage, halo: Mapping[str, int]
) -> tuple[Roi, HaloMargins]:
    """Grow a ROI by a per-axis pixel margin, clipped to the image.

    The margin actually applied is returned alongside, per axis and per side,
    because clipping at an image border makes it asymmetric — a tile at `y=0`
    grows downwards only, and whatever crops the result back has to know that
    rather than assume the requested margin.

    Args:
        roi: The core ROI: the region that will be written.
        ref_image: The image the ROI is read from; supplies the pixel size and
            the bounds to clip against.
        halo: Pixels to add on each side, per axis name. Axes absent from the
            ROI, or absent here, are left alone.

    Returns:
        The grown ROI, and the applied `(before, after)` margin per axis. Axes
        that did not grow are omitted from the margins.
    """
    if not halo:
        return roi, {}

    pixel_size = ref_image.pixel_size
    roi_px = roi.to_pixel(pixel_size=pixel_size)
    margins: HaloMargins = {}
    for axis_name, margin in halo.items():
        if margin == 0:
            continue
        if margin < 0:
            raise NgioValueError(
                f"Halo along '{axis_name}' must be >= 0, got {margin}."
            )
        roi_slice = roi_px.get(axis_name)
        if roi_slice is None or roi_slice.start is None or roi_slice.length is None:
            # An axis the ROI does not pin has nothing to grow around.
            continue
        dim = ref_image.dimensions.get(axis_name, default=1)
        # floor/ceil + clamp, exactly as the slicing normalization rounds a
        # non-integer pixel ROI (`_slicing_tuple_boundary_check`): the margins
        # must measure the distance between the *written* core and the *read*
        # grown region, both of which are cut by that normalization. The halo
        # crop's shape check is the loud proof the two stay in agreement.
        start = max(0, min(math.floor(roi_slice.start), dim))
        end = max(0, min(math.ceil(roi_slice.start + roi_slice.length), dim))
        new_start = max(0, start - margin)
        new_end = min(dim, end + margin)
        before, after = start - new_start, new_end - end
        if before == 0 and after == 0:
            continue
        margins[axis_name] = (before, after)
        roi_px = roi_px.update_slice(axis_name, (new_start, new_end - new_start))

    if not margins:
        return roi, {}
    return roi_px.to_world(pixel_size=pixel_size), margins


def rois_product(rois_a: list[Roi], rois_b: list[Roi]) -> list[Roi]:
    """Compute the product of two sets of ROIs."""
    rois_product = []
    for roi_a in rois_a:
        for roi_b in rois_b:
            intersection = roi_a.intersection(roi_b)
            if intersection:
                rois_product.append(intersection)
    return rois_product


TailPolicy = Literal["clip", "balance", "shift", "drop"]
"""How `by_grid` handles the non-whole tile at the end of an axis."""

# The tiling axes, in the order the tile names spell them.
_GRID_AXES = ("t", "z", "y", "x")


def _axis_intervals(
    axis_name: str, dim: int, size: int, stride: int, tail: TailPolicy
) -> list[tuple[int, int]]:
    """The `(start, length)` tiles along one axis, under a tail policy.

    - `"clip"`: the tail tile keeps its start and shrinks to the border.
    - `"balance"`: the last full tile and the tail are re-split into two
      near-equal tiles, so a 1-pixel overhang never yields a 1-pixel tile.
      Only defined for adjacent tiling (`stride == size`).
    - `"shift"`: the tail tile moves back to stay full-size — it overlaps its
      neighbour, which is the caller's problem to reconcile.
    - `"drop"`: the tail is discarded; coverage becomes incomplete.
    """
    if stride < 1:
        raise NgioValueError(
            f"Grid stride along '{axis_name}' must be >= 1, got {stride}. "
            "This can happen when the requested overlap is equal to or "
            "larger than the tile size."
        )
    if tail == "balance" and stride != size:
        raise NgioValueError(
            f"tail='balance' along '{axis_name}' needs adjacent tiles "
            f"(stride == size), got stride={stride} and size={size}: with "
            "overlapping tiles there is no well-defined 'last two' to "
            "re-balance."
        )

    if dim < 1:
        # `range(0, 0, stride)` is empty under every tail policy; without
        # this the error below would blame `tail='drop'` for a 0-sized axis.
        raise NgioValueError(
            f"Cannot tile along '{axis_name}': the axis has size {dim}."
        )

    intervals: list[tuple[int, int]] = []
    for start in range(0, dim, stride):
        if start + size <= dim:
            intervals.append((start, size))
            continue
        # The non-whole tail.
        if tail == "drop":
            continue
        if tail == "clip":
            intervals.append((start, dim - start))
        elif tail == "shift":
            intervals.append((max(0, dim - size), min(size, dim)))
        elif tail == "balance":
            remainder = dim - start
            if not intervals:
                # A tile larger than the axis: nothing to balance with.
                intervals.append((start, remainder))
                continue
            last_start, last_length = intervals.pop()
            total = last_length + remainder
            first = (total + 1) // 2
            intervals.append((last_start, first))
            intervals.append((last_start + first, total - first))

    if not intervals:
        # Only `drop` can get here: an axis shorter than the tile size has
        # nothing but a tail. Silently yielding zero tiles would turn the
        # whole `map`/`reduce` into a no-op.
        raise NgioValueError(
            f"tail='drop' along '{axis_name}' drops every tile: the axis "
            f"(size {dim}) is shorter than the tile (size {size}). Shrink "
            "the tile or use tail='clip'."
        )

    # `shift` can collapse the tail onto the previous tile; keep one of each.
    seen: set[tuple[int, int]] = set()
    unique = []
    for interval in intervals:
        if interval not in seen:
            seen.add(interval)
            unique.append(interval)
    return unique


def _axis_partition(axis_name: str, dim: int, num: int) -> list[tuple[int, int]]:
    """Split one axis into `num` near-equal `(start, length)` blocks."""
    if num < 1:
        raise NgioValueError(
            f"Number of blocks along '{axis_name}' must be >= 1, got {num}."
        )
    if num > dim:
        raise NgioValueError(
            f"Cannot split '{axis_name}' (size {dim}) into {num} blocks: "
            "some blocks would be empty."
        )
    base, remainder = divmod(dim, num)
    lengths = [base + 1] * remainder + [base] * (num - remainder)
    starts = itertools.accumulate([0, *lengths[:-1]])
    return list(zip(starts, lengths, strict=True))


def _grid_tiles(
    ref_image: AbstractImage,
    intervals: dict[str, list[tuple[int, int]]],
    base_name: str,
) -> Iterator[Roi]:
    """Cross the per-axis pixel intervals into named world-space tiles."""
    for combo in itertools.product(*(intervals[axis] for axis in _GRID_AXES)):
        slices = dict(zip(_GRID_AXES, combo, strict=True))
        tile_name = "_".join(f"{axis}{start}" for axis, (start, _) in slices.items())
        name = f"{base_name}_{tile_name}" if base_name else tile_name
        yield Roi.from_values(name=name, slices=slices, space="pixel").to_world(
            pixel_size=ref_image.pixel_size
        )


def _grid_intervals(
    ref_image: AbstractImage,
    sizes: Mapping[str, int | None],
    strides: Mapping[str, int | None],
    tail: TailPolicy,
    roi_px: Roi | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Tile each axis within the image or the ROI's readable pixel bounds."""
    intervals = {}
    for axis in _GRID_AXES:
        dim = ref_image.dimensions.get(axis, default=1)
        start, stop = 0, dim
        if roi_px is not None and (bounds := roi_px.get(axis)) is not None:
            # Use the same floor/ceil and clipping as array slicing.
            if bounds.start is not None:
                start = max(0, min(math.floor(bounds.start), dim))
            if bounds.end is not None:
                stop = max(0, min(math.ceil(bounds.end), dim))

        extent = stop - start
        size = sizes[axis]
        if size is None:
            size = extent
        stride = strides[axis]
        if stride is None:
            stride = size
        axis_tiles = _axis_intervals(axis, extent, size, stride, tail)
        intervals[axis] = [(start + offset, length) for offset, length in axis_tiles]
    return intervals


def by_grid(
    rois: list[Roi],
    ref_image: AbstractImage,
    *,
    size_x: int | None = None,
    size_y: int | None = None,
    size_z: int | None = None,
    size_t: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    stride_z: int | None = None,
    stride_t: int | None = None,
    tail: TailPolicy = "clip",
    anchor: Literal["image", "roi"] = "image",
    base_name: str = "",
) -> list[Roi]:
    """Tile the ROIs with a regular grid of tiles.

    Sizes default to the full axis extent (no tiling along that axis) and
    strides default to the sizes (adjacent, non-overlapping tiles); a stride
    smaller than the size produces overlapping tiles. `tail` decides what
    happens where the grid does not divide the axis — see `_axis_intervals`.
    Anchor the grid to the image and intersect it with the existing ROIs, or
    anchor to each ROI and apply the tail policy at its own bounds.
    """
    sizes = {"t": size_t, "z": size_z, "y": size_y, "x": size_x}
    strides = {"t": stride_t, "z": stride_z, "y": stride_y, "x": stride_x}

    if anchor == "image":
        intervals = _grid_intervals(ref_image, sizes, strides, tail)
        return rois_product(rois, list(_grid_tiles(ref_image, intervals, base_name)))
    if anchor != "roi":
        raise NgioValueError(f"Grid anchor must be 'image' or 'roi', got {anchor!r}.")

    names: set[str | None] = set()
    tiles = []
    for roi_index, roi in enumerate(rois, start=1):
        roi_px = roi.to_pixel(pixel_size=ref_image.pixel_size)
        intervals = _grid_intervals(ref_image, sizes, strides, tail, roi_px)
        parent_slices = {s.axis_name: s for s in roi.slices}
        for tile in _grid_tiles(ref_image, intervals, base_name):
            name = _join_roi_names(roi.name, tile.name)
            while name in names:
                # Unnamed or identically named overlapping ROIs can share a tile origin.
                name = f"{name}:roi{roi_index}"
            names.add(name)
            slices = parent_slices | {s.axis_name: s for s in tile.slices}
            tiles.append(roi._updated(name=name, slices=tuple(slices.values())))
    return tiles


def by_blocks(
    rois: list[Roi],
    ref_image: AbstractImage,
    *,
    num_x: int = 1,
    num_y: int = 1,
    num_z: int = 1,
    num_t: int = 1,
    base_name: str = "",
) -> list[Roi]:
    """Divide each axis into a fixed number of near-equal blocks.

    The complement of `by_grid`: you say how many tiles, not how big. Block
    lengths along an axis differ by at most one pixel, so there is no tail to
    police — the partition is balanced by construction.
    """
    nums = {"t": num_t, "z": num_z, "y": num_y, "x": num_x}
    intervals = {
        axis: _axis_partition(
            axis, ref_image.dimensions.get(axis, default=1), nums[axis]
        )
        for axis in _GRID_AXES
    }
    return rois_product(rois, list(_grid_tiles(ref_image, intervals, base_name)))


def by_yx(rois: list[Roi], ref_image: AbstractImage) -> list[Roi]:
    """Return a new iterator that iterates over ROIs by YX coordinates."""
    return by_grid(
        rois=rois,
        ref_image=ref_image,
        size_z=1,
        stride_z=1,
        size_t=1,
        stride_t=1,
    )


def by_zyx(rois: list[Roi], ref_image: AbstractImage, strict: bool = True) -> list[Roi]:
    """Return a new iterator that iterates over ROIs by ZYX coordinates."""
    if strict and not ref_image.is_3d:
        raise NgioValueError(
            "Reference Input image must be 3D to iterate by ZXY coordinates. "
            f"Current dimensions: {ref_image.dimensions}"
        )
    return by_grid(
        rois=rois,
        ref_image=ref_image,
        size_t=1,
        stride_t=1,
    )


def by_storage_units(
    rois: list[Roi],
    ref_image: AbstractImage,
    overlap_x: int = 0,
    overlap_y: int = 0,
    overlap_z: int = 0,
    overlap_t: int = 0,
    grid_image: AbstractImage | None = None,
) -> list[Roi]:
    """Tile the ROIs on a storage grid.

    By default the tiles are sized by `ref_image`'s chunk grid. When
    `grid_image` is given, its write granularity (shard shape when sharded,
    chunk shape otherwise) sizes the tiles instead; the ROIs themselves stay
    in `ref_image`'s space. An axis present on `ref_image` but absent on
    `grid_image` is left un-tiled (one tile spans it) — coarser, never unsafe.

    The tail policy and anchor are pinned to `"clip"` and `"image"` to
    keep the tiles aligned to the storage grid.
    """
    if grid_image is None:
        chunk_size = ref_image.chunks
        axes_handler = ref_image.axes_handler
    else:
        chunk_size = grid_image.write_granularity
        axes_handler = grid_image.axes_handler
    t_axis = axes_handler.get_index("t")
    z_axis = axes_handler.get_index("z")
    y_axis = axes_handler.get_index("y")
    x_axis = axes_handler.get_index("x")

    size_x = chunk_size[x_axis] if x_axis is not None else None
    size_y = chunk_size[y_axis] if y_axis is not None else None
    size_z = chunk_size[z_axis] if z_axis is not None else None
    size_t = chunk_size[t_axis] if t_axis is not None else None
    stride_x = size_x - overlap_x if size_x is not None else None
    stride_y = size_y - overlap_y if size_y is not None else None
    stride_z = size_z - overlap_z if size_z is not None else None
    stride_t = size_t - overlap_t if size_t is not None else None
    return by_grid(
        rois=rois,
        ref_image=ref_image,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_t=size_t,
        stride_x=stride_x,
        stride_y=stride_y,
        stride_z=stride_z,
        stride_t=stride_t,
        tail="clip",
        anchor="image",
    )
