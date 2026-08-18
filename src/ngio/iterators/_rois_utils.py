from collections.abc import Mapping

from ngio.common import Roi
from ngio.images._abstract_image import AbstractImage
from ngio.utils import NgioValueError

# Per-axis pixels added on each side of a ROI when reading, keyed by axis name.
HaloMargins = dict[str, tuple[int, int]]


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
        start = int(roi_slice.start)
        end = start + int(roi_slice.length)
        dim = ref_image.dimensions.get(axis_name, default=1)
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


def grid(
    rois: list[Roi],
    ref_image: AbstractImage,
    size_x: int | None = None,
    size_y: int | None = None,
    size_z: int | None = None,
    size_t: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    stride_z: int | None = None,
    stride_t: int | None = None,
    base_name: str | None = None,
) -> list[Roi]:
    """Tile the ROIs with a regular grid of tiles.

    Sizes default to the full axis extent (no tiling along that axis) and
    strides default to the sizes (adjacent, non-overlapping tiles); a stride
    smaller than the size produces overlapping tiles. The final tile along an
    axis is clipped against the parent ROI. The grid is intersected with the
    existing ROIs, so tiling composes with `product` and ROI tables.
    """
    t_dim = ref_image.dimensions.get("t", default=1)
    z_dim = ref_image.dimensions.get("z", default=1)
    y_dim = ref_image.dimensions.get("y", default=1)
    x_dim = ref_image.dimensions.get("x", default=1)

    size_t = size_t if size_t is not None else t_dim
    size_z = size_z if size_z is not None else z_dim
    size_y = size_y if size_y is not None else y_dim
    size_x = size_x if size_x is not None else x_dim

    stride_t = stride_t if stride_t is not None else size_t
    stride_z = stride_z if stride_z is not None else size_z
    stride_y = stride_y if stride_y is not None else size_y
    stride_x = stride_x if stride_x is not None else size_x

    for axis_name, stride in (
        ("t", stride_t),
        ("z", stride_z),
        ("y", stride_y),
        ("x", stride_x),
    ):
        if stride < 1:
            raise NgioValueError(
                f"Grid stride along '{axis_name}' must be >= 1, got {stride}. "
                "This can happen when the requested overlap is equal to or "
                "larger than the tile size."
            )

    # Here we would create a grid of ROIs based on the specified parameters.
    new_rois = []
    for t in range(0, t_dim, stride_t):
        for z in range(0, z_dim, stride_z):
            for y in range(0, y_dim, stride_y):
                for x in range(0, x_dim, stride_x):
                    tile_name = f"t{t}_z{z}_y{y}_x{x}"
                    name = f"{base_name}_{tile_name}" if base_name else tile_name
                    roi = Roi.from_values(
                        name=name,
                        slices={
                            "x": (x, size_x),
                            "y": (y, size_y),
                            "z": (z, size_z),
                            "t": (t, size_t),
                        },
                        space="pixel",
                    )
                    new_rois.append(roi.to_world(pixel_size=ref_image.pixel_size))

    return rois_product(rois, new_rois)


def by_yx(rois: list[Roi], ref_image: AbstractImage) -> list[Roi]:
    """Return a new iterator that iterates over ROIs by YX coordinates."""
    return grid(
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
    return grid(
        rois=rois,
        ref_image=ref_image,
        size_t=1,
        stride_t=1,
    )


def by_chunks(
    rois: list[Roi],
    ref_image: AbstractImage,
    overlap_xy: int = 0,
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
    stride_x = size_x - overlap_xy if size_x is not None else None
    stride_y = size_y - overlap_xy if size_y is not None else None
    stride_z = size_z - overlap_z if size_z is not None else None
    stride_t = size_t - overlap_t if size_t is not None else None
    return grid(
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
    )
