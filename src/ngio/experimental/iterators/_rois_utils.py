from ngio import Roi, RoiPixels
from ngio.images._abstract_image import AbstractImage


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
    base_name: str = "",
) -> list[Roi]:
    """This method is a placeholder for creating a regular grid of ROIs."""
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

    # Here we would create a grid of ROIs based on the specified parameters.
    new_rois = []
    for t in range(0, t_dim, stride_t):
        for z in range(0, z_dim, stride_z):
            for y in range(0, y_dim, stride_y):
                for x in range(0, x_dim, stride_x):
                    roi = RoiPixels(
                        name=f"{base_name}({t}, {z}, {y}, {x})",
                        x=x,
                        y=y,
                        z=z,
                        t=t,
                        x_length=size_x,
                        y_length=size_y,
                        z_length=size_z,
                        t_length=size_t,
                    )
                    new_rois.append(roi.to_roi(pixel_size=ref_image.pixel_size))

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
        raise ValueError(
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
) -> list[Roi]:
    """This method is a placeholder for chunked processing."""
    chunk_size = ref_image.chunks
    t_axis = ref_image.axes_mapper.get_index("t")
    z_axis = ref_image.axes_mapper.get_index("z")
    y_axis = ref_image.axes_mapper.get_index("y")
    x_axis = ref_image.axes_mapper.get_index("x")

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
