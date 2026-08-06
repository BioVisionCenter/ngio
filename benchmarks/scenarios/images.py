"""Open, metadata and array-IO scenarios."""

from benchmarks._registry import benchmark
from ngio import open_image, open_ome_zarr_container


@benchmark(name="images/open_container", area="images", fixture="img_v05")
def open_container(ctx):
    return open_ome_zarr_container(ctx.store_for("img_v05"), cache=False, mode="r")


@benchmark(name="images/open_container_cached", area="images", fixture="img_v05")
def open_container_cached(ctx):
    # `cache=True` should cost strictly fewer store reads than `cache=False`.
    # It currently does not, because `ZarrGroupHandler.load_attrs` reopens the
    # group unconditionally; the gap between these two counts is that bug.
    return open_ome_zarr_container(ctx.store_for("img_v05"), cache=True, mode="r")


@benchmark(name="images/open_container_v04", area="images", fixture="img_v04")
def open_container_v04(ctx):
    # Paired with `images/open_container`: v0.4 attrs decode on the first
    # registry entry, v0.5 attrs only after one failed pydantic validation.
    return open_ome_zarr_container(ctx.store_for("img_v04"), cache=False, mode="r")


@benchmark(
    name="images/get_as_numpy_level2",
    area="images",
    fixture="img_v05",
    setup=lambda ctx: open_image(ctx.store_for("img_v05"), path="2", mode="r"),
)
def get_as_numpy_level2(image):
    # `AbstractImage.dimensions` is re-derived on every get, and each derivation
    # re-reads and re-parses the image metadata, so `get.meta` here is overhead
    # rather than payload.
    return image.get_as_numpy()


@benchmark(
    name="images/dimensions_x10",
    area="images",
    fixture="img_v05",
    setup=lambda ctx: open_image(ctx.store_for("img_v05"), path="2", mode="r"),
)
def dimensions_x10(image):
    # Pure metadata access, no array IO. Counts should be flat in the number of
    # accesses once the meta handler caches; today they grow linearly.
    return [image.dimensions for _ in range(10)]
