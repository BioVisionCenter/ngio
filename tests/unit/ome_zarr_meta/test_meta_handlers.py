"""The metadata memo: it must save work without ever serving a stale answer."""

from pathlib import Path

import pytest

from ngio import create_empty_ome_zarr, open_ome_zarr_container
from ngio.ome_zarr_meta import ImageMetaHandler
from ngio.utils import ZarrGroupHandler


@pytest.fixture
def image_handler(tmp_path: Path) -> ImageMetaHandler:
    store = tmp_path / "image.zarr"
    create_empty_ome_zarr(
        store,
        shape=(2, 4, 32, 32),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        overwrite=True,
    )
    return ImageMetaHandler(ZarrGroupHandler(store, mode="r+"))


def test_get_meta_never_hands_out_the_memo_itself(image_handler: ImageMetaHandler):
    """Two reads must not alias.

    The decoded models are mutable and `image.meta` is public, so a shared
    instance would let one caller's edit surface in another's read.
    """
    first, second = image_handler.get_meta(), image_handler.get_meta()
    assert first is not second
    assert first.paths == second.paths


def test_mutating_without_writing_back_does_not_poison_the_memo(
    image_handler: ImageMetaHandler,
):
    """The usual pattern mutates the object it was handed.

    A caller that mutates and then does *not* call `update_meta` — because it
    raised, or because it changed its mind — must not leave the handler serving
    something that is not on disk.
    """
    before = image_handler.get_meta().name

    meta = image_handler.get_meta()
    meta._name = "never_written"

    assert image_handler.get_meta().name == before


def test_a_write_through_the_handler_is_visible(image_handler: ImageMetaHandler):
    meta = image_handler.get_meta()
    meta._name = "written"
    image_handler.update_meta(meta)

    assert image_handler.get_meta().name == "written"


def test_a_write_by_someone_else_is_visible(tmp_path: Path, image_handler):
    """The memo is keyed on the raw attributes, not on a dirty flag.

    `get_meta` still reads the group every call, so a change made by another
    writer invalidates the memo on its own. This is the freshness guarantee ngio
    has always had, and the memo must not weaken it. A second handler on the
    same group stands in for the other process: it has its own memo, so nothing
    but the store connects the two.
    """
    image_handler.get_meta()  # prime it

    url = image_handler._group_handler.full_url
    assert url is not None
    other_writer = ImageMetaHandler(ZarrGroupHandler(url, mode="r+"))
    meta = other_writer.get_meta()
    meta._name = "changed_elsewhere"
    other_writer.update_meta(meta)

    assert image_handler.get_meta().name == "changed_elsewhere"


def test_memo_survives_a_container_round_trip(tmp_path: Path):
    """The high-level path must agree with the handler-level one."""
    store = tmp_path / "container.zarr"
    create_empty_ome_zarr(
        store,
        shape=(2, 4, 32, 32),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        overwrite=True,
    )
    container = open_ome_zarr_container(store, mode="r+")
    image = container.get_image()

    container.set_channel_labels(["first", "second"])
    assert image.channel_labels == ["first", "second"]


def test_dimensions_is_derived_once(tmp_path: Path):
    """`dimensions` is fixed for the object's lifetime, like `zarr_array`.

    It is the hottest property in the library — every `get_*`/`set_*` reads it,
    iterators once per ROI and masked ones twice — and rebuilding it cost a full
    metadata reload. Freezing it is not a new assumption: `self._zarr_array` is
    fetched once in `__init__` and never refreshed, so shape and chunks were
    already a construction-time snapshot.
    """
    store = tmp_path / "dims.zarr"
    create_empty_ome_zarr(
        store,
        shape=(2, 4, 32, 32),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        overwrite=True,
    )
    image = open_ome_zarr_container(store, mode="r").get_image()

    first = image.dimensions
    assert image.dimensions is first


def test_a_write_through_the_image_redirives_dimensions(tmp_path: Path):
    """Writing metadata through this object must invalidate the derived value."""
    store = tmp_path / "dims_write.zarr"
    create_empty_ome_zarr(
        store,
        shape=(2, 4, 32, 32),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        space_unit="micrometer",
        overwrite=True,
    )
    container = open_ome_zarr_container(store, mode="r+")
    image = container.get_image()

    assert image.dimensions.pixel_size.space_unit == "micrometer"
    image.set_axes_units(space_unit="nanometer")

    assert image.dimensions.pixel_size.space_unit == "nanometer"


def test_refresh_drops_the_derived_dimensions(tmp_path: Path):
    """`refresh()` has to reach the derived values, not just the raw attributes.

    `clean_cache()` alone clears the group handler's caches, which is invisible
    to a `dimensions` already derived — so `refresh()` would silently not
    refresh the one thing callers reach for it about.
    """
    store = tmp_path / "dims_refresh.zarr"
    create_empty_ome_zarr(
        store,
        shape=(2, 4, 32, 32),
        axes_names=["c", "z", "y", "x"],
        levels=2,
        pixelsize=(0.5, 0.5),
        dtype="uint16",
        space_unit="micrometer",
        overwrite=True,
    )
    reader = open_ome_zarr_container(store, mode="r")
    # Held across the refresh on purpose: `get_image()` hands back a fresh
    # `Image` whose cache is empty, so re-fetching would pass whether or not
    # `refresh` reached anything.
    image = reader.get_image()
    assert image.dimensions.pixel_size.space_unit == "micrometer"

    writer = open_ome_zarr_container(store, mode="r+")
    writer.get_image().set_axes_units(space_unit="nanometer")

    reader.refresh()
    assert image.dimensions.pixel_size.space_unit == "nanometer"
