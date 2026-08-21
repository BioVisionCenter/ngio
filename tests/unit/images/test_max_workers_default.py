"""The notice that plate-wide fan-outs will stop being serial by default."""

from pathlib import Path

import numpy as np
import pytest

from ngio import create_empty_plate, create_ome_zarr_from_array
from ngio.ome_zarr_meta import ImageInWellPath
from ngio.utils import NgioFutureWarning


@pytest.fixture
def plate(tmp_path: Path):
    images = [
        ImageInWellPath(row=row, column=f"{col + 1:02d}", path="0")
        for row in ("A", "B")
        for col in range(2)
    ]
    return create_empty_plate(
        tmp_path / "notice.zarr", name="plate", images=images, overwrite=True
    )


def test_omitting_max_workers_announces_the_coming_default(plate):
    with pytest.warns(NgioFutureWarning, match="max_workers"):
        plate.get_wells()


@pytest.mark.parametrize("max_workers", [1, 4, "auto"])
def test_picking_a_value_silences_it(plate, max_workers, recwarn):
    """Both opting in and opting out must be quiet.

    `1` is the way to keep reading serially on purpose, which is why the notice
    points at it rather than at `None` — `None` is the value whose meaning is
    about to change.
    """
    plate.get_wells(max_workers=max_workers)
    assert [w for w in recwarn if issubclass(w.category, NgioFutureWarning)] == []


def test_nothing_is_said_when_there_is_nothing_to_parallelise(tmp_path, recwarn):
    """One item runs the same either way, so the default cannot matter."""
    single = create_empty_plate(
        tmp_path / "one.zarr",
        name="plate",
        images=[ImageInWellPath(row="A", column="01", path="0")],
        overwrite=True,
    )
    single.get_wells()
    assert [w for w in recwarn if issubclass(w.category, NgioFutureWarning)] == []


def test_the_notice_points_at_the_caller_not_at_ngio(plate):
    """A notice blaming ngio's own source is useless, and worse than useless.

    `get_wells()` reaches the fan-out through two ngio frames and
    `images_paths()` through three, so a fixed `stacklevel` misattributes one of
    them. The `warnings` module also keys its deduplication on the reported
    location, so a misattributed notice collapses every caller onto one entry
    and most of them are never told at all.
    """
    this_file = __file__

    with pytest.warns(NgioFutureWarning) as direct:
        plate.get_wells()
    with pytest.warns(NgioFutureWarning) as nested:
        plate.images_paths()

    assert direct[0].filename == this_file
    assert nested[0].filename == this_file
    # ...and they are distinct locations, so neither hides the other.
    assert direct[0].lineno != nested[0].lineno


@pytest.fixture
def populated_plate(tmp_path: Path):
    images = [
        ImageInWellPath(row="A", column="01", path="0"),
        ImageInWellPath(row="A", column="02", path="0"),
    ]
    plate = create_empty_plate(
        tmp_path / "populated.zarr", name="plate", images=images, overwrite=True
    )
    for column in ("01", "02"):
        store = plate.get_image_store(row="A", column=column, image_path="0")
        create_ome_zarr_from_array(
            array=np.zeros((8, 8), dtype="uint8"),
            store=store,
            pixelsize=1.0,
            axes_names=["y", "x"],
            levels=1,
            consolidation_mode="auto",
        )
    return plate


@pytest.mark.parametrize(
    "call",
    [
        lambda plate: plate.get_images(max_workers="auto"),
        lambda plate: plate.list_image_tables(max_workers="auto"),
    ],
)
def test_opting_in_silences_the_nested_fan_outs_too(populated_plate, call, recwarn):
    """`max_workers="auto"` must reach every fan-out a call runs internally.

    `get_images` lists paths through `images_paths` → `get_wells` before
    opening anything; dropping the forward on that hop made the notice fire
    at callers who had already opted in — transitively from every
    table-listing helper built on `get_images`.
    """
    call(populated_plate)
    assert [w for w in recwarn if issubclass(w.category, NgioFutureWarning)] == []
