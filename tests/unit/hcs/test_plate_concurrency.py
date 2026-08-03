"""Contention tests for the plate's parallel-safe metadata updates.

`atomic_add_image` is a read-modify-write on a single `zarr.json` — the plate's
for the well list, then the well's for the image list — so every worker touching
a plate contends on the same two files. Without the file lock one update
overwrites another and the entry simply disappears, which is what these tests
assert against.

Each task opens its own plate rather than sharing one object, so it gets its own
`ZarrGroupHandler` and therefore its own `FileLock` and file descriptor. That is
what makes the lock genuinely exclusive between workers, and it is also the
realistic shape: in a distributed run each worker opens the plate itself.
"""

from pathlib import Path

import dask
import pytest

from ngio import create_empty_plate, open_ome_zarr_plate

_WELLS = [("B", "03"), ("B", "04"), ("C", "03"), ("C", "04")]
_IMAGES_PER_WELL = 6
_NUM_WORKERS = 8


def _expected_paths() -> tuple[list[str], list[str]]:
    wells = [f"{row}/{column}" for row, column in _WELLS]
    images = [f"{well}/{i}" for well in wells for i in range(_IMAGES_PER_WELL)]
    return sorted(wells), sorted(images)


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_atomic_add_image_loses_no_updates(tmp_path: Path, ngff_version):
    store = tmp_path / "plate.zarr"
    create_empty_plate(store, name="contended_plate", ngff_version=ngff_version)

    @dask.delayed  # type: ignore
    def add(row: str, column: str, image_path: str) -> str:
        plate = open_ome_zarr_plate(store, mode="r+")
        return plate.atomic_add_image(row=row, column=column, image_path=image_path)

    tasks = [
        add(row, column, str(i))
        for row, column in _WELLS
        for i in range(_IMAGES_PER_WELL)
    ]
    with dask.config.set(scheduler="threads", num_workers=_NUM_WORKERS):
        dask.compute(*tasks)  # type: ignore

    expected_wells, expected_images = _expected_paths()
    plate = open_ome_zarr_plate(store, mode="r")
    # Sorted rather than set-compared: a duplicated entry is as wrong as a
    # missing one, and only the sorted list catches both.
    assert sorted(plate.wells_paths()) == expected_wells
    assert sorted(plate.images_paths()) == expected_images
