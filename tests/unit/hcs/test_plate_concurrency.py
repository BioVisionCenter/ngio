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

Skipped on Windows, where the lock does not hold. `filelock` deletes the lock
file on release on every platform, but only its Unix backend checks that the
file it locked still exists — `_unix.py::_acquire` discards the descriptor when
`os.fstat(fd).st_nlink == 0`, and `_windows.py::_acquire` has no equivalent
(`grep -c st_nlink` is 1 and 0 respectively, as of `filelock` 3.29.4). So on
Windows worker A can open the lock file, worker B release and unlink it, A lock
the now orphaned handle believing it holds the lock, and worker C re-create the
file and lock that: two workers in the critical section, one update lost. It
manifests as a silently missing image rather than an error, and it is
timing-dependent — it fired in one of four Windows jobs. Concurrent writers on
Windows are not a use case ngio supports, so this is documented rather than
worked around; see the `atomic_*` docstrings in `ngio/hcs/_plate.py`.

Skipping costs no Windows coverage of the file-sharing retry in
`ngio/utils/_retry.py` — `test_multiprocessing_safety`, in
`tests/unit/utils/test_zarr_utils.py`, exercises that path and runs everywhere.
"""

import sys
from pathlib import Path

import dask
import pytest

from ngio import create_empty_plate, open_ome_zarr_plate

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "filelock's Windows backend can hand one lock to two workers, so these "
        "assertions do not hold there; see the module docstring"
    ),
)

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
