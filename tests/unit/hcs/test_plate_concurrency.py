"""Contention tests for the plate's parallel-safe metadata updates.

`atomic_add_image` is a read-modify-write on a single `zarr.json` — the plate's
for the well list, then the well's for the image list — so every worker touching
a plate contends on the same two files. Without the file lock one update
overwrites another and the entry simply disappears, which is what these tests
assert against.

Run in **separate processes**, not threads. `atomic_add_image` promises safety
"across threads and processes", and the process half is the half that is hard:
threads share one interpreter, so a lock that is merely thread-safe would pass a
threaded test while the guarantee stays broken. An earlier version of this file
used dask's threaded scheduler and could not see that. `spawn`, not `fork`: zarr
runs an IO event loop on its own thread, and forking a multi-threaded process
warns (fatal here — `filterwarnings = ["error"]`).

Each worker opens its own plate, so it gets its own `ZarrGroupHandler` and
therefore its own `FileLock` and file descriptor. That is what makes the lock
genuinely exclusive between workers, and it is also the realistic shape: in a
distributed run each worker opens the plate itself.

Skipped on Windows, where `atomic_*` raises instead of taking a lock that
`filelock` cannot make exclusive there — see `ZarrGroupHandler._create_lock`.
`test_atomic_add_image_rejects_windows` covers that refusal on every platform.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from ngio import create_empty_plate, open_ome_zarr_plate
from ngio.utils import NgioValueError
from ngio.utils import _retry as _retry_module

_WELLS = [("B", "03"), ("B", "04"), ("C", "03"), ("C", "04")]
_IMAGES_PER_WELL = 6
_NUM_WORKERS = 4


def _add_image(args: tuple[str, str, str, str]) -> tuple[str, int]:
    """Add one image to a plate, in a worker process.

    Module level and picklable by reference, which is what `spawn` needs. The
    pid comes back so the test can prove the work really left the parent.
    """
    store, row, column, image_path = args
    plate = open_ome_zarr_plate(store, mode="r+")
    path = plate.atomic_add_image(row=row, column=column, image_path=image_path)
    return path, os.getpid()


def _expected_paths() -> tuple[list[str], list[str]]:
    wells = [f"{row}/{column}" for row, column in _WELLS]
    images = [f"{well}/{i}" for well in wells for i in range(_IMAGES_PER_WELL)]
    return sorted(wells), sorted(images)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="atomic_* is unsupported on Windows; see the module docstring",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_atomic_add_image_loses_no_updates(tmp_path: Path, ngff_version):
    store = tmp_path / "plate.zarr"
    create_empty_plate(store, name="contended_plate", ngff_version=ngff_version)

    jobs = [
        (str(store), row, column, str(i))
        for row, column in _WELLS
        for i in range(_IMAGES_PER_WELL)
    ]
    with ProcessPoolExecutor(
        max_workers=_NUM_WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_add_image, jobs))

    # Guard against the test quietly becoming a single-process one again, which
    # is how the threaded version of this file passed while the lock was broken.
    pids = {pid for _, pid in results}
    assert os.getpid() not in pids
    assert len(pids) > 1

    expected_wells, expected_images = _expected_paths()
    plate = open_ome_zarr_plate(store, mode="r")
    # Sorted rather than set-compared: a duplicated entry is as wrong as a
    # missing one, and only the sorted list catches both.
    assert sorted(plate.wells_paths()) == expected_wells
    assert sorted(plate.images_paths()) == expected_images


def test_atomic_add_image_rejects_windows(tmp_path: Path, monkeypatch):
    """`atomic_*` refuses on Windows rather than losing an update silently.

    Simulated so the refusal is exercised on every platform, not only in the
    Windows CI jobs where the tests above do not run at all.
    """
    store = tmp_path / "plate.zarr"
    create_empty_plate(store, name="win_plate", ngff_version="0.5")
    plate = open_ome_zarr_plate(store, mode="r+")

    monkeypatch.setattr(_retry_module, "_IS_WINDOWS", True)
    with pytest.raises(NgioValueError, match="not exclusive on Windows"):
        plate.atomic_add_image(row="B", column="03", image_path="0")

    # The non-atomic path stays available for single-worker use.
    assert plate.add_image(row="B", column="03", image_path="0") == "B/03/0"
