"""The parallel mappers: bit-identity, the collision gate, and process safety.

`func`s used with `ProcessMapper` are module-level on purpose — that is the
picklability contract the mapper documents.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import ProcessMapper, SegmentationIterator, ThreadedMapper
from ngio.utils import NgioValueError


def _build_ome_zarr(store, chunks=(1, 16, 16), ngff_version="0.4"):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    return create_ome_zarr_from_array(
        store=store,
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        chunks=chunks,
        ngff_version=ngff_version,
        consolidation_mode="dask",
    )


def _threshold(patch):
    return (patch > 128).astype("uint8")


def _mean_and_pid(patch):
    return float(patch.mean()), os.getpid()


def _segmentation_iterator(store):
    ome_zarr = _build_ome_zarr(store)
    label = ome_zarr.derive_label("out")
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx", consolidation_mode="dask"
    )
    return ome_zarr, iterator.by_chunks(grid="write")


def test_threaded_map_is_bit_identical_to_serial():
    serial_zarr, serial_it = _segmentation_iterator(MemoryStore())
    serial_it.map_as_numpy(_threshold)

    threaded_zarr, threaded_it = _segmentation_iterator(MemoryStore())
    threaded_it.map_as_numpy(_threshold, max_workers="auto")

    np.testing.assert_array_equal(
        serial_zarr.get_label("out").get_as_numpy(),
        threaded_zarr.get_label("out").get_as_numpy(),
    )


def test_process_map_is_bit_identical_to_serial(tmp_path: Path):
    serial_zarr, serial_it = _segmentation_iterator(tmp_path / "serial.zarr")
    serial_it.map_as_numpy(_threshold)

    process_zarr, process_it = _segmentation_iterator(tmp_path / "process.zarr")
    process_it.map_as_numpy(_threshold, mapper=ProcessMapper(max_workers=2))

    np.testing.assert_array_equal(
        serial_zarr.get_label("out").get_as_numpy(),
        process_zarr.get_label("out").get_as_numpy(),
    )


def test_reduce_runs_in_child_processes(tmp_path: Path):
    """Results come back in ROI order, computed in a process that is not ours.

    The parent's pid is the deterministic assertion; how the pool spreads
    units across its workers is scheduling noise and is not asserted.
    """
    _, iterator = _segmentation_iterator(tmp_path / "reduce.zarr")

    serial = iterator.reduce_as_numpy(_mean_and_pid)
    parallel = iterator.reduce_as_numpy(
        _mean_and_pid, mapper=ProcessMapper(max_workers=2)
    )

    assert [mean for mean, _ in parallel] == [mean for mean, _ in serial]
    parent = os.getpid()
    assert all(pid == parent for _, pid in serial)
    assert all(pid != parent for _, pid in parallel)


def test_map_returns_placeholders_from_processes(tmp_path: Path):
    """Written units do not ship their pixels back to the parent."""
    _, iterator = _segmentation_iterator(tmp_path / "placeholders.zarr")
    mapper = ProcessMapper(max_workers=2)
    results = mapper(_threshold, list(iterator._numpy_units_generator()))
    assert results == [None] * len(iterator.rois)


def test_parallel_map_refuses_colliding_write_units():
    """ROIs sharing an output write unit must refuse to parallelize, loudly."""
    ome_zarr = _build_ome_zarr(MemoryStore(), chunks=(1, 16, 16))
    coarse = ome_zarr.derive_label("coarse", chunks=(1, 32, 32))
    image = ome_zarr.get_image()
    base = SegmentationIterator(image, coarse, channel_selection=0, axes_order="yx")

    read_tiled = base.by_chunks(grid="read")
    assert read_tiled.check_if_chunks_overlap()
    with pytest.raises(NgioValueError, match="same write unit"):
        read_tiled.map_as_numpy(_threshold, max_workers=4)

    # The named fix works: write-unit tiling passes.
    base.by_chunks(grid="write").map_as_numpy(_threshold, max_workers=4)


def test_process_mapper_refuses_memory_stores():
    _, iterator = _segmentation_iterator(MemoryStore())
    with pytest.raises(NgioValueError, match="MemoryStore pickles by value"):
        iterator.map_as_numpy(_threshold, mapper=ProcessMapper(max_workers=2))


def test_mapper_and_max_workers_are_mutually_exclusive():
    _, iterator = _segmentation_iterator(MemoryStore())
    with pytest.raises(NgioValueError, match="not both"):
        iterator.map_as_numpy(
            _threshold, mapper=ThreadedMapper(), max_workers="auto"
        )
    # `None` and `1` say "serial", which a custom mapper overrides untroubled.
    iterator.map_as_numpy(_threshold, mapper=ThreadedMapper(), max_workers=1)


def test_threaded_mapper_with_one_unit_degrades_to_serial():
    ome_zarr = _build_ome_zarr(MemoryStore(), chunks=(1, 64, 64))
    label = ome_zarr.derive_label("out")
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    assert len(iterator.rois) == 1
    iterator.map_as_numpy(_threshold, max_workers="auto")
    assert ome_zarr.get_label("out").get_as_numpy().max() <= 1
