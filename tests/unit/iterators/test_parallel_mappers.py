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
    threaded_it.map_as_numpy(_threshold, mapper=ThreadedMapper("auto"))

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
        read_tiled.map_as_numpy(_threshold, mapper=ThreadedMapper(4))

    # The named fix works: write-unit tiling passes.
    base.by_chunks(grid="write").map_as_numpy(_threshold, mapper=ThreadedMapper(4))


def test_process_mapper_refuses_memory_stores():
    _, iterator = _segmentation_iterator(MemoryStore())
    with pytest.raises(NgioValueError, match="MemoryStore pickles by value"):
        iterator.map_as_numpy(_threshold, mapper=ProcessMapper(max_workers=2))


def test_max_workers_is_not_an_iterator_argument():
    """Concurrency is the mapper's to own; there is no second spelling for it."""
    _, iterator = _segmentation_iterator(MemoryStore())
    with pytest.raises(TypeError, match="max_workers"):
        iterator.map_as_numpy(_threshold, max_workers="auto")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="max_workers"):
        iterator.reduce_as_numpy(_threshold, max_workers="auto")  # type: ignore[call-arg]


def test_threaded_mapper_with_one_unit_degrades_to_serial():
    ome_zarr = _build_ome_zarr(MemoryStore(), chunks=(1, 64, 64))
    label = ome_zarr.derive_label("out")
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    assert len(iterator.rois) == 1
    iterator.map_as_numpy(_threshold, mapper=ThreadedMapper("auto"))
    assert ome_zarr.get_label("out").get_as_numpy().max() <= 1


def test_zero_or_negative_max_workers_is_refused():
    from ngio.utils import NgioValueError

    with pytest.raises(NgioValueError, match="max_workers must be >= 1"):
        ThreadedMapper(0)
    with pytest.raises(NgioValueError, match="max_workers must be >= 1"):
        ProcessMapper(-2)


def _measure_labels(image, label, roi):
    ids = [int(value) for value in np.unique(label) if value]
    return {
        "label": ids,
        "mean": [float(image[label == i].mean()) for i in ids],
    }


def test_reduce_to_table_across_processes(tmp_path: Path):
    """Per-ROI dicts pickle back from the workers; the joined table matches serial."""
    from ngio.iterators import FeatureExtractorIterator

    ome_zarr = _build_ome_zarr(tmp_path / "features.zarr")
    label = ome_zarr.derive_label("nuclei")
    label_data = np.zeros((64, 64), dtype="uint32")
    label_data[4:12, 4:12] = 1
    label_data[40:48, 40:48] = 2
    label.set_array(label_data)
    label.consolidate()

    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.get_label("nuclei"),
        channel_selection=0,
        axes_order="yx",
    ).grid(size_x=16, size_y=16)

    serial = iterator.reduce_to_table(_measure_labels)
    from_processes = iterator.reduce_to_table(
        _measure_labels, mapper=ProcessMapper(max_workers=2)
    )
    assert from_processes.dataframe.equals(serial.dataframe)


def _detect_bright(patch, roi):
    ys, xs = np.nonzero(patch > 128)
    if not len(ys):
        return {"x_min": [], "x_max": [], "y_min": [], "y_max": [], "confidence": []}
    return {
        "x_min": [int(xs.min())],
        "x_max": [int(xs.max()) + 1],
        "y_min": [int(ys.min())],
        "y_max": [int(ys.max()) + 1],
        "confidence": [float((patch > 128).mean())],
    }


def test_detect_across_processes(tmp_path: Path):
    """Per-tile boxes pickle back from the workers; the table matches serial."""
    from ngio import ObjectDetectionIterator

    data = np.zeros((1, 64, 64), dtype="uint8")
    data[0, 10:20, 26:38] = 255
    data[0, 40:50, 8:18] = 255
    ome_zarr = create_ome_zarr_from_array(
        store=tmp_path / "detect.zarr",
        array=data,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        consolidation_mode="dask",
    )
    iterator = (
        ObjectDetectionIterator(
            ome_zarr.get_image(), channel_selection=0, axes_order="yx"
        )
        .grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    serial = iterator.detect(_detect_bright)
    from_processes = iterator.detect(
        _detect_bright, mapper=ProcessMapper(max_workers=2)
    )
    assert from_processes.dataframe.equals(serial.dataframe)
