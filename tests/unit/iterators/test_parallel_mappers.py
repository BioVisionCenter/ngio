"""The parallel mappers: bit-identity, wave scheduling, and process safety.

`func`s used with `ProcessMapper` are module-level on purpose — that is the
picklability contract the mapper documents.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import Roi, create_ome_zarr_from_array
from ngio.io_pipes._ops_slices_utils import chunk_rects_intersect
from ngio.iterators import (
    MaskedSegmentationIterator,
    ProcessMapper,
    SegmentationIterator,
    ThreadedMapper,
)
from ngio.iterators._mappers import plan_waves
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
    return ome_zarr, iterator.by_write_units()


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


def _colliding_iterator(store):
    """A segmentation iterator whose read tiling collides on the output.

    Input chunks are 16x16 and the output label's are 32x32, so `by_chunks()`
    produces four read tiles per output write unit.
    """
    ome_zarr = _build_ome_zarr(store, chunks=(1, 16, 16))
    coarse = ome_zarr.derive_label("coarse", chunks=(1, 32, 32))
    image = ome_zarr.get_image()
    base = SegmentationIterator(image, coarse, channel_selection=0, axes_order="yx")
    return ome_zarr, base


def test_parallel_map_schedules_colliding_write_units_in_waves():
    """ROIs sharing an output write unit run in separate waves, not refused."""
    serial_zarr, serial_base = _colliding_iterator(MemoryStore())
    serial_base.by_chunks().map_as_numpy(_threshold)

    threaded_zarr, threaded_base = _colliding_iterator(MemoryStore())
    read_tiled = threaded_base.by_chunks()
    assert read_tiled.check_if_write_units_overlap()
    read_tiled.map_as_numpy(_threshold, mapper=ThreadedMapper(4))

    np.testing.assert_array_equal(
        serial_zarr.get_label("coarse").get_as_numpy(),
        threaded_zarr.get_label("coarse").get_as_numpy(),
    )

    # Write-unit tiling is conflict-free by construction: a single wave.
    single = plan_waves(list(threaded_base.by_write_units()._numpy_units_generator()))
    assert len(single) == 1


def test_plan_waves_partition_is_conflict_free_and_deterministic():
    _, base = _colliding_iterator(MemoryStore())
    units = list(base.by_chunks()._numpy_units_generator())

    waves = plan_waves(units)
    assert len(waves) > 1
    assert sorted(u.index for wave in waves for u in wave) == [u.index for u in units]
    for wave in waves:
        footprints = [u.write_footprint for u in wave]
        for i, rect in enumerate(footprints):
            for other in footprints[i + 1 :]:
                assert not chunk_rects_intersect(rect, other)

    again = plan_waves(units)
    assert [[u.index for u in wave] for wave in again] == [
        [u.index for u in wave] for wave in waves
    ]

    # Read-only units conflict with nothing: always a single wave.
    readonly = plan_waves(
        list(base.by_chunks()._numpy_units_generator(with_setters=False))
    )
    assert len(readonly) == 1


def test_waves_conflict_across_distinct_handles_of_one_array():
    """Two `zarr.Array` handles onto one stored array are one array.

    `get_label` builds a fresh handle per call; keying the conflict graph on
    handle identity would let two units targeting the same write unit share a
    wave — a silent lost update. The graph must key on the stored array
    (store + path).
    """
    ome_zarr, base = _colliding_iterator(MemoryStore())
    units_a = list(base.by_chunks()._numpy_units_generator())
    other = SegmentationIterator(
        ome_zarr.get_image(),
        ome_zarr.get_label("coarse"),
        channel_selection=0,
        axes_order="yx",
    )
    units_b = list(other.by_chunks()._numpy_units_generator())

    first = units_a[0]
    assert first.setter is not None
    partner = next(
        unit
        for unit in units_b
        if unit.index != first.index
        and unit.setter is not None
        and unit.write_footprint is not None
        and first.write_footprint is not None
        and chunk_rects_intersect(first.write_footprint, unit.write_footprint)
    )
    assert partner.setter is not None
    assert first.setter.zarr_array is not partner.setter.zarr_array

    waves = plan_waves([first, partner])
    assert len(waves) == 2


def test_process_mapper_runs_waves_across_one_pool(tmp_path: Path):
    serial_zarr, serial_base = _colliding_iterator(tmp_path / "serial.zarr")
    serial_base.by_chunks().map_as_numpy(_threshold)

    process_zarr, process_base = _colliding_iterator(tmp_path / "process.zarr")
    process_base.by_chunks().map_as_numpy(
        _threshold, mapper=ProcessMapper(max_workers=2)
    )

    np.testing.assert_array_equal(
        serial_zarr.get_label("coarse").get_as_numpy(),
        process_zarr.get_label("coarse").get_as_numpy(),
    )


def _single_chunk_output_iterator(store):
    ome_zarr = _build_ome_zarr(store, chunks=(1, 16, 16))
    out = ome_zarr.derive_label("out", chunks=(1, 64, 64))
    image = ome_zarr.get_image()
    base = SegmentationIterator(image, out, channel_selection=0, axes_order="yx")
    return ome_zarr, base.by_chunks()


def test_fully_conflicting_units_run_serially_and_warn(caplog):
    """Every unit in one output chunk: a serial schedule, correct and logged."""
    serial_zarr, serial_it = _single_chunk_output_iterator(MemoryStore())
    serial_it.map_as_numpy(_threshold)

    threaded_zarr, threaded_it = _single_chunk_output_iterator(MemoryStore())
    with caplog.at_level(logging.WARNING, logger="ngio:ngio.iterators._mappers"):
        threaded_it.map_as_numpy(_threshold, mapper=ThreadedMapper(4))
    assert any("serial schedule" in record.message for record in caplog.records)

    np.testing.assert_array_equal(
        serial_zarr.get_label("out").get_as_numpy(),
        threaded_zarr.get_label("out").get_as_numpy(),
    )


def _masked_iterator(store):
    """A masked segmentation iterator whose objects share output chunks."""
    ome_zarr = _build_ome_zarr(store, chunks=(1, 16, 16))
    masking = ome_zarr.derive_label("masking")
    mask = np.zeros(masking.shape, dtype="uint32")
    mask[..., 2:10, 2:10] = 1  # chunk (0, 0)
    mask[..., 12:20, 4:12] = 2  # spans chunk rows 0-1, shares (0, 0) with 1
    mask[..., 40:56, 40:56] = 3  # four chunks of its own
    masking.set_array(mask)
    masking.consolidate()
    ome_zarr.add_table("masking_ROI_table", masking.build_masking_roi_table())

    image = ome_zarr.get_masked_image(masking_label_name="masking")
    out = ome_zarr.derive_label("out")
    iterator = MaskedSegmentationIterator(
        image, out, channel_selection=0, axes_order="yx"
    )
    return ome_zarr, iterator


def test_masked_segmentation_parallel_matches_serial():
    """Chunk-sharing objects parallelize in waves, bit-identical to serial."""
    serial_zarr, serial_it = _masked_iterator(MemoryStore())
    assert serial_it.check_if_write_units_overlap()
    serial_it.map_as_numpy(_threshold)

    threaded_zarr, threaded_it = _masked_iterator(MemoryStore())
    threaded_it.map_as_numpy(_threshold, mapper=ThreadedMapper(4))

    np.testing.assert_array_equal(
        serial_zarr.get_label("out").get_as_numpy(),
        threaded_zarr.get_label("out").get_as_numpy(),
    )


def _overlapping_bbox_iterator(store):
    """Two objects whose *bounding boxes* overlap but whose pixels do not.

    The organoid case: bbox overlap forces the objects into the same write
    units, so correctness rests on wave scheduling plus the masked write's
    read-modify-write protection — not on disjoint boxes.
    """
    ome_zarr = _build_ome_zarr(store, chunks=(1, 16, 16))
    masking = ome_zarr.derive_label("masking")
    mask = np.zeros(masking.shape, dtype="uint32")
    # Two diagonal blobs: bboxes [4:28, 4:28] and [20:44, 20:44] overlap in
    # [20:28, 20:28], the pixels stay disjoint.
    mask[..., 4:28, 4:20] = 1
    mask[..., 4:20, 20:28] = 1
    mask[..., 28:44, 20:44] = 2
    mask[..., 20:28, 28:44] = 2
    masking.set_array(mask)
    masking.consolidate()
    ome_zarr.add_table("masking_ROI_table", masking.build_masking_roi_table())

    image = ome_zarr.get_masked_image(masking_label_name="masking")
    out = ome_zarr.derive_label("out")
    iterator = MaskedSegmentationIterator(
        image, out, channel_selection=0, axes_order="yx"
    )
    return ome_zarr, iterator


def test_masked_overlapping_bboxes_parallel_matches_serial():
    """Overlapping bounding boxes with disjoint pixels stay bit-correct."""
    serial_zarr, serial_it = _overlapping_bbox_iterator(MemoryStore())
    assert serial_it.check_if_regions_overlap()
    serial_it.map_as_numpy(_threshold)

    threaded_zarr, threaded_it = _overlapping_bbox_iterator(MemoryStore())
    threaded_it.map_as_numpy(_threshold, mapper=ThreadedMapper(4))

    serial = serial_zarr.get_label("out").get_as_numpy()
    np.testing.assert_array_equal(serial, threaded_zarr.get_label("out").get_as_numpy())
    # The masked write protects each object's pixels from its neighbour: the
    # second object's pass over the shared bbox must not clear the first —
    # one tile's background 0 never overwrites another tile's written ids.
    mask = serial_zarr.get_label("masking").get_as_numpy()
    assert (serial[mask == 0] == 0).all()
    for object_label in (1, 2):
        inside = serial[mask == object_label]
        assert inside.any(), f"object {object_label} lost its written ids"


def test_serial_map_matches_parallel_on_overlapping_writes():
    """BasicMapper adopts wave order: contested pixels land identically."""
    from ngio.iterators._mappers import canonical_unit_order

    def _fill_mean(patch):
        return np.full_like(patch, int(patch.mean()) or 1)

    serial_zarr, serial_base = _colliding_iterator(MemoryStore())
    serial_it = serial_base.by_grid(size_y=24, size_x=24, tail="shift")
    assert serial_it.check_if_regions_overlap()
    serial_it.map_as_numpy(_fill_mean)

    threaded_zarr, threaded_base = _colliding_iterator(MemoryStore())
    threaded_base.by_grid(size_y=24, size_x=24, tail="shift").map_as_numpy(
        _fill_mean, mapper=ThreadedMapper(4)
    )

    np.testing.assert_array_equal(
        serial_zarr.get_label("coarse").get_as_numpy(),
        threaded_zarr.get_label("coarse").get_as_numpy(),
    )

    # With no conflicts the canonical order is plain index order.
    _, disjoint = _segmentation_iterator(MemoryStore())
    units = list(disjoint._numpy_units_generator())
    assert [u.index for u in canonical_unit_order(units)] == [u.index for u in units]


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


def test_measure_across_processes(tmp_path: Path):
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
    ).by_grid(size_x=16, size_y=16)

    serial = iterator.measure(_measure_labels)
    from_processes = iterator.measure(
        _measure_labels, mapper=ProcessMapper(max_workers=2)
    )
    assert serial is not None and from_processes is not None
    assert from_processes.dataframe.equals(serial.dataframe)


def _detect_bright(patch):
    ys, xs = np.nonzero(patch > 128)
    if not len(ys):
        return []
    x_min, y_min = int(xs.min()), int(ys.min())
    return [
        Roi.from_values(
            slices={
                "x": (x_min, int(xs.max()) + 1 - x_min),
                "y": (y_min, int(ys.max()) + 1 - y_min),
            },
            name=None,
            space="pixel",
            confidence=float((patch > 128).mean()),
        )
    ]


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
        .by_grid(size_x=32, size_y=32)
        .with_halo(x=8, y=8)
    )

    serial = iterator.detect(_detect_bright)
    from_processes = iterator.detect(
        _detect_bright, mapper=ProcessMapper(max_workers=2)
    )
    assert serial is not None and from_processes is not None
    assert from_processes.dataframe.equals(serial.dataframe)
