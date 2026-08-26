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
    # Overlapping segmentation writes need a declared resolution; "last" is
    # exactly the wave-order behavior this test pins.
    serial_it = serial_base.by_grid(size_y=24, size_x=24, tail="shift").on_overlap(
        "last"
    )
    assert serial_it.check_if_regions_overlap()
    serial_it.map_as_numpy(_fill_mean)

    threaded_zarr, threaded_base = _colliding_iterator(MemoryStore())
    threaded_base.by_grid(size_y=24, size_x=24, tail="shift").on_overlap(
        "last"
    ).map_as_numpy(_fill_mean, mapper=ThreadedMapper(4))

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
    from ngio.iterators import ObjectDetectionIterator

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


def test_map_is_bit_identical_to_iter_on_overlapping_writes():
    """On contested pixels `map` and the manual `iter` loop write the same image.

    The canonical order puts the higher ROI index last wherever pixels
    genuinely overlap, which is exactly the order the manual loop runs in.
    """

    def _fill_mean(patch):
        return np.full_like(patch, int(patch.mean()) or 1)

    mapped_zarr, mapped_base = _colliding_iterator(MemoryStore())
    mapped_base.by_grid(size_x=24, stride_x=18).on_overlap(
        "last", write_order="roi"
    ).map_as_numpy(_fill_mean)

    looped_zarr, looped_base = _colliding_iterator(MemoryStore())
    looped = looped_base.by_grid(size_x=24, stride_x=18).on_overlap(
        "last", write_order="roi"
    )
    for patch, writer in looped.iter(lazy=False, data_mode="numpy"):
        writer(_fill_mean(patch))

    np.testing.assert_array_equal(
        mapped_zarr.get_label("coarse").get_as_numpy(),
        looped_zarr.get_label("coarse").get_as_numpy(),
    )


def test_canonical_order_is_index_monotone_on_pixel_overlaps():
    """Every pixel-overlapping pair runs lower index first, chain included.

    `by_grid(size_x=24, stride_x=18)` overlaps each tile with its successor
    (a chain, the shape greedy colouring used to invert), so the canonical
    order must be plain index order here.
    """
    from ngio.iterators._mappers import canonical_unit_order

    _, base = _colliding_iterator(MemoryStore())
    chained = base.by_grid(size_x=24, stride_x=18).on_overlap("last", write_order="roi")
    units = list(chained._numpy_units_generator())
    order = [u.index for u in canonical_unit_order(units)]
    assert order == sorted(order)


def test_pixel_disjoint_collisions_keep_packed_waves():
    """Write-unit sharing without pixel overlap keeps the packed schedule.

    `by_chunks()` into a coarser output shares write units clique-wise but
    writes disjoint pixels — the index-precedence rule must not serialise it.
    """
    _, base = _colliding_iterator(MemoryStore())
    units = list(base.by_chunks()._numpy_units_generator())
    waves = plan_waves(units, log=False)
    assert len(waves) == 4
    assert all(len(wave) == 4 for wave in waves)


def test_write_order_any_collapses_the_wavefront():
    """All-"any" units schedule for parallelism alone; all-"roi" stays pinned."""
    _, base = _colliding_iterator(MemoryStore())
    overlapping = base.by_grid(size_x=24, stride_x=18)

    ordered_units = list(
        overlapping.on_overlap("last", write_order="roi")._numpy_units_generator()
    )
    relaxed_units = list(
        overlapping.on_overlap("last", write_order="any")._numpy_units_generator()
    )
    # The undeclared kwarg defaults to "any" — pin it.
    default_units = list(overlapping.on_overlap("last")._numpy_units_generator())
    assert all(u.write_order == "any" for u in default_units)
    ordered_waves = plan_waves(ordered_units, log=False)
    relaxed_waves = plan_waves(relaxed_units, log=False)

    # The chain serializes under ordering; safety-only packs it tighter.
    assert len(relaxed_waves) < len(ordered_waves)
    # Safety still holds: pairwise-disjoint footprints per wave.
    for wave in relaxed_waves:
        footprints = [u.write_footprint for u in wave]
        for i, fi in enumerate(footprints):
            for fj in footprints[i + 1 :]:
                assert not chunk_rects_intersect(fi, fj)
    # And the schedule is a pure function of the unit sequence.
    again = plan_waves(relaxed_units, log=False)
    assert [[u.index for u in w] for w in again] == [
        [u.index for u in w] for w in relaxed_waves
    ]


def test_write_order_mixed_stream_keeps_roi_precedence():
    """One "roi" unit among "any" units keeps its ordering edges."""
    from dataclasses import replace

    _, base = _colliding_iterator(MemoryStore())
    units = list(
        base.by_grid(size_x=24, stride_x=18)
        .on_overlap("last", write_order="any")
        ._numpy_units_generator()
    )
    # Unit 1 insists on the canonical order; its overlapping neighbours
    # (0 and 2) must still schedule around it in index order.
    units[1] = replace(units[1], write_order="roi")
    waves = plan_waves(units, log=False)
    wave_of = {u.index: i for i, w in enumerate(waves) for u in w}
    assert wave_of[0] < wave_of[1] < wave_of[2]


def test_write_order_any_components_and_results_unchanged():
    """Job splitting and result order are order-agnostic by design."""
    from ngio.iterators import write_conflict_components

    _, base = _colliding_iterator(MemoryStore())
    overlapping = base.by_grid(size_x=24, stride_x=18)
    ordered = list(
        overlapping.on_overlap("last", write_order="roi")._numpy_units_generator()
    )
    relaxed = list(
        overlapping.on_overlap("last", write_order="any")._numpy_units_generator()
    )
    assert write_conflict_components(ordered) == write_conflict_components(relaxed)


def test_write_order_any_map_is_exact_for_commutative_merges():
    """A commutative merge gives identical pixels under either order."""

    def _fill_mean(patch):
        return np.full_like(patch, int(patch.mean()) or 1)

    roi_zarr, roi_base = _colliding_iterator(MemoryStore())
    roi_base.by_grid(size_x=24, stride_x=18).on_overlap(
        "max", write_order="roi"
    ).map_as_numpy(_fill_mean, mapper=ThreadedMapper(4))

    any_zarr, any_base = _colliding_iterator(MemoryStore())
    any_base.by_grid(size_x=24, stride_x=18).on_overlap(
        "max", write_order="any"
    ).map_as_numpy(_fill_mean, mapper=ThreadedMapper(4))

    np.testing.assert_array_equal(
        roi_zarr.get_label("coarse").get_as_numpy(),
        any_zarr.get_label("coarse").get_as_numpy(),
    )


def test_write_order_any_last_holds_exactly_one_tile_per_pixel():
    """Under "any" + "last" every contested pixel holds one tile's value."""

    def _fill_index(patch):
        # Each tile's fill is its own mean-derived constant; with a
        # constant-4 input all tiles agree, so write a per-call marker.
        _fill_index.calls += 1
        return np.full_like(patch, _fill_index.calls)

    _fill_index.calls = 0
    ome_zarr, base = _colliding_iterator(MemoryStore())
    it = base.by_grid(size_x=24, stride_x=18).on_overlap("last", write_order="any")
    it.map_as_numpy(_fill_index)
    written = ome_zarr.get_label("coarse").get_as_numpy()
    # Whole image covered, and every pixel holds exactly one tile's marker.
    assert set(np.unique(written)).issubset(set(range(1, _fill_index.calls + 1)))
    assert (written > 0).all()


def test_masked_units_schedule_for_parallelism_alone():
    """Mask-protected writes never contest a pixel: no precedence edges."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        chunks=(1, 64, 64),
        consolidation_mode="dask",
    )
    mask = ome_zarr.derive_label("mask")
    patch = np.zeros(mask.shape, dtype="uint8")
    # Two adjacent objects whose bounding boxes overlap in pixel space.
    patch[..., 8:40, 8:40] = 1
    patch[..., 24:56, 24:56] = 2
    mask.set_array(patch)
    mask.consolidate()
    ome_zarr.get_masked_image(masking_label_name="mask")
    out = ome_zarr.derive_label("out", chunks=(1, 64, 64))
    iterator = MaskedSegmentationIterator(
        ome_zarr.get_masked_image(masking_label_name="mask"),
        out,
        axes_order="yx",
    )
    units = list(iterator._numpy_units_generator())
    assert all(unit.write_order == "any" for unit in units)
