"""Partition selection: embarrassing independence, determinism, bit-identity.

`for_job(job_index, n_jobs)` restricts an iterator to one of
`n_jobs` shares that never overlap at write granularity; its `map` runs the
share without finalizing, and the gather is a fresh unrestricted iterator's
`finalize()`. The tests mirror `test_parallel_mappers.py`'s discipline: every
execution shape must be bit-identical to the serial reference, and the
independence guarantee is asserted structurally (no cross-partition footprint
intersection), not by luck.

`func`s used across processes are module-level on purpose — that is the
picklability contract, and `_run_one_job` is the true SLURM simulation: each
process re-executes the user code and derives the partition independently.
"""

import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

from ngio import create_ome_zarr_from_array, open_ome_zarr_container
from ngio.io_pipes._ops_slices_utils import chunk_rects_intersect
from ngio.iterators import (
    FeatureExtractorIterator,
    SegmentationIterator,
    ThreadedMapper,
    write_conflict_components,
)
from ngio.utils import NgioValueError


def _build_ome_zarr(store, chunks=(1, 16, 16), levels=3):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    return create_ome_zarr_from_array(
        store=store,
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=levels,
        chunks=chunks,
        consolidation_mode="dask",
    )


def _threshold(patch):
    return (patch > 128).astype("uint32")


def _fresh_iterator(ome_zarr, name="out"):
    """The identical construction every job (and the gather) must repeat."""
    image = ome_zarr.get_image()
    label = ome_zarr.get_label(name)
    return SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx", consolidation_mode="dask"
    ).by_chunks()


def _partition_layout(iterator, n_jobs):
    return [iterator.for_job(i, n_jobs=n_jobs).partition_indices for i in range(n_jobs)]


def _label_levels(ome_zarr, name="out"):
    label = ome_zarr.get_label(name)
    handler = label._group_handler
    return {
        path: handler.get_array(path)[...]
        for path in label.meta_handler.get_meta().paths
    }


def _serial_reference(tmp_path):
    ome_zarr = _build_ome_zarr(tmp_path / "serial.zarr")
    ome_zarr.derive_label("out")
    _fresh_iterator(ome_zarr).map(_threshold)
    return _label_levels(ome_zarr)


@pytest.mark.parametrize("n_jobs", [1, 2, 3, 7])
@pytest.mark.parametrize("parallel", [False, True])
def test_jobs_in_any_order_match_serial(tmp_path: Path, n_jobs: int, parallel: bool):
    reference = _serial_reference(tmp_path)

    ome_zarr = _build_ome_zarr(tmp_path / "jobs.zarr")
    ome_zarr.derive_label("out")
    job_order = list(range(n_jobs))
    random.Random(42).shuffle(job_order)
    for job_index in job_order:
        # A fresh iterator per job: each derives the partition on its own,
        # exactly as separate SLURM tasks would.
        _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=n_jobs).map(
            _threshold,
            mapper=ThreadedMapper(2) if parallel else None,
        )
    _fresh_iterator(ome_zarr).finalize()

    for path, level in _label_levels(ome_zarr).items():
        np.testing.assert_array_equal(
            level, reference[path], err_msg=f"level {path} differs"
        )


def test_partitions_keep_conflicting_tiles_together(tmp_path: Path):
    """The independence guarantee, asserted structurally.

    Input chunks 16px, output chunks 32px: `by_chunks()` yields four read
    tiles per output write unit, so the conflict components have four units
    each — and no component may be divided across partitions.
    """
    ome_zarr = _build_ome_zarr(tmp_path / "conflict.zarr", chunks=(1, 16, 16))
    coarse = ome_zarr.derive_label("coarse", chunks=(1, 32, 32))
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(
        image, coarse, channel_selection=0, axes_order="yx"
    ).by_chunks()

    units = list(iterator._numpy_units_generator())
    components = write_conflict_components(units)
    assert any(len(component) > 1 for component in components)

    for n_jobs in (2, 3, 5):
        layout = _partition_layout(iterator, n_jobs)
        job_of = {index: job for job, indices in enumerate(layout) for index in indices}
        # Components stay whole ...
        for component in components:
            assert len({job_of[index] for index in component}) == 1
        # ... and no two units in different partitions intersect at write
        # granularity.
        footprints = {unit.index: unit.write_footprint for unit in units}
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                if job_of[a] != job_of[b]:
                    assert not chunk_rects_intersect(footprints[a], footprints[b])


def test_shifted_grid_overlaps_share_a_partition(tmp_path: Path):
    """`by_grid(tail="shift")` makes genuinely overlapping tiles: one job."""
    ome_zarr = _build_ome_zarr(tmp_path / "shift.zarr")
    ome_zarr.derive_label("out")
    iterator = _fresh_iterator(ome_zarr).by_grid(size_y=48, size_x=48, tail="shift")

    units = list(iterator._numpy_units_generator())
    components = write_conflict_components(units)
    # The shifted tail tiles overlap their neighbours: everything conflicts
    # into components larger than one unit.
    assert any(len(component) > 1 for component in components)

    layout = _partition_layout(iterator, 2)
    job_of = {index: job for job, indices in enumerate(layout) for index in indices}
    for component in components:
        assert len({job_of[index] for index in component}) == 1


def test_partition_is_deterministic(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "det.zarr")
    ome_zarr.derive_label("out")
    first = _partition_layout(_fresh_iterator(ome_zarr), 3)
    second = _partition_layout(_fresh_iterator(ome_zarr), 3)
    assert first == second
    # Every unit appears exactly once across the partitions.
    flat = sorted(index for job in first for index in job)
    assert flat == list(range(len(_fresh_iterator(ome_zarr).rois)))
    # An unrestricted iterator reports no selection.
    assert _fresh_iterator(ome_zarr).partition_indices is None


def test_prepare_jobs_drops_empty_partitions(tmp_path: Path):
    """The parallelization list never contains a job with nothing to do."""
    ome_zarr = _build_ome_zarr(tmp_path / "prepare.zarr")
    ome_zarr.derive_label("out")
    iterator = _fresh_iterator(ome_zarr)
    n_units = len(iterator.rois)

    args_list = iterator.prepare_jobs(n_jobs=n_units + 5)
    assert len(args_list) == n_units
    assert all(args["n_jobs"] == n_units + 5 for args in args_list)
    # The listed jobs are exactly the non-empty ones.
    for args in args_list:
        slice_ = _fresh_iterator(ome_zarr).for_job(**args)
        assert slice_.partition_indices


def test_prepare_jobs_refuses_on_a_slice(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "prepare_slice.zarr")
    ome_zarr.derive_label("out")
    restricted = _fresh_iterator(ome_zarr).for_job(0, n_jobs=2)
    with pytest.raises(NgioValueError, match="unrestricted"):
        restricted.prepare_jobs(n_jobs=2)


# --- stitched distributed runs (prepare -> jobs -> gather) ------------------


def _stitched_setup(store, chunks=(32, 32), levels=2):
    """An image with one object crossing the x=32 tile boundary."""
    data = np.zeros((64, 64), dtype="uint8")
    data[8:16, 24:40] = 255  # crosses x=32
    data[40:48, 8:16] = 255
    return create_ome_zarr_from_array(
        store=store,
        array=data,
        pixelsize=1.0,
        axes_names="yx",
        levels=levels,
        chunks=chunks,
        consolidation_mode="dask",
    )


def _label_components(patch):
    from scipy import ndimage

    labeled, _ = ndimage.label(patch > 128)
    return labeled.astype("uint32")


def _stitched_iterator(ome_zarr, size=32):
    return (
        SegmentationIterator(
            ome_zarr.get_image(),
            ome_zarr.get_label("seg"),
            axes_order="yx",
            consolidation_mode="dask",
            stitch=True,
        )
        .by_grid(size_y=size, size_x=size)
        .with_halo(y=4, x=4)
    )


def _stitched_serial_reference(tmp_path):
    ome_zarr = _stitched_setup(tmp_path / "stitch_serial.zarr")
    ome_zarr.derive_label("seg")
    _stitched_iterator(ome_zarr).map(_label_components)
    return ome_zarr.get_label("seg").get_as_numpy()


def test_stitched_jobs_match_serial_stitched(tmp_path: Path):
    """prepare -> shuffled jobs -> gather, bit-identical to a serial stitch."""
    reference = _stitched_serial_reference(tmp_path)

    ome_zarr = _stitched_setup(tmp_path / "stitch_jobs.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    assert all(set(args) == {"job_index", "n_jobs"} for args in args_list)

    for args in reversed(args_list):
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)
    _stitched_iterator(ome_zarr).finalize()

    np.testing.assert_array_equal(ome_zarr.get_label("seg").get_as_numpy(), reference)
    # The gather resolved and cleaned up: the scratch group is gone.
    label = ome_zarr.get_label("seg")
    assert "_ngio_stitch" not in list(label._group_handler.group.keys())


def test_stitched_for_job_requires_prepare(tmp_path: Path):
    ome_zarr = _stitched_setup(tmp_path / "stitch_noprep.zarr")
    ome_zarr.derive_label("seg")
    with pytest.raises(NgioValueError, match="prepare_jobs"):
        _stitched_iterator(ome_zarr).for_job(0, n_jobs=2)


def test_stitched_fingerprint_mismatch_refuses(tmp_path: Path):
    """A job built with a different tiling than the prepared one fails loud."""
    ome_zarr = _stitched_setup(tmp_path / "stitch_drift.zarr")
    ome_zarr.derive_label("seg")
    _stitched_iterator(ome_zarr, size=32).prepare_jobs(n_jobs=2)

    retiled = _stitched_iterator(ome_zarr, size=16)
    with pytest.raises(NgioValueError, match="different plan"):
        retiled.for_job(0, n_jobs=2)
    # Wrong n_jobs against a matching tiling fails the same way.
    with pytest.raises(NgioValueError, match="different plan"):
        _stitched_iterator(ome_zarr, size=32).for_job(0, n_jobs=3)


def test_failed_job_leaves_scratch_and_reprepare_resets(tmp_path: Path):
    ome_zarr = _stitched_setup(tmp_path / "stitch_fail.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)

    _stitched_iterator(ome_zarr).for_job(**args_list[0]).map(_label_components)

    def boom(patch):
        raise RuntimeError("job crashed")

    with pytest.raises(RuntimeError):
        _stitched_iterator(ome_zarr).for_job(**args_list[1]).map(boom)

    # The crash must not have destroyed the other job's banked bands.
    label = ome_zarr.get_label("seg")
    assert "_ngio_stitch" in list(label._group_handler.group.keys())

    # Re-running the failed job and gathering still matches serial.
    _stitched_iterator(ome_zarr).for_job(**args_list[1]).map(_label_components)
    _stitched_iterator(ome_zarr).finalize()
    np.testing.assert_array_equal(
        label.get_as_numpy(), _stitched_serial_reference(tmp_path)
    )


def test_standalone_stitched_map_still_works(tmp_path: Path):
    """The classic single-call stitched map needs no prepare and cleans up."""
    ome_zarr = _stitched_setup(tmp_path / "stitch_solo.zarr")
    ome_zarr.derive_label("seg")
    _stitched_iterator(ome_zarr).map(_label_components)
    label = ome_zarr.get_label("seg")
    assert "_ngio_stitch" not in list(label._group_handler.group.keys())
    np.testing.assert_array_equal(
        label.get_as_numpy(), _stitched_serial_reference(tmp_path)
    )


def test_misaligned_stitched_bands_couple_components(tmp_path: Path):
    """Tiles finer than the label chunks: band claims coarsen the partition.

    Without the band claims, tiles in different label chunks would look
    independent while banking into the same scratch chunk — the read-modify-
    write race the claims exist to prevent.
    """
    ome_zarr = _stitched_setup(tmp_path / "stitch_misaligned.zarr")
    ome_zarr.derive_label("seg")

    plain = SegmentationIterator(
        ome_zarr.get_image(),
        ome_zarr.get_label("seg"),
        axes_order="yx",
        consolidation_mode="dask",
    ).by_grid(size_y=16, size_x=16)
    stitched = _stitched_iterator(ome_zarr, size=16)

    plain_components = write_conflict_components(list(plain._numpy_units_generator()))
    stitched_components = write_conflict_components(
        list(stitched._numpy_units_generator())
    )
    assert len(stitched_components) < len(plain_components)


def _run_stitched_job(store: str, job_index: int, n_jobs: int) -> None:
    """One array task of a distributed stitched run."""
    ome_zarr = open_ome_zarr_container(store)
    _stitched_iterator(ome_zarr).for_job(job_index=job_index, n_jobs=n_jobs).map(
        _label_components
    )


def test_stitched_jobs_across_processes_match_serial(tmp_path: Path):
    reference = _stitched_serial_reference(tmp_path)

    store = tmp_path / "stitch_procs.zarr"
    ome_zarr = _stitched_setup(store)
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)

    with ProcessPoolExecutor(
        max_workers=len(args_list), mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_run_stitched_job, str(store), **args) for args in args_list
        ]
        for future in futures:
            future.result()

    _stitched_iterator(ome_zarr).finalize()
    np.testing.assert_array_equal(ome_zarr.get_label("seg").get_as_numpy(), reference)


def test_for_job_refuses_readonly(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "readonly.zarr")
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")
    iterator = FeatureExtractorIterator(image, label, axes_order="yx")
    with pytest.raises(NgioValueError, match="read-only"):
        iterator.for_job(0, n_jobs=2)


def test_for_job_validates_indices(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "validate.zarr")
    ome_zarr.derive_label("out")
    iterator = _fresh_iterator(ome_zarr)
    with pytest.raises(NgioValueError, match="n_jobs"):
        iterator.for_job(0, n_jobs=0)
    with pytest.raises(NgioValueError, match="job_index"):
        iterator.for_job(-1, n_jobs=2)
    with pytest.raises(NgioValueError, match="job_index"):
        iterator.for_job(2, n_jobs=2)


def test_slice_is_the_last_builder_call(tmp_path: Path):
    """Reshaping or re-partitioning a slice refuses: selection is positional."""
    ome_zarr = _build_ome_zarr(tmp_path / "last.zarr")
    ome_zarr.derive_label("out")
    restricted = _fresh_iterator(ome_zarr).for_job(0, n_jobs=2)

    with pytest.raises(NgioValueError, match="for_job"):
        restricted.by_grid(size_y=32, size_x=32)
    with pytest.raises(NgioValueError, match="for_job"):
        restricted.with_halo(y=2, x=2)
    with pytest.raises(NgioValueError, match="for_job"):
        restricted.product(restricted.rois)
    with pytest.raises(NgioValueError, match="do not nest"):
        restricted.for_job(0, n_jobs=2)


def test_empty_partition_is_a_noop(tmp_path: Path):
    """More jobs than components: the surplus partitions run and write nothing."""
    ome_zarr = _build_ome_zarr(tmp_path / "empty.zarr")
    ome_zarr.derive_label("out")
    iterator = _fresh_iterator(ome_zarr)
    n_jobs = len(iterator.rois) + 3
    layout = _partition_layout(iterator, n_jobs)
    assert [] in layout

    before = _label_levels(ome_zarr)
    empty_job = layout.index([])
    _fresh_iterator(ome_zarr).for_job(empty_job, n_jobs=n_jobs).map(_threshold)
    for path, level in _label_levels(ome_zarr).items():
        np.testing.assert_array_equal(level, before[path])


def test_slice_map_does_not_finalize_and_slice_finalize_raises(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "nofinalize.zarr")
    ome_zarr.derive_label("out")
    for job_index in range(2):
        _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=2).map(_threshold)

    levels = _label_levels(ome_zarr)
    paths = sorted(levels, key=lambda p: int(p))
    assert levels[paths[0]].any(), "level 0 must hold the written labels"
    for path in paths[1:]:
        assert not levels[path].any(), (
            f"level {path} must still be stale before finalize()"
        )

    restricted = _fresh_iterator(ome_zarr).for_job(0, n_jobs=2)
    with pytest.raises(NgioValueError, match="gather"):
        restricted.finalize()

    _fresh_iterator(ome_zarr).finalize()
    assert _label_levels(ome_zarr)[paths[1]].any()


def _run_one_job(store: str, job_index: int, n_jobs: int) -> None:
    """One SLURM array task: reopen, rebuild identically, run its share."""
    ome_zarr = open_ome_zarr_container(store)
    _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=n_jobs).map(_threshold)


def test_jobs_across_processes_match_serial(tmp_path: Path):
    """Concurrent worker processes, each deriving the partition on its own."""
    reference = _serial_reference(tmp_path)

    store = tmp_path / "processes.zarr"
    ome_zarr = _build_ome_zarr(store)
    ome_zarr.derive_label("out")

    n_jobs = 3
    with ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_run_one_job, str(store), job_index, n_jobs)
            for job_index in range(n_jobs)
        ]
        for future in futures:
            future.result()

    _fresh_iterator(ome_zarr).finalize()

    for path, level in _label_levels(ome_zarr).items():
        np.testing.assert_array_equal(
            level, reference[path], err_msg=f"level {path} differs"
        )
