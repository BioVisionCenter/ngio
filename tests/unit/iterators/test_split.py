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
    return [
        iterator.for_job(i, n_jobs=n_jobs).partition_indices
        for i in range(n_jobs)
    ]


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
        job_of = {
            index: job for job, indices in enumerate(layout) for index in indices
        }
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


def test_for_job_refuses_stitch(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "stitch.zarr", chunks=(1, 32, 32))
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")
    iterator = SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx", stitch=True
    ).by_grid(size_y=32, size_x=32)
    with pytest.raises(NgioValueError, match="stitch"):
        iterator.for_job(0, n_jobs=2)


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
    _fresh_iterator(ome_zarr).for_job(empty_job, n_jobs=n_jobs).map(
        _threshold
    )
    for path, level in _label_levels(ome_zarr).items():
        np.testing.assert_array_equal(level, before[path])


def test_slice_map_does_not_finalize_and_slice_finalize_raises(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "nofinalize.zarr")
    ome_zarr.derive_label("out")
    for job_index in range(2):
        _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=2).map(
            _threshold
        )

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
    _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=n_jobs).map(
        _threshold
    )


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
