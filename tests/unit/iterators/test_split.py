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
    ObjectDetectionIterator,
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
        )
        .with_stitch()
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


def _boom(patch):
    raise RuntimeError("worker died")


def test_failed_gather_map_keeps_the_jobs_banks(tmp_path: Path):
    """A failed unrestricted map on a prepared plan must not delete the scratch.

    The unrestricted iterator *opens* a matching prepared root (a resumed run
    or the gather step) rather than creating it; the banks in it are every
    job's work, and one failure must not destroy them. Only a run that
    created its own scratch may clean it up on failure.
    """
    ome_zarr = _stitched_setup(tmp_path / "stitch_fail.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    for args in args_list:
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)

    with pytest.raises(RuntimeError, match="worker died"):
        _stitched_iterator(ome_zarr).map(_boom)

    label = ome_zarr.get_label("seg")
    assert "_ngio_stitch" in list(label._group_handler.group.keys())

    # The banks survived, so the gather still resolves and cleans up.
    _stitched_iterator(ome_zarr).finalize()
    label = ome_zarr.get_label("seg")
    assert "_ngio_stitch" not in list(label._group_handler.group.keys())


def _overlapping_stitched_iterator(ome_zarr):
    return (
        SegmentationIterator(
            ome_zarr.get_image(),
            ome_zarr.get_label("seg"),
            axes_order="yx",
            consolidation_mode="dask",
        )
        .with_stitch()
        .by_grid(size_y=32, size_x=32, stride_y=24, stride_x=24, tail="clip")
        .with_halo(y=4, x=4)
    )


def test_distributed_overlapping_tiles_match_serial(tmp_path: Path):
    """Overlapping tiles: one job by construction, and job ≡ serial.

    Overlapping cores share label chunks, so they form one conflict
    component and land in one job — that is what keeps contested pixels
    deterministic across a distributed run.
    """
    serial_zarr = _stitched_setup(tmp_path / "overlap_serial.zarr")
    serial_zarr.derive_label("seg")
    _overlapping_stitched_iterator(serial_zarr).map(_label_components)
    reference = serial_zarr.get_label("seg").get_as_numpy()

    ome_zarr = _stitched_setup(tmp_path / "overlap_jobs.zarr")
    ome_zarr.derive_label("seg")
    args_list = _overlapping_stitched_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    assert len(args_list) == 1, "overlapping tiles are one conflict component"

    for args in args_list:
        _overlapping_stitched_iterator(ome_zarr).for_job(**args).map(_label_components)
    _overlapping_stitched_iterator(ome_zarr).finalize()

    np.testing.assert_array_equal(ome_zarr.get_label("seg").get_as_numpy(), reference)


def test_stitched_finalize_lists_missing_banks(tmp_path: Path):
    """A half-finished run must error, naming the jobs that never banked."""
    ome_zarr = _stitched_setup(tmp_path / "stitch_missing.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    assert len(args_list) > 1

    # Run every job but the first.
    for args in args_list[1:]:
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)
    with pytest.raises(NgioValueError, match="never banked"):
        _stitched_iterator(ome_zarr).finalize()


def test_a_tile_killed_between_core_write_and_bank_fails_loud(
    tmp_path: Path, monkeypatch
):
    """A job killed after the core write but before banking is caught.

    The core lands first and the bank second, so this crash window leaves a
    written-but-unbanked tile that the gather refuses by name — the opposite
    order left a valid-looking bank and a tile silently missing from the
    output.
    """
    import ngio.iterators._stitch as stitch_mod

    ome_zarr = _stitched_setup(tmp_path / "stitch_kill.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    assert len(args_list) == 2
    _stitched_iterator(ome_zarr).for_job(**args_list[0]).map(_label_components)

    # Kill at the bank of the first tile with content: zero tiles bank
    # normally, so the killed tile is guaranteed to have visible core pixels.
    killed = []
    original_bank = stitch_mod.StitchPlan.bank

    def _kill_on_content(self, work, patch):
        if patch.any():
            killed.append(work)
            raise RuntimeError("killed after core write, before bank")
        return original_bank(self, work, patch)

    monkeypatch.setattr(stitch_mod.StitchPlan, "bank", _kill_on_content)
    with pytest.raises(RuntimeError, match="killed after core write"):
        _stitched_iterator(ome_zarr).for_job(**args_list[1]).map(_label_components)
    monkeypatch.undo()

    # The killed tile's core landed BEFORE the bank — this is the ordering
    # under test: a bank-first revert would raise before the core write and
    # leave this region empty.
    (y0, y1), (x0, x1) = killed[0].core
    label = ome_zarr.get_label("seg").get_as_numpy()
    assert (label[y0:y1, x0:x1] > 0).any()

    with pytest.raises(NgioValueError, match="never banked"):
        _stitched_iterator(ome_zarr).finalize()

    # Re-running the killed job is the documented recovery.
    _stitched_iterator(ome_zarr).for_job(**args_list[1]).map(_label_components)
    _stitched_iterator(ome_zarr).finalize()
    reference = _stitched_serial_reference(tmp_path)
    np.testing.assert_array_equal(ome_zarr.get_label("seg").get_as_numpy(), reference)


def test_interrupted_compaction_refuses_the_retry(tmp_path: Path, monkeypatch):
    """A finalize killed mid-compaction is refused on retry, not re-walked.

    The in-place renumbering is the one non-idempotent phase: re-resolving a
    half-compacted label silently splits every object straddling the crash
    point. The scratch carries a `resolving` marker for exactly this window.
    """
    import ngio.iterators._stitch as stitch_mod

    ome_zarr = _stitched_setup(tmp_path / "stitch_crash.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in args_list:
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)

    def _killed(*args, **kwargs):
        raise RuntimeError("killed mid-compaction")

    monkeypatch.setattr(stitch_mod, "relabel_sequential", _killed)
    with pytest.raises(RuntimeError, match="killed mid-compaction"):
        _stitched_iterator(ome_zarr).finalize()
    monkeypatch.undo()

    # The retry — and any job re-run against the marked scratch — refuses.
    with pytest.raises(NgioValueError, match="interrupted while compacting"):
        _stitched_iterator(ome_zarr).finalize()
    with pytest.raises(NgioValueError, match="interrupted while compacting"):
        _stitched_iterator(ome_zarr).for_job(**args_list[0]).map(_label_components)

    # `prepare_jobs` deliberately starts over: it clears the marker and the
    # regenerated run completes.
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in args_list:
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)
    _stitched_iterator(ome_zarr).finalize()
    reference = _stitched_serial_reference(tmp_path)
    np.testing.assert_array_equal(ome_zarr.get_label("seg").get_as_numpy(), reference)


def test_stitched_reprepare_invalidates_old_banks(tmp_path: Path):
    """Banks from a superseded prepare are stale, not silently resolved."""
    ome_zarr = _stitched_setup(tmp_path / "stitch_stale.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in args_list:
        _stitched_iterator(ome_zarr).for_job(**args).map(_label_components)

    # A second prepare wipes/invalidates everything the first run banked.
    _stitched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    with pytest.raises(NgioValueError, match="never banked"):
        _stitched_iterator(ome_zarr).finalize()


def test_stitched_for_job_requires_prepare(tmp_path: Path):
    ome_zarr = _stitched_setup(tmp_path / "stitch_noprep.zarr")
    ome_zarr.derive_label("seg")
    with pytest.raises(NgioValueError, match="prepare_jobs"):
        _stitched_iterator(ome_zarr).for_job(0, n_jobs=2)


def test_drifted_gather_refuses_and_preserves_banks(tmp_path: Path):
    """A consolidate task with the wrong plan must not wipe the jobs' banks."""
    ome_zarr = _stitched_setup(tmp_path / "stitch_drifted_gather.zarr")
    ome_zarr.derive_label("seg")
    args_list = _stitched_iterator(ome_zarr, size=32).prepare_jobs(n_jobs=2)
    for args in args_list:
        _stitched_iterator(ome_zarr, size=32).for_job(**args).map(_label_components)

    label = ome_zarr.get_label("seg")
    banks_before = sorted(label._group_handler.group["_ngio_stitch"].keys())
    assert banks_before, "jobs banked nothing"

    retiled = _stitched_iterator(ome_zarr, size=16)
    with pytest.raises(NgioValueError, match="Refusing to wipe"):
        retiled.finalize()
    banks_after = sorted(label._group_handler.group["_ngio_stitch"].keys())
    assert banks_after == banks_before

    # The matching gather still works afterwards.
    _stitched_iterator(ome_zarr, size=32).finalize()
    np.testing.assert_array_equal(
        label.get_as_numpy(), _stitched_serial_reference(tmp_path)
    )


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


def test_stitching_does_not_coarsen_the_partition(tmp_path: Path):
    """Banking claims nothing shared: each tile banks into its own array.

    Under the old shared-band scratch, stitching tiles finer than the label
    chunks coupled otherwise-independent tiles into one component. Per-ROI
    banks are conflict-free by construction, so a stitched iterator splits
    exactly like a plain one.
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
    assert stitched_components == plain_components


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


def test_readonly_for_job_requires_prepare(tmp_path: Path):
    ome_zarr = _build_ome_zarr(tmp_path / "readonly.zarr")
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")
    iterator = FeatureExtractorIterator(image, label, axes_order="yx")
    with pytest.raises(NgioValueError, match="prepare_jobs"):
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


# --- read-only distributed runs (prepare -> partials -> merge) --------------


def _feature_setup(store):
    rng = np.random.default_rng(0)
    ome_zarr = create_ome_zarr_from_array(
        store=store,
        array=rng.integers(0, 255, size=(64, 64)).astype("uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=(16, 16),
        consolidation_mode="dask",
    )
    label = ome_zarr.derive_label("objs")
    label_image = np.zeros((64, 64), dtype="uint32")
    label_image[4:12, 4:12] = 1
    label_image[20:28, 40:48] = 2
    label_image[50:60, 10:20] = 3
    label.set_array(label_image)
    label.consolidate(mode="dask")
    return ome_zarr


def _measure(image, label_patch, roi):
    rows = {"label": [], "mean_intensity": []}
    for obj in np.unique(label_patch):
        if obj == 0:
            continue
        rows["label"].append(int(obj))
        rows["mean_intensity"].append(float(image[label_patch == obj].mean()))
    return rows


def _feature_iterator(ome_zarr):
    return FeatureExtractorIterator(
        ome_zarr.get_image(), ome_zarr.get_label("objs"), axes_order="yx"
    ).by_grid(size_y=32, size_x=32)


def _global_norm_join(results):
    """Non-decomposable on purpose: normalizes by the global mean.

    Receives the normalized per-ROI frames (`roi_index`/`roi_name`
    included), identically on the serial and the distributed path — the
    frame equality asserted by the parity tests covers those columns too.
    """
    import pandas as pd

    from ngio.tables import FeatureTable

    frames = [frame for frame in results if len(frame)]
    joined = pd.concat(frames).set_index("label")
    joined["norm"] = joined["mean_intensity"] / joined["mean_intensity"].mean()
    return FeatureTable(table_data=joined, reference_label="objs")


@pytest.mark.parametrize("custom_join", [False, True])
def test_feature_merge_matches_serial(tmp_path: Path, custom_join: bool):
    import pandas as pd

    join = _global_norm_join if custom_join else None
    serial_oz = _feature_setup(tmp_path / "feat_serial.zarr")
    serial_it = _feature_iterator(serial_oz)
    if join is not None:
        serial_it = serial_it.with_join(join)
    serial = serial_it.measure(_measure)
    assert serial is not None

    ome_zarr = _feature_setup(tmp_path / "feat_jobs.zarr")
    args_list = _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in reversed(args_list):
        assert _feature_iterator(ome_zarr).for_job(**args).measure(_measure) is None
    gather_it = _feature_iterator(ome_zarr)
    if join is not None:
        gather_it = gather_it.with_join(join)
    merged = gather_it.finalize()

    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)
    # The merge cleaned the partials group up.
    label = ome_zarr.get_label("objs")
    assert "_ngio_partials" not in list(label._group_handler.group.keys())


def _hetero_measure(image, label_patch, roi):
    """Different column sets per ROI: `area` (int) odd objects, `score` even."""
    objs = [int(obj) for obj in np.unique(label_patch) if obj]
    rows: dict = {"label": objs}
    if any(obj % 2 for obj in objs):
        rows["area"] = [int((label_patch == obj).sum()) for obj in objs]
    if any(obj % 2 == 0 for obj in objs):
        rows["score"] = [float(image[label_patch == obj].mean()) for obj in objs]
    return rows


def _column_mean_join(results):
    """Reads `frame.columns` per ROI — sensitive to the column-union bug."""
    import pandas as pd

    from ngio.tables import FeatureTable

    rows = []
    for frame in results:
        if not len(frame):
            continue
        measured = [c for c in frame.columns if c not in ("roi_index", "roi_name")]
        rows.append(
            {
                "label": int(frame["label"].iloc[0]),
                "n_columns": len(measured),
                "columns": ",".join(measured),
            }
        )
    return FeatureTable(table_data=pd.DataFrame(rows), reference_label="objs")


@pytest.mark.parametrize("custom_join", [False, True])
def test_feature_merge_matches_serial_heterogeneous_columns(
    tmp_path: Path, custom_join: bool
):
    """Per-ROI column sets and dtypes survive the partial round-trip.

    The partial concat unions columns across ROIs and NaN-fills the gaps
    (upcasting ints to floats), so before the schema record a custom join
    saw different frames — and the default join different dtypes — on the
    distributed path than on the serial one, even at `n_jobs=1`.
    """
    import pandas as pd

    join = _column_mean_join if custom_join else None
    serial_oz = _feature_setup(tmp_path / "feat_het_serial.zarr")
    serial_it = _feature_iterator(serial_oz)
    if join is not None:
        serial_it = serial_it.with_join(join)
    serial = serial_it.measure(_hetero_measure)
    assert serial is not None
    if not custom_join:
        # The premise: the ROIs really produce different schemas.
        assert serial.dataframe["area"].isna().any()

    ome_zarr = _feature_setup(tmp_path / "feat_het_jobs.zarr")
    args_list = _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in reversed(args_list):
        assert (
            _feature_iterator(ome_zarr).for_job(**args).measure(_hetero_measure) is None
        )
    gather_it = _feature_iterator(ome_zarr)
    if join is not None:
        gather_it = gather_it.with_join(join)
    merged = gather_it.finalize()

    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)


def test_feature_slice_measure_banks_and_slice_finalize_refuses(tmp_path: Path):
    ome_zarr = _feature_setup(tmp_path / "feat_guard.zarr")
    iterator = _feature_iterator(ome_zarr)
    args_list = iterator.prepare_jobs(n_jobs=2)
    restricted = _feature_iterator(ome_zarr).for_job(**args_list[0])

    # A declared join is inert on a slice: the slice banks regardless.
    joined_slice = _feature_iterator(ome_zarr).with_join(_global_norm_join)
    restricted = joined_slice.for_job(**args_list[0])
    # The slice's measure banks its partial and hands nothing back.
    assert restricted.measure(_measure) is None
    label = ome_zarr.get_label("objs")
    partials = label._group_handler.group["_ngio_partials"]
    assert f"job_{args_list[0]['job_index']}" in list(partials.keys())
    # The gather stays global.
    with pytest.raises(NgioValueError, match="gather"):
        restricted.finalize()


def test_feature_merge_refuses_incomplete_run(tmp_path: Path):
    ome_zarr = _feature_setup(tmp_path / "feat_incomplete.zarr")
    args_list = _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    # Only one of two jobs runs.
    _feature_iterator(ome_zarr).for_job(**args_list[0]).measure(_measure)
    with pytest.raises(NgioValueError, match="incomplete"):
        _feature_iterator(ome_zarr).finalize()


def test_feature_merge_refuses_without_prepare(tmp_path: Path):
    ome_zarr = _feature_setup(tmp_path / "feat_noprep.zarr")
    with pytest.raises(NgioValueError, match="No partials"):
        _feature_iterator(ome_zarr).finalize()


def test_feature_fingerprint_drift_refuses(tmp_path: Path):
    ome_zarr = _feature_setup(tmp_path / "feat_drift.zarr")
    _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    retiled = FeatureExtractorIterator(
        ome_zarr.get_image(), ome_zarr.get_label("objs"), axes_order="yx"
    ).by_grid(size_y=16, size_x=16)
    with pytest.raises(NgioValueError, match="different plan"):
        retiled.for_job(0, n_jobs=2)
    with pytest.raises(NgioValueError, match="different plan"):
        retiled.finalize()


def _detection_setup(store):
    rng = np.random.default_rng(1)
    return create_ome_zarr_from_array(
        store=store,
        array=rng.integers(0, 255, size=(64, 64)).astype("uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        chunks=(16, 16),
        consolidation_mode="dask",
    )


def _detector(patch):
    from scipy import ndimage

    from ngio import Roi

    labeled, count = ndimage.label(patch > 200)
    boxes = []
    for obj in range(1, count + 1):
        ys, xs = np.where(labeled == obj)
        boxes.append(
            Roi.from_values(
                slices={
                    "x": (float(xs.min()), float(xs.max() + 1 - xs.min())),
                    "y": (float(ys.min()), float(ys.max() + 1 - ys.min())),
                },
                name=None,
                space="pixel",
                confidence=float(patch[labeled == obj].mean()),
            )
        )
    return boxes


def _detection_iterator(ome_zarr):
    return (
        ObjectDetectionIterator(ome_zarr.get_image(), axes_order="yx")
        .by_grid(size_y=32, size_x=32)
        .with_halo(y=8, x=8)
    )


def test_detection_merge_matches_serial(tmp_path: Path):
    """Global NMS at merge: bit-identical to serial `detect`.

    The noisy image produces hundreds of overlapping boxes across the haloed
    tile borders — dense suppression chains that a per-job NMS followed by a
    merge NMS (hierarchical) would resolve differently.
    """
    import pandas as pd

    serial_oz = _detection_setup(tmp_path / "det_serial.zarr")
    serial = _detection_iterator(serial_oz).detect(_detector)
    assert serial is not None
    assert len(serial.dataframe) > 50

    ome_zarr = _detection_setup(tmp_path / "det_jobs.zarr")
    args_list = _detection_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    for args in reversed(args_list):
        assert _detection_iterator(ome_zarr).for_job(**args).detect(_detector) is None
    merged = _detection_iterator(ome_zarr).finalize()

    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)
    image = ome_zarr.get_image()
    assert "_ngio_partials" not in list(image._group_handler.group.keys())


def _flagged_detector(patch):
    """`_detector` plus an integer `class_id` extra on even-count tiles only.

    Tile-homogeneous on purpose: the extra's column exists in some tiles'
    partials and not others, so the cross-tile concat used to NaN-fill and
    float-promote it.
    """
    from scipy import ndimage

    from ngio import Roi

    labeled, count = ndimage.label(patch > 200)
    flagged = count % 2 == 0
    boxes = []
    for obj in range(1, count + 1):
        ys, xs = np.where(labeled == obj)
        extras: dict = {"confidence": float(patch[labeled == obj].mean())}
        if flagged:
            extras["class_id"] = 7
        boxes.append(
            Roi.from_values(
                slices={
                    "x": (float(xs.min()), float(xs.max() + 1 - xs.min())),
                    "y": (float(ys.min()), float(ys.max() + 1 - ys.min())),
                },
                name=None,
                space="pixel",
                **extras,
            )
        )
    return boxes


def test_detection_merge_keeps_integer_extras(tmp_path: Path):
    """A tile's int extra survives the partial round-trip as an int.

    Tiles without the field used to NaN-fill its column at the merge,
    promoting the survivors to float — `class_id=7` came back `7.0` from
    `finalize()` while a serial `detect` returned `7`.
    """
    import pandas as pd

    serial_oz = _detection_setup(tmp_path / "det_int_serial.zarr")
    serial = _detection_iterator(serial_oz).detect(_flagged_detector)
    assert serial is not None
    flags = {"class_id" in (roi.model_extra or {}) for roi in serial.rois()}
    assert flags == {True, False}, "premise: some tiles flagged, some not"

    ome_zarr = _detection_setup(tmp_path / "det_int_jobs.zarr")
    args_list = _detection_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    for args in reversed(args_list):
        assert (
            _detection_iterator(ome_zarr).for_job(**args).detect(_flagged_detector)
            is None
        )
    merged = _detection_iterator(ome_zarr).finalize()

    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)
    for serial_roi, merged_roi in zip(serial.rois(), merged.rois(), strict=True):
        serial_extra = (serial_roi.model_extra or {}).get("class_id")
        merged_extra = (merged_roi.model_extra or {}).get("class_id")
        assert (serial_extra is None) == (merged_extra is None)
        if merged_extra is not None:
            assert merged_extra == 7
            assert not isinstance(merged_extra, float)


def test_detection_slice_detect_banks_and_slice_finalize_refuses(tmp_path: Path):
    ome_zarr = _detection_setup(tmp_path / "det_guard.zarr")
    args_list = _detection_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    restricted = _detection_iterator(ome_zarr).for_job(**args_list[0])
    # The slice's detect banks its pre-NMS boxes and hands nothing back.
    assert restricted.detect(_detector) is None
    image = ome_zarr.get_image()
    partials = image._group_handler.group["_ngio_partials"]
    assert f"job_{args_list[0]['job_index']}" in list(partials.keys())
    # The gather stays global.
    with pytest.raises(NgioValueError, match="gather"):
        restricted.finalize()


def test_detection_partial_refuses_colliding_extras(tmp_path: Path):
    """An extra field shadowing the partial table's own columns fails in the job."""
    from ngio import Roi

    ome_zarr = _detection_setup(tmp_path / "det_reserved.zarr")
    args_list = _detection_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    restricted = _detection_iterator(ome_zarr).for_job(**args_list[0])

    def shadowing(patch):
        return [
            Roi.from_values(
                slices={"x": (0, 1), "y": (0, 1)},
                name=None,
                space="pixel",
                x_start=5.0,
            )
        ]

    with pytest.raises(NgioValueError, match="collide with the partial table"):
        restricted.detect(shadowing)


def _run_feature_job(store: str, job_index: int, n_jobs: int) -> None:
    """One array task of a distributed feature run."""
    ome_zarr = open_ome_zarr_container(store)
    _feature_iterator(ome_zarr).for_job(job_index=job_index, n_jobs=n_jobs).measure(
        _measure
    )


def test_feature_jobs_across_processes_match_serial(tmp_path: Path):
    import pandas as pd

    serial_oz = _feature_setup(tmp_path / "feat_proc_serial.zarr")
    serial = _feature_iterator(serial_oz).measure(_measure)
    assert serial is not None

    store = tmp_path / "feat_procs.zarr"
    ome_zarr = _feature_setup(store)
    args_list = _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)

    with ProcessPoolExecutor(
        max_workers=len(args_list), mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_run_feature_job, str(store), **args) for args in args_list
        ]
        for future in futures:
            future.result()

    merged = _feature_iterator(ome_zarr).finalize()
    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)


# --- topic verbs (process/segment/measure/detect + the one finalize) ---------


def test_segment_matches_map(tmp_path: Path):
    """The topic verb is `map` end to end: same levels, auto-finalized."""
    reference = _serial_reference(tmp_path)

    ome_zarr = _build_ome_zarr(tmp_path / "segment.zarr")
    ome_zarr.derive_label("out")
    _fresh_iterator(ome_zarr).segment(_threshold)

    for path, level in _label_levels(ome_zarr).items():
        np.testing.assert_array_equal(
            level, reference[path], err_msg=f"level {path} differs"
        )


def test_process_matches_map(tmp_path: Path):
    from ngio.iterators import ImageProcessingIterator

    def halve(patch):
        return patch // 2

    def _proc_iterator(ome_zarr, out):
        return ImageProcessingIterator(
            ome_zarr.get_image(), out.get_image(), consolidation_mode="dask"
        ).by_chunks()

    source = _build_ome_zarr(tmp_path / "proc_src.zarr")
    map_out = source.derive_image(store=tmp_path / "proc_map.zarr")
    topic_out = source.derive_image(store=tmp_path / "proc_topic.zarr")

    _proc_iterator(source, map_out).map(halve)
    _proc_iterator(source, topic_out).process(halve)

    np.testing.assert_array_equal(
        topic_out.get_image().zarr_array[...],
        map_out.get_image().zarr_array[...],
    )


def test_slice_segment_does_not_finalize(tmp_path: Path):
    """The topic verb inherits `map`'s slice behavior: write, no gather."""
    ome_zarr = _build_ome_zarr(tmp_path / "seg_slice.zarr")
    ome_zarr.derive_label("out")
    for job_index in range(2):
        _fresh_iterator(ome_zarr).for_job(job_index, n_jobs=2).segment(_threshold)

    levels = _label_levels(ome_zarr)
    paths = sorted(levels, key=lambda p: int(p))
    assert levels[paths[0]].any(), "level 0 must hold the written labels"
    for path in paths[1:]:
        assert not levels[path].any(), (
            f"level {path} must still be stale before finalize()"
        )

    _fresh_iterator(ome_zarr).finalize()
    assert _label_levels(ome_zarr)[paths[1]].any()


def test_detection_finalize_refuses_without_prepare(tmp_path: Path):
    ome_zarr = _detection_setup(tmp_path / "det_noprep.zarr")
    with pytest.raises(NgioValueError, match="No partials"):
        _detection_iterator(ome_zarr).finalize()


def _haloed_feature_iterator(ome_zarr):
    return _feature_iterator(ome_zarr).with_halo(y=8, x=8)


@pytest.mark.parametrize("custom_join", [False, True])
def test_feature_merge_matches_serial_with_halo(tmp_path: Path, custom_join: bool):
    """The read-only halo distributes: grown reads, same one global join."""
    import pandas as pd

    join = _global_norm_join if custom_join else None
    serial_oz = _feature_setup(tmp_path / "feat_halo_serial.zarr")
    serial_it = _haloed_feature_iterator(serial_oz)
    if join is not None:
        serial_it = serial_it.with_join(join)
    serial = serial_it.measure(_measure)
    assert serial is not None
    # The halo makes border objects appear in several grown regions.
    assert not serial.dataframe.index.is_unique

    ome_zarr = _feature_setup(tmp_path / "feat_halo_jobs.zarr")
    args_list = _haloed_feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in reversed(args_list):
        assert (
            _haloed_feature_iterator(ome_zarr).for_job(**args).measure(_measure) is None
        )
    gather_it = _haloed_feature_iterator(ome_zarr)
    if join is not None:
        gather_it = gather_it.with_join(join)
    merged = gather_it.finalize()

    pd.testing.assert_frame_equal(serial.dataframe, merged.dataframe)


def test_feature_slice_measure_refuses_reserved_columns(tmp_path: Path):
    """The provenance guard fires in the banking path too."""
    ome_zarr = _feature_setup(tmp_path / "feat_reserved.zarr")
    args_list = _feature_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    restricted = _feature_iterator(ome_zarr).for_job(**args_list[0])

    def shadowing(image, label_patch, roi):
        return {"label": [1], "roi_name": ["boom"]}

    with pytest.raises(NgioValueError, match="reserved column"):
        restricted.measure(shadowing)


class _StrictSeamMatcher:
    """A picklable non-default matcher (module-level would pickle too)."""

    def __call__(self, patch_a, patch_b):
        from ngio.iterators import IouSeamMatcher

        return IouSeamMatcher(0.9)(patch_a, patch_b)


def test_distributed_custom_seam_matcher_matches_serial(tmp_path: Path):
    """A custom matcher declared identically on every phase is bit-identical."""
    from ngio.iterators import StitchConfig

    def _matched_iterator(ome_zarr):
        return (
            SegmentationIterator(
                ome_zarr.get_image(),
                ome_zarr.get_label("seg"),
                axes_order="yx",
                consolidation_mode="dask",
            )
            .with_stitch(StitchConfig(seam_matcher=_StrictSeamMatcher()))
            .by_grid(size_y=32, size_x=32)
            .with_halo(y=4, x=4)
        )

    serial_oz = _stitched_setup(tmp_path / "matcher_serial.zarr")
    serial_oz.derive_label("seg")
    _matched_iterator(serial_oz).segment(_threshold)

    ome_zarr = _stitched_setup(tmp_path / "matcher_jobs.zarr")
    ome_zarr.derive_label("seg")
    args_list = _matched_iterator(ome_zarr).prepare_jobs(n_jobs=2)
    for args in args_list:
        _matched_iterator(ome_zarr).for_job(**args).segment(_threshold)
    _matched_iterator(ome_zarr).finalize()

    np.testing.assert_array_equal(
        ome_zarr.get_label("seg").get_as_numpy(),
        serial_oz.get_label("seg").get_as_numpy(),
    )
