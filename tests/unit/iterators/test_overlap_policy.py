"""The overlap contract: segmentation refuses undeclared overlapping writes.

`on_overlap` (writers) and `with_stitch` (segmentation) are the two declared
resolutions; without one, overlapping segmentation write footprints refuse at
every writing verb. Image processing keeps the permissive wave-order default,
and masked writes are exempt by construction.
"""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import (
    ImageProcessingIterator,
    SegmentationIterator,
    ThreadedMapper,
)
from ngio.utils import NgioValueError


def _build_ome_zarr(levels=1):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(64, 64)).astype("uint8")
    return create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="yx",
        levels=levels,
        chunks=(16, 16),
        consolidation_mode="dask",
    )


def _threshold(patch):
    return (patch > 128).astype("uint32")


def _seg_iterator(ome_zarr, name="out"):
    return SegmentationIterator(
        ome_zarr.get_image(),
        ome_zarr.get_label(name),
        axes_order="yx",
        consolidation_mode="dask",
    )


def _overlapping(iterator):
    """A shifted-tail grid: the last tile genuinely overlaps its neighbour."""
    return iterator.by_grid(size_y=24, size_x=24, tail="shift")


def test_undeclared_overlapping_segmentation_refuses_everywhere():
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    undeclared = _overlapping(_seg_iterator(ome_zarr))

    with pytest.raises(NgioValueError, match="with_stitch"):
        undeclared.map(_threshold)
    with pytest.raises(NgioValueError, match="on_overlap"):
        undeclared.segment(_threshold)
    with pytest.raises(NgioValueError, match="declared resolution"):
        undeclared.iter(data_mode="numpy")
    with pytest.raises(NgioValueError, match="declared resolution"):
        undeclared.iter(data_mode="numpy", batch_size=2)
    with pytest.raises(NgioValueError, match="declared resolution"):
        undeclared.prepare_jobs(n_jobs=2)


def test_disjoint_segmentation_needs_no_declaration():
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    _seg_iterator(ome_zarr).by_grid(size_y=32, size_x=32).segment(_threshold)
    assert ome_zarr.get_label("out").get_as_numpy().any()


def test_halo_is_not_an_overlap_trigger():
    """Halo-grown reads write cropped cores: disjoint, no declaration needed."""
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    _seg_iterator(ome_zarr).by_grid(size_y=32, size_x=32).with_halo(
        y=4, x=4
    ).segment(_threshold)
    assert ome_zarr.get_label("out").get_as_numpy().any()


def test_on_overlap_last_is_bit_identical_to_the_old_default():
    """`"last"` declares wave order; the pixels match an undeclared 1-tile run."""
    reference = _build_ome_zarr()
    reference.derive_label("out")
    _seg_iterator(reference).segment(_threshold)  # single region: no overlap

    declared = _build_ome_zarr()
    declared.derive_label("out")
    _overlapping(_seg_iterator(declared)).on_overlap("last").segment(_threshold)

    np.testing.assert_array_equal(
        declared.get_label("out").get_as_numpy(),
        reference.get_label("out").get_as_numpy(),
    )


def test_on_overlap_merge_is_order_independent_under_threads():
    ome_zarr_a = _build_ome_zarr()
    ome_zarr_a.derive_label("out")
    ome_zarr_b = _build_ome_zarr()
    ome_zarr_b.derive_label("out")

    def _fill_mean(patch):
        return np.full_like(patch, int(patch.mean()) or 1).astype("uint32")

    _overlapping(_seg_iterator(ome_zarr_a)).on_overlap("max").segment(_fill_mean)
    _overlapping(_seg_iterator(ome_zarr_b)).on_overlap("max").segment(
        _fill_mean, mapper=ThreadedMapper(4)
    )
    np.testing.assert_array_equal(
        ome_zarr_a.get_label("out").get_as_numpy(),
        ome_zarr_b.get_label("out").get_as_numpy(),
    )


def test_invalid_overlap_rule_refuses_at_declaration():
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    with pytest.raises(NgioValueError):
        _seg_iterator(ome_zarr).on_overlap("not-a-rule")


def test_on_overlap_refused_after_for_job():
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    restricted = _seg_iterator(ome_zarr).by_grid(size_y=32, size_x=32).for_job(
        0, n_jobs=2
    )
    with pytest.raises(NgioValueError, match="for_job"):
        restricted.on_overlap("last")
    with pytest.raises(NgioValueError, match="for_job"):
        restricted.with_stitch()


def test_stitch_and_on_overlap_are_mutually_exclusive():
    ome_zarr = _build_ome_zarr()
    ome_zarr.derive_label("out")
    iterator = _seg_iterator(ome_zarr)
    with pytest.raises(NgioValueError, match="cannot be combined"):
        iterator.with_stitch().on_overlap("last")
    with pytest.raises(NgioValueError, match="cannot be combined"):
        iterator.on_overlap("last").with_stitch()


def test_masked_on_overlap_refused_and_overlap_exempt():
    from ngio.iterators import MaskedSegmentationIterator

    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, size=(64, 64)).astype("uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        consolidation_mode="dask",
    )
    mask = ome_zarr.derive_label("mask")
    mask_data = np.zeros((64, 64), dtype="uint32")
    mask_data[4:40, 4:40] = 1  # overlapping bounding boxes with...
    mask_data[30:60, 30:60] = 2  # ...this neighbour
    mask.set_array(mask_data)
    mask.consolidate(mode="dask")
    ome_zarr.add_table("masking_ROI_table", mask.build_masking_roi_table())
    ome_zarr.derive_label("nuclei")

    iterator = MaskedSegmentationIterator(
        ome_zarr.get_masked_image(masking_label_name="mask"),
        ome_zarr.get_label("nuclei"),
        axes_order="yx",
        consolidation_mode="dask",
    )
    with pytest.raises(NgioValueError, match="mask-protected"):
        iterator.on_overlap("last")
    # Overlapping per-object boxes run undeclared: writes never contest.
    iterator.segment(_threshold)


def test_image_processing_overlap_stays_permissive():
    ome_zarr = _build_ome_zarr()
    out = ome_zarr.derive_image(store=MemoryStore())

    def halve(patch):
        return patch // 2

    base = ImageProcessingIterator(
        ome_zarr.get_image(), out.get_image(), consolidation_mode="dask"
    )
    # Undeclared overlap: still runs (wave-order default).
    _overlapping(base).process(halve)
    # And a declared merge is accepted too.
    _overlapping(base).on_overlap("sum").process(halve)
