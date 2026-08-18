"""Tests for `UniqueLabelsTransform`: disjoint id blocks, derived not counted."""

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import Roi, create_ome_zarr_from_array
from ngio.transforms import MaskMerge, UniqueLabelsTransform
from ngio.utils import NgioValueError


def _label(dtype: str = "uint32", shape: tuple[int, int] = (32, 32)):
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(),
        array=np.zeros(shape, dtype="uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=1,
        consolidation_mode="dask",
    )
    label = ome_zarr.derive_label("seg", dtype=dtype)
    return ome_zarr, label


def _roi(start: int, size: int = 16, label: int | None = None) -> Roi:
    return Roi.from_values(
        slices={"y": (start, size), "x": (0, size)}, name=f"r{start}", label=label
    )


def test_labels_are_shifted_into_their_block():
    _, label = _label()
    patch = np.array([[0, 1], [2, 3]], dtype="uint32")

    label.set_roi(
        roi=_roi(0, size=2),
        patch=patch,
        transforms=[UniqueLabelsTransform(1000, block_index=4)],
    )

    np.testing.assert_array_equal(
        label.get_as_numpy()[:2, :2], [[0, 4001], [4002, 4003]]
    )


def test_blocks_from_different_regions_are_disjoint():
    _, label = _label()
    patch = np.arange(1, 5, dtype="uint32").reshape(2, 2)

    for index, start in enumerate((0, 4, 8)):
        label.set_roi(
            roi=_roi(start, size=2),
            patch=patch,
            transforms=[UniqueLabelsTransform(100, block_index=index)],
        )

    written = label.get_as_numpy()
    ids = written[written != 0]
    assert len(set(ids.tolist())) == len(ids), "ids collided across blocks"
    assert set(ids.tolist()) == {1, 2, 3, 4, 101, 102, 103, 104, 201, 202, 203, 204}


def test_background_is_never_offset():
    _, label = _label()
    patch = np.zeros((4, 4), dtype="uint32")
    patch[1, 1] = 5

    label.set_roi(
        roi=_roi(0, size=4),
        patch=patch,
        transforms=[UniqueLabelsTransform(1000, block_index=2)],
    )

    written = label.get_as_numpy()[:4, :4]
    assert written[1, 1] == 2005
    assert (np.delete(written.ravel(), 5) == 0).all()


def test_rerunning_a_region_reproduces_its_ids():
    """Derived, not counted: no shared state, so a retry is a no-op."""
    _, label = _label()
    patch = np.arange(1, 5, dtype="uint32").reshape(2, 2)
    transform = UniqueLabelsTransform(1000, block_index=7)

    label.set_roi(roi=_roi(0, size=2), patch=patch, transforms=[transform])
    first = label.get_as_numpy().copy()
    label.set_roi(roi=_roi(0, size=2), patch=patch, transforms=[transform])

    np.testing.assert_array_equal(label.get_as_numpy(), first)


def test_roi_label_supplies_the_block_index():
    """Masked iterators carry a label per ROI; no explicit index needed."""
    _, label = _label()
    patch = np.array([[1, 2]], dtype="uint32")

    label.set_roi(
        roi=_roi(0, size=2, label=3),
        patch=np.vstack([patch, patch]),
        transforms=[UniqueLabelsTransform(1000)],
    )

    assert set(label.get_as_numpy()[:2, :2].ravel().tolist()) == {3001, 3002}


def test_missing_block_index_raises():
    _, label = _label()
    with pytest.raises(NgioValueError, match="needs a block index"):
        label.set_roi(
            roi=_roi(0, size=2),
            patch=np.ones((2, 2), dtype="uint32"),
            transforms=[UniqueLabelsTransform(1000)],
        )


def test_on_get_returns_local_ids():
    _, label = _label()
    transform = UniqueLabelsTransform(1000, block_index=4)
    patch = np.array([[0, 1], [2, 3]], dtype="uint32")

    label.set_roi(roi=_roi(0, size=2), patch=patch, transforms=[transform])
    np.testing.assert_array_equal(
        label.get_roi_as_numpy(roi=_roi(0, size=2), transforms=[transform]), patch
    )


def test_overflow_is_refused():
    """A uint16 label runs out of id space fast; say so instead of wrapping."""
    _, label = _label(dtype="uint16")

    with pytest.raises(NgioValueError, match="past what uint16 can hold"):
        label.set_roi(
            roi=_roi(0, size=2),
            patch=np.ones((2, 2), dtype="uint16"),
            transforms=[UniqueLabelsTransform(10_000, block_index=7)],
        )


def test_rejects_bad_construction():
    with pytest.raises(NgioValueError, match="block_size must be > 0"):
        UniqueLabelsTransform(0)
    with pytest.raises(NgioValueError, match="block_index must be >= 0"):
        UniqueLabelsTransform(10, block_index=-1)


def test_composes_with_a_reduce_merge():
    """Previously refused, and previously corrupting: the merge now sees raw ids.

    On-disk id 500 belongs to another region. The merge compares it against the
    offset patch (4003) in the array's own space and keeps the larger. The old
    design compared against 500 shifted down by 4000, which underflowed and
    made the on-disk value win every time.
    """
    _, label = _label()
    label.set_array(np.full((32, 32), 500, dtype="uint32"))

    label.set_roi(
        roi=_roi(0, size=2),
        patch=np.full((2, 2), 3, dtype="uint32"),
        transforms=[UniqueLabelsTransform(1000, block_index=4)],
        merge="max",
    )

    assert label.get_as_numpy()[0, 0] == 4003
    assert label.get_as_numpy()[31, 31] == 500


def test_composes_with_a_mask_merge():
    """Also previously refused. Protected ids survive untouched."""
    ome_zarr, label = _label()
    label.set_array(np.full((32, 32), 500, dtype="uint32"))
    mask = ome_zarr.derive_label("m")
    mask_img = np.zeros((32, 32), dtype=mask.zarr_array.dtype)
    mask_img[0:1, 0:1] = 1
    mask.set_array(mask_img)

    label.set_roi(
        roi=_roi(0, size=2, label=1),
        patch=np.full((2, 2), 3, dtype="uint32"),
        transforms=[UniqueLabelsTransform(1000, block_index=4)],
        merge=MaskMerge(label=mask, target_image=label),
    )

    written = label.get_as_numpy()
    assert written[0, 0] == 4003, "inside the mask the offset patch should land"
    assert written[1, 1] == 500, "outside the mask the on-disk id must survive"
