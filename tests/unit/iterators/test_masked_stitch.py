"""Stitching within a mask: sub-tiles of one object merge, masks never mix."""

import warnings

import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array
from ngio.iterators import (
    MaskedSegmentationIterator,
    ProcessMapper,
    SegmentationIterator,
    ThreadedMapper,
)
from ngio.iterators._stitch import StitchConfig
from ngio.iterators._stitch_geometry import TileExtent, touching_unstitched_axes
from ngio.transforms import UniqueLabelsTransform
from ngio.utils import NgioUserWarning, NgioValueError


def _label_over_128(patch: np.ndarray) -> np.ndarray:
    """A tiny 2D labeller of `patch > 128`, so tests do not depend on scipy."""
    out = np.zeros(patch.shape, dtype="uint32")
    next_id = 0
    for start in zip(*np.nonzero(patch > 128), strict=True):
        if out[start]:
            continue
        next_id += 1
        stack = [start]
        out[start] = next_id
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < patch.shape[0]
                    and 0 <= nx < patch.shape[1]
                    and patch[ny, nx] > 128
                    and not out[ny, nx]
                ):
                    out[ny, nx] = next_id
                    stack.append((ny, nx))
    return out


def _label_everything(patch: np.ndarray) -> np.ndarray:
    """A pathological detector: labels every pixel, the fill region included."""
    return np.ones(patch.shape, dtype="uint32")


def _masked_setup(store, *, touching=False, levels=1):
    """A 96x96 image with two organoids and three nuclei.

    Default layout: organoid 1 spans several 32px tiles and holds nucleus A
    (crossing the y=40 sub-tile boundary of its bbox grid) and nucleus B;
    organoid 2 is separate and holds nucleus C. With `touching=True` the two
    organoids share the x=48 face and their nuclei abut across it.
    """
    mask = np.zeros((96, 96), dtype="uint32")
    data = np.zeros((96, 96), dtype="uint8")
    if touching:
        mask[8:56, 8:48] = 1
        mask[8:56, 48:88] = 2
        data[20:32, 40:48] = 200  # nucleus in organoid 1, ends at the shared face
        data[20:32, 48:60] = 200  # nucleus in organoid 2, starts at the shared face
    else:
        mask[8:88, 8:56] = 1
        mask[8:40, 64:88] = 2
        data[30:50, 16:28] = 200  # nucleus A: crosses y=40 of organoid 1's grid
        data[70:80, 20:30] = 200  # nucleus B: wholly inside one sub-tile
        data[16:28, 68:80] = 200  # nucleus C: inside organoid 2

    ome_zarr = create_ome_zarr_from_array(
        store=store,
        array=data,
        pixelsize=1.0,
        axes_names="yx",
        levels=levels,
        chunks=(32, 32),
        consolidation_mode="dask",
    )
    organoids = ome_zarr.derive_label("organoids")
    organoids.set_array(mask)
    organoids.consolidate(mode="dask")
    ome_zarr.add_table("masking_ROI_table", organoids.build_masking_roi_table())
    ome_zarr.derive_label("nuclei")
    return ome_zarr, mask


def _stitched_masked_iterator(ome_zarr, *, stitch=True, halo=6):
    iterator = MaskedSegmentationIterator(
        ome_zarr.get_masked_image(masking_label_name="organoids"),
        ome_zarr.get_label("nuclei"),
        axes_order="yx",
        consolidation_mode="dask",
        stitch=stitch,
    ).by_grid(size_y=32, size_x=32)
    if halo:
        iterator = iterator.with_halo(y=halo, x=halo)
    return iterator


def _ids(written: np.ndarray) -> set[int]:
    return {int(value) for value in np.unique(written) if value}


def test_masked_split_object_becomes_one_id():
    ome_zarr, mask = _masked_setup(MemoryStore())
    _stitched_masked_iterator(ome_zarr).map(_label_over_128)

    written = ome_zarr.get_label("nuclei").get_as_numpy()
    # Three nuclei, one of them split by a sub-tile boundary: three dense ids.
    assert _ids(written) == {1, 2, 3}
    # Nucleus A is one id despite the split.
    assert len(_ids(written[30:50, 16:28])) == 1
    # Nothing leaked outside the masks.
    assert (written[mask == 0] == 0).all()


def test_masked_stitch_parallel_matches_serial():
    serial_zarr, _ = _masked_setup(MemoryStore())
    _stitched_masked_iterator(serial_zarr).map(_label_over_128)

    threaded_zarr, _ = _masked_setup(MemoryStore())
    _stitched_masked_iterator(threaded_zarr).map(
        _label_over_128, mapper=ThreadedMapper(4)
    )

    np.testing.assert_array_equal(
        serial_zarr.get_label("nuclei").get_as_numpy(),
        threaded_zarr.get_label("nuclei").get_as_numpy(),
    )


def test_masked_stitch_never_merges_across_masks():
    ome_zarr, _ = _masked_setup(MemoryStore(), touching=True)
    _stitched_masked_iterator(ome_zarr).map(_label_over_128)

    written = ome_zarr.get_label("nuclei").get_as_numpy()
    left = _ids(written[20:32, 40:48])
    right = _ids(written[20:32, 48:60])
    assert len(left) == 1 and len(right) == 1
    assert left != right, "objects of different masks were merged"


def test_masked_garbage_extension_does_not_bridge_masks():
    """A func that labels the fill region must not union across masks."""
    ome_zarr, mask = _masked_setup(MemoryStore(), touching=True)
    _stitched_masked_iterator(ome_zarr).map(_label_everything)

    written = ome_zarr.get_label("nuclei").get_as_numpy()
    # One merged id per organoid, and the masks keep their pixels apart.
    assert _ids(written) == {1, 2}
    assert len(_ids(written[mask == 1])) == 1
    assert len(_ids(written[mask == 2])) == 1
    assert _ids(written[mask == 1]) != _ids(written[mask == 2])
    assert (written[mask == 0] == 0).all()


def test_masked_stitch_with_coarser_masking_label():
    """The masking label at a coarser level still masks the banks correctly."""
    from ngio.images._masked_image import MaskedImage

    ome_zarr, mask = _masked_setup(MemoryStore(), levels=2)
    image = ome_zarr.get_image()
    coarse = ome_zarr.get_label("organoids", path="1")
    masked = MaskedImage(
        group_handler=image._group_handler,
        path=image.path,
        meta_handler=image.meta_handler,
        label=coarse,
        masking_roi_table=ome_zarr.get_masking_roi_table("masking_ROI_table"),
    )
    iterator = (
        MaskedSegmentationIterator(
            masked,
            ome_zarr.get_label("nuclei"),
            axes_order="yx",
            consolidation_mode="dask",
            stitch=True,
        )
        .by_grid(size_y=32, size_x=32)
        .with_halo(y=6, x=6)
    )
    iterator.map(_label_over_128)

    written = ome_zarr.get_label("nuclei").get_as_numpy()
    assert _ids(written) == {1, 2, 3}
    assert len(_ids(written[30:50, 16:28])) == 1


def test_masked_stitched_jobs_match_serial(tmp_path):
    serial_zarr, _ = _masked_setup(tmp_path / "serial.zarr")
    _stitched_masked_iterator(serial_zarr).map(_label_over_128)
    reference = serial_zarr.get_label("nuclei").get_as_numpy()

    ome_zarr, _ = _masked_setup(tmp_path / "jobs.zarr")
    args_list = _stitched_masked_iterator(ome_zarr).prepare_jobs(n_jobs=3)
    for args in reversed(args_list):
        _stitched_masked_iterator(ome_zarr).for_job(**args).map(_label_over_128)
    _stitched_masked_iterator(ome_zarr).finalize()

    np.testing.assert_array_equal(
        ome_zarr.get_label("nuclei").get_as_numpy(), reference
    )


def test_masked_stitch_bank_transform_pickles(tmp_path):
    """ProcessMapper ships the setters (and the bank mask) to the workers."""
    serial_zarr, _ = _masked_setup(tmp_path / "serial.zarr")
    _stitched_masked_iterator(serial_zarr).map(_label_over_128)

    process_zarr, _ = _masked_setup(tmp_path / "process.zarr")
    _stitched_masked_iterator(process_zarr).map(
        _label_over_128, mapper=ProcessMapper(max_workers=2)
    )

    np.testing.assert_array_equal(
        serial_zarr.get_label("nuclei").get_as_numpy(),
        process_zarr.get_label("nuclei").get_as_numpy(),
    )


def test_masked_stitch_refuses_unique_labels_transform():
    ome_zarr, _ = _masked_setup(MemoryStore())
    masked = ome_zarr.get_masked_image(masking_label_name="organoids")
    nuclei = ome_zarr.get_label("nuclei")
    with pytest.raises(NgioValueError, match="UniqueLabelsTransform"):
        MaskedSegmentationIterator(
            masked,
            nuclei,
            axes_order="yx",
            stitch=True,
            output_transforms=[UniqueLabelsTransform(block_size=1000)],
        )
    # The base class shares the double-offset failure mode.
    with pytest.raises(NgioValueError, match="UniqueLabelsTransform"):
        SegmentationIterator(
            ome_zarr.get_image(),
            nuclei,
            axes_order="yx",
            stitch=True,
            output_transforms=[UniqueLabelsTransform(block_size=1000, block_index=0)],
        )


def test_masked_stitch_refuses_the_dask_path():
    ome_zarr, _ = _masked_setup(MemoryStore())
    iterator = _stitched_masked_iterator(ome_zarr)
    with pytest.raises(NgioValueError, match="numpy path"):
        iterator.build_dask_setter(iterator.rois[0])


def test_masked_get_init_kwargs_round_trip():
    """`stitch=True` must survive the builder chain's reconstruction."""
    ome_zarr, _ = _masked_setup(MemoryStore())
    iterator = _stitched_masked_iterator(ome_zarr)
    assert iterator._stitch is not None
    assert isinstance(iterator.get_init_kwargs()["stitch"], StitchConfig)


def test_masked_single_tile_objects_warn_noop_not_error():
    """Untiled masked objects: a soft no-op warning, never the halo error."""
    ome_zarr, _ = _masked_setup(MemoryStore())
    iterator = MaskedSegmentationIterator(
        ome_zarr.get_masked_image(masking_label_name="organoids"),
        ome_zarr.get_label("nuclei"),
        axes_order="yx",
        consolidation_mode="dask",
        stitch=True,
    ).with_halo(y=6, x=6)

    with pytest.warns(NgioUserWarning, match="per-object no-op"):
        iterator.map(_label_over_128)
    written = ome_zarr.get_label("nuclei").get_as_numpy()
    assert _ids(written) == {1, 2, 3}

    # With tiling (some object split into several tiles) there is real work:
    # no warning at all.
    tiled_zarr, _ = _masked_setup(MemoryStore())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stitched_masked_iterator(tiled_zarr).map(_label_over_128)
    assert not [w for w in caught if issubclass(w.category, NgioUserWarning)]


def test_touching_unstitched_axes_scoped_to_same_label():
    extents = [
        TileExtent(index=0, core=((0, 1), (0, 64)), grown=((0, 1), (0, 64)), label=1),
        TileExtent(index=1, core=((1, 2), (0, 64)), grown=((1, 2), (0, 64)), label=2),
    ]
    # Different masks touching: not a seam anyone wants stitched.
    assert touching_unstitched_axes(extents, [0], same_label_only=True) == []
    # Same mask touching along an unhaloed axis: still worth the warning.
    same = [
        TileExtent(index=0, core=((0, 1), (0, 64)), grown=((0, 1), (0, 64)), label=1),
        TileExtent(index=1, core=((1, 2), (0, 64)), grown=((1, 2), (0, 64)), label=1),
    ]
    assert touching_unstitched_axes(same, [0], same_label_only=True) == [0]
    # Unlabelled extents keep the unrestricted behaviour.
    unlabelled = [
        TileExtent(index=0, core=((0, 1), (0, 64)), grown=((0, 1), (0, 64))),
        TileExtent(index=1, core=((1, 2), (0, 64)), grown=((1, 2), (0, 64))),
    ]
    assert touching_unstitched_axes(unlabelled, [0], same_label_only=True) == [0]
    assert touching_unstitched_axes(extents, [0]) == [0]
