"""Writing iterators consolidate only what they wrote, and stitch opts out.

The parity trick: after a region-restricted `map` (whose `finalize` used
`regions=`), forcing a *full* consolidate must change nothing — the region
path is only ever a faster spelling of the full rebuild.
"""

import numpy as np

from ngio import Roi, create_ome_zarr_from_array
from ngio.iterators import FeatureExtractorIterator, SegmentationIterator


def _build_ome_zarr(store, levels=3, chunks=(1, 16, 16)):
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
    return (patch > 128).astype("uint8")


def _quadrant_roi(image):
    roi = Roi.from_values(
        name="quadrant", slices={"y": slice(0, 32), "x": slice(0, 32)}, space="pixel"
    )
    return roi.to_world(pixel_size=image.pixel_size)


def _level_arrays(label):
    handler = label._group_handler
    return {
        path: handler.get_array(path)[...]
        for path in label.meta_handler.get_meta().paths
    }


def test_restricted_map_consolidates_partially_and_exactly(tmp_path):
    ome_zarr = _build_ome_zarr(tmp_path / "partial.zarr")
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")

    iterator = SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx", consolidation_mode="dask"
    ).product([_quadrant_roi(image)])
    iterator.map(_threshold)

    after_partial = _level_arrays(label)
    label.consolidate(mode="dask")  # full rebuild over the same level 0
    after_full = _level_arrays(label)

    for path, full_level in after_full.items():
        np.testing.assert_array_equal(
            after_partial[path], full_level, err_msg=f"level {path} differs"
        )


def test_touched_write_regions_are_the_setters_tuples(tmp_path):
    ome_zarr = _build_ome_zarr(tmp_path / "regions.zarr")
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")

    iterator = SegmentationIterator(
        image, label, channel_selection=0, axes_order="yx", consolidation_mode="dask"
    ).product([_quadrant_roi(image)])

    regions = iterator._touched_write_regions()
    assert regions is not None
    assert len(regions) == len(iterator.rois)
    expected = [
        iterator.build_numpy_setter(roi).slicing_ops.normalized_slicing_tuple
        for roi in iterator.rois
    ]
    assert list(regions) == expected


def test_readonly_iterator_has_no_write_regions(tmp_path):
    ome_zarr = _build_ome_zarr(tmp_path / "readonly.zarr")
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")

    iterator = FeatureExtractorIterator(image, label, axes_order="yx")
    assert iterator._touched_write_regions() is None


def test_stitching_finalize_forces_a_full_rebuild(tmp_path, monkeypatch):
    """With `with_stitch()` the resolve rewrites level 0 globally: no regions."""
    ome_zarr = _build_ome_zarr(tmp_path / "stitch.zarr", chunks=(1, 32, 32))
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("out")

    iterator = (
        SegmentationIterator(
            image,
            label,
            channel_selection=0,
            axes_order="yx",
            consolidation_mode="dask",
        )
        .with_stitch()
        .by_grid(size_y=32, size_x=32)
        .with_halo(y=4, x=4)
    )

    calls = []
    original = type(label).consolidate

    def recording(self, mode=None, regions=None):
        calls.append(regions)
        return original(self, mode=mode, regions=regions)

    monkeypatch.setattr(type(label), "consolidate", recording)
    iterator.map(lambda patch: (patch > 128).astype("uint32"))

    assert calls == [None]
