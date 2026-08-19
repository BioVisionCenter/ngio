"""`consolidate(regions=...)` speaks `Roi` — the same vocabulary as `set_roi`.

The natural loop is `set_roi(roi, patch)` then `consolidate(regions=[roi])`:
the Roi is resolved through the same pipe machinery the setter used, so the
regions name exactly the written pixels. Raw on-disk tuples (what setter
pipes produce) pass through unchanged, and the two forms mix freely.
"""

import numpy as np

from ngio import Roi, create_ome_zarr_from_array


def _build(store):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    container = create_ome_zarr_from_array(
        store=store,
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=3,
        chunks=(1, 16, 16),
        consolidation_mode="dask",
    )
    image = container.get_image()
    image.consolidate(mode="dask")
    return image


def _levels(image):
    handler = image._group_handler
    return {
        path: handler.get_array(path)[...]
        for path in image.meta_handler.get_meta().paths
    }


def _pixel_roi(image, name, y, x):
    roi = Roi.from_values(name=name, slices={"y": y, "x": x}, space="pixel")
    return roi.to_world(pixel_size=image.pixel_size)


def test_roi_regions_match_full_rebuild(tmp_path):
    image = _build(tmp_path / "roi.zarr")
    roi = _pixel_roi(image, "patch", slice(0, 20), slice(10, 40))
    patch = np.full((1, 20, 30), 200, dtype="uint8")

    image.set_roi(roi, patch)
    image.consolidate(mode="dask", regions=[roi])

    after_partial = _levels(image)
    image.consolidate(mode="dask")  # full rebuild over the same level 0
    after_full = _levels(image)

    for path, full_level in after_full.items():
        np.testing.assert_array_equal(
            after_partial[path], full_level, err_msg=f"level {path} differs"
        )


def test_track_writes_matches_full_rebuild(tmp_path):
    import dask.array as da

    image = _build(tmp_path / "track.zarr")
    roi = _pixel_roi(image, "patch", slice(0, 20), slice(10, 40))

    with image.track_writes() as regions:
        image.set_roi(roi, np.full((1, 20, 30), 200, dtype="uint8"))
        image.set_array(
            da.full((1, 16, 16), 9, dtype="uint8"), y=slice(32, 48), x=slice(32, 48)
        )
    assert len(regions) == 2

    image.consolidate(mode="dask", regions=regions)
    after_partial = _levels(image)
    image.consolidate(mode="dask")
    after_full = _levels(image)

    for path, full_level in after_full.items():
        np.testing.assert_array_equal(
            after_partial[path], full_level, err_msg=f"level {path} differs"
        )


def test_track_writes_is_partial(tmp_path):
    """An untouched level-1 chunk is not rewritten by the tracked consolidate."""
    image = _build(tmp_path / "track_partial.zarr")
    handler = image._group_handler
    level_1 = handler.get_array(image.meta_handler.get_meta().paths[1])

    poison_region = (slice(None), slice(16, 32), slice(16, 32))
    poison = np.full((1, 16, 16), 123, dtype="uint8")
    level_1[poison_region] = poison

    with image.track_writes() as regions:
        image.set_array(
            np.full((1, 16, 16), 7, dtype="uint8"), y=slice(0, 16), x=slice(0, 16)
        )
    image.consolidate(mode="dask", regions=regions)

    np.testing.assert_array_equal(level_1[poison_region], poison)


def test_untracked_writes_do_not_record(tmp_path):
    image = _build(tmp_path / "untracked.zarr")
    patch = np.full((1, 16, 16), 5, dtype="uint8")

    image.set_array(patch, y=slice(0, 16), x=slice(0, 16))  # before: not seen
    with image.track_writes() as regions:
        image.set_array(patch, y=slice(16, 32), x=slice(16, 32))
    image.set_array(patch, y=slice(32, 48), x=slice(32, 48))  # after: not seen

    assert len(regions) == 1


def test_empty_tracked_regions_consolidate_is_noop(tmp_path):
    """No writes tracked -> nothing to rebuild, not a full rebuild."""
    image = _build(tmp_path / "noop.zarr")
    handler = image._group_handler
    level_1 = handler.get_array(image.meta_handler.get_meta().paths[1])
    poison = np.full(level_1.shape, 123, dtype="uint8")
    level_1[...] = poison

    with image.track_writes() as regions:
        pass
    image.consolidate(mode="dask", regions=regions)

    # Any consolidation at all would repair the poison.
    np.testing.assert_array_equal(level_1[...], poison)


def test_nested_blocks_each_record(tmp_path):
    image = _build(tmp_path / "nested.zarr")
    patch = np.full((1, 16, 16), 5, dtype="uint8")

    with image.track_writes() as outer:
        image.set_array(patch, y=slice(0, 16), x=slice(0, 16))
        with image.track_writes() as inner:
            image.set_array(patch, y=slice(16, 32), x=slice(16, 32))

    assert len(inner) == 1
    assert len(outer) == 2


def test_masked_write_is_recorded(tmp_path):
    rng = np.random.default_rng(0)
    ome_zarr = create_ome_zarr_from_array(
        store=tmp_path / "masked.zarr",
        array=rng.integers(0, 255, size=(64, 64)).astype("uint8"),
        pixelsize=1.0,
        axes_names="yx",
        levels=2,
        chunks=(16, 16),
        consolidation_mode="dask",
    )
    label = ome_zarr.derive_label("mask")
    label_image = np.zeros((64, 64), dtype="uint32")
    label_image[0:16, 0:16] = 1
    label_image[32:48, 32:48] = 2
    label.set_array(label_image)
    label.consolidate(mode="dask")
    ome_zarr.add_table("mask_table", label.build_masking_roi_table())

    masked = ome_zarr.get_masked_image(masking_label_name="mask")
    patch = masked.get_roi_masked_as_numpy(label=2)
    with masked.track_writes() as regions:
        masked.set_roi_masked(label=2, patch=patch + 1)

    assert len(regions) == 1

    masked.consolidate(mode="dask", regions=regions)
    after_partial = {
        path: masked._group_handler.get_array(path)[...]
        for path in masked.meta_handler.get_meta().paths
    }
    masked.consolidate(mode="dask")
    for path, partial_level in after_partial.items():
        np.testing.assert_array_equal(
            masked._group_handler.get_array(path)[...],
            partial_level,
            err_msg=f"level {path} differs",
        )


def test_rois_and_tuples_mix(tmp_path):
    image = _build(tmp_path / "mix.zarr")
    roi = _pixel_roi(image, "patch", slice(0, 16), slice(0, 16))
    region = (slice(None), slice(32, 48), slice(32, 48))

    image.set_roi(roi, np.full((1, 16, 16), 7, dtype="uint8"))
    image.set_array(
        np.full((1, 16, 16), 9, dtype="uint8"), y=slice(32, 48), x=slice(32, 48)
    )
    image.consolidate(mode="dask", regions=[roi, region])

    after_partial = _levels(image)
    image.consolidate(mode="dask")
    after_full = _levels(image)

    for path, full_level in after_full.items():
        np.testing.assert_array_equal(
            after_partial[path], full_level, err_msg=f"level {path} differs"
        )
