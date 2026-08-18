from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngio import create_ome_zarr_from_array, open_ome_zarr_container
from ngio.iterators import (
    FeatureExtractorIterator,
    ImageProcessingIterator,
    MaskedSegmentationIterator,
    SegmentationIterator,
)
from ngio.utils import NgioDeprecationWarning, NgioValueError

# The iterators run through the version-agnostic container API, so the
# expensive write tests cover the full axes matrix on v05 only, plus a v04
# smoke subset; the full v04 write path is monitored by creation/store tests.
WRITER_ZARR_NAMES = [
    f"v05/test_image_{axes}.zarr"
    for axes in ("yx", "cyx", "zyx", "czyx", "c1yx", "tyx", "tcyx", "tzyx", "tczyx")
] + [
    "v04/test_image_yx.zarr",
    "v04/test_image_tczyx.zarr",
]


@pytest.mark.parametrize("zarr_name", WRITER_ZARR_NAMES)
def test_segmentation_iterator(single_image_copy: Path):
    # Base test only the API, not the actual segmentation logic
    path = single_image_copy
    ome_zarr = open_ome_zarr_container(path)
    image = ome_zarr.get_image()
    label = ome_zarr.get_label("label")
    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    assert iterator.__repr__().startswith("SegmentationIterator")

    iterator = iterator.by_yx()
    assert len(iterator.rois) == image.dimensions.get("t", 1) * image.dimensions.get(
        "z", 1
    )
    iterator = iterator.by_chunks()

    iterator.require_no_regions_overlap()
    if image.is_3d or image.is_time_series:
        # 3D images should overlap in chunks
        with pytest.raises(NgioValueError):
            iterator.require_no_chunks_overlap()
    else:
        # 2D or 2D+channels does not overlap in chunks
        iterator.require_no_chunks_overlap()

    for i, (img_chunk, writer) in enumerate(iterator.iter_as_numpy()):
        label_patch = np.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map(lambda x: np.zeros_like(x, dtype=np.uint8))

    with pytest.warns(NgioDeprecationWarning):
        for i, (img_chunk, writer) in enumerate(iterator.iter_as_dask()):
            label_patch = da.full(
                shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8
            )
            writer(label_patch)

    with pytest.warns(NgioDeprecationWarning):
        iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


@pytest.mark.parametrize("zarr_name", WRITER_ZARR_NAMES)
def test_masked_segmentation_iterator(single_image_copy: Path):
    # Base test only the API, not the actual segmentation logic
    path = single_image_copy
    ome_zarr = open_ome_zarr_container(path)

    masked_label = ome_zarr.derive_label("masking_label")
    masked_label.set_array(np.ones(shape=masked_label.shape, dtype=np.uint8))
    masked_label.consolidate()

    ome_zarr.add_table(
        "masking_label_ROI_table", masked_label.build_masking_roi_table()
    )

    image = ome_zarr.get_masked_image(masking_label_name="masking_label")
    label = ome_zarr.get_label("label")

    iterator = MaskedSegmentationIterator(
        image, label, channel_selection=0, axes_order="yx"
    )

    iterator = iterator.by_yx()
    for i, (img_chunk, writer) in enumerate(iterator.iter_as_numpy()):
        label_patch = np.full(shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8)
        writer(label_patch)

    iterator.map(lambda x: np.zeros_like(x, dtype=np.uint8))

    with pytest.warns(NgioDeprecationWarning):
        for i, (img_chunk, writer) in enumerate(iterator.iter_as_dask()):
            label_patch = da.full(
                shape=img_chunk.shape, fill_value=i + 1, dtype=np.uint8
            )
            writer(label_patch)

    with pytest.warns(NgioDeprecationWarning):
        iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


def test_img_processing_iterator(
    images_all_versions_readonly: dict[str, Path], zarr_name: str
):
    # Base test only the API, not the actual segmentation logic
    path = images_all_versions_readonly[zarr_name]
    ome_zarr = open_ome_zarr_container(path)
    image = ome_zarr.get_image()
    t_ome_zarr = ome_zarr.derive_image(store=MemoryStore())
    t_image = t_ome_zarr.get_image()

    iterator = ImageProcessingIterator(input_image=image, output_image=t_image)

    assert len(iterator.rois) == 1
    roi_table = image.build_image_roi_table()
    iterator = iterator.product(roi_table)
    assert len(iterator.rois) == 1

    iterator = iterator.by_zyx(strict=False)
    assert len(iterator.rois) == image.dimensions.get("t", 1)

    iterator = iterator.grid(size_x=64, size_y=64)
    for img_chunk, writer in iterator.iter_as_numpy():
        label_patch = np.zeros_like(img_chunk, dtype=np.uint8)
        writer(label_patch)

    iterator.map(lambda x: np.zeros_like(x, dtype=np.uint8))

    with pytest.warns(NgioDeprecationWarning):
        for img_chunk, writer in iterator.iter_as_dask():
            label_patch = da.zeros_like(img_chunk, dtype=np.uint8)
            writer(label_patch)

    with pytest.warns(NgioDeprecationWarning):
        iterator.map_as_dask(lambda x: da.zeros_like(x, dtype=np.uint8))


def test_features_iterator(
    images_all_versions_readonly: dict[str, Path], zarr_name: str
):
    # Base test only the API, not the actual segmentation logic
    path = images_all_versions_readonly[zarr_name]
    ome_zarr = open_ome_zarr_container(path)

    image = ome_zarr.get_image()
    label = ome_zarr.get_label(name="label")

    feat_iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label,
        channel_selection=0,
        axes_order="xy",
    )
    feat_iterator = feat_iterator.by_yx()
    for data, seg, _ in feat_iterator.iter_as_numpy():
        assert data.shape == seg.shape


def _build_ome_zarr(chunks=(1, 16, 16), ngff_version="0.4"):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(1, 64, 64)).astype("uint8")
    return create_ome_zarr_from_array(
        store=MemoryStore(),
        array=array,
        pixelsize=1.0,
        axes_names="cyx",
        levels=1,
        chunks=chunks,
        ngff_version=ngff_version,
    )


def test_chunk_gate_measures_write_grid():
    """The regression the write-side gate fix targets: ROIs disjoint on the
    input chunk grid but sharing chunks of a coarser-chunked output must be
    flagged (the getter-sourced gate returned False here)."""
    ome_zarr = _build_ome_zarr(chunks=(1, 16, 16))
    # chunks are given at the reference (cyx) rank; the channel axis is squeezed
    label = ome_zarr.derive_label("coarse", chunks=(1, 32, 32))
    image = ome_zarr.get_image()

    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    iterator = iterator.grid(size_x=16, size_y=16)

    assert iterator.check_if_chunks_overlap()
    with pytest.raises(NgioValueError):
        iterator.require_no_chunks_overlap()

    # Same tiling against a matching output grid is safe.
    matching = ome_zarr.derive_label("matching")
    iterator = SegmentationIterator(
        image, matching, channel_selection=0, axes_order="yx"
    )
    iterator = iterator.grid(size_x=16, size_y=16)
    assert not iterator.check_if_chunks_overlap()
    iterator.require_no_chunks_overlap()


def test_chunk_gate_shard_granularity():
    """ROIs disjoint at inner-chunk granularity but landing in one shard must be
    flagged: writes are atomic per shard object."""
    ome_zarr = _build_ome_zarr(chunks=(1, 16, 16), ngff_version="0.5")
    image = ome_zarr.get_image()

    sharded = ome_zarr.derive_label("sharded", chunks=(1, 16, 16), shards=(1, 64, 64))
    iterator = SegmentationIterator(
        image, sharded, channel_selection=0, axes_order="yx"
    )
    iterator = iterator.grid(size_x=16, size_y=16)
    assert iterator.check_if_chunks_overlap()

    # The identical layout without sharding is chunk-disjoint and passes.
    unsharded = ome_zarr.derive_label("unsharded", chunks=(1, 16, 16))
    iterator = SegmentationIterator(
        image, unsharded, channel_selection=0, axes_order="yx"
    )
    iterator = iterator.grid(size_x=16, size_y=16)
    assert not iterator.check_if_chunks_overlap()


def test_by_chunks_grid_parameter():
    """by_chunks(grid="write") tiles on the output's write granularity, so the
    resulting ROIs pass the write-side gate; the default grid="read" keeps the
    input tiling this method always had."""
    # Default-derived output: grids match, tiling passes either way
    ome_zarr = _build_ome_zarr(chunks=(1, 16, 16))
    label = ome_zarr.derive_label("default")
    image = ome_zarr.get_image()
    iterator = SegmentationIterator(image, label, channel_selection=0, axes_order="yx")
    assert not iterator.by_chunks().check_if_chunks_overlap()

    # Coarser-chunked output: write tiling passes, the read default flags
    coarse = ome_zarr.derive_label("coarse", chunks=(1, 32, 32))
    iterator = SegmentationIterator(image, coarse, channel_selection=0, axes_order="yx")
    write_tiled = iterator.by_chunks(grid="write")
    assert len(write_tiled.rois) == 4  # 32x32 tiles from the output grid
    assert not write_tiled.check_if_chunks_overlap()
    read_tiled = iterator.by_chunks()
    assert len(read_tiled.rois) == 16  # 16x16 tiles from the input grid
    assert read_tiled.check_if_chunks_overlap()

    # Sharded output: one tile per shard
    ome_zarr_v5 = _build_ome_zarr(chunks=(1, 16, 16), ngff_version="0.5")
    image_v5 = ome_zarr_v5.get_image()
    sharded = ome_zarr_v5.derive_label(
        "sharded", chunks=(1, 16, 16), shards=(1, 64, 64)
    )
    iterator = SegmentationIterator(
        image_v5, sharded, channel_selection=0, axes_order="yx"
    )
    shard_tiled = iterator.by_chunks(grid="write")
    assert len(shard_tiled.rois) == 1  # the single 64x64 shard
    assert not shard_tiled.check_if_chunks_overlap()
    assert iterator.by_chunks(grid="read").check_if_chunks_overlap()

    with pytest.raises(NgioValueError):
        iterator.by_chunks(grid="bogus")  # ty: ignore[invalid-argument-type]

    # Read-only iterators fall back to the input grid even for grid="write"
    feature_iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label,
        channel_selection=0,
        axes_order="yx",
    )
    assert len(feature_iterator.by_chunks(grid="write").rois) == 16


def test_chunk_gate_readonly_returns_false():
    """A read-only iterator has no write hazard: the gate returns False and
    require_no_chunks_overlap does not raise, even for chunk-sharing ROIs."""
    ome_zarr = _build_ome_zarr(chunks=(1, 64, 64))
    label = ome_zarr.derive_label("label")
    image = ome_zarr.get_image()

    iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label,
        channel_selection=0,
        axes_order="yx",
    )
    iterator = iterator.grid(size_x=16, size_y=16)

    assert iterator.check_if_chunks_overlap() is False
    iterator.require_no_chunks_overlap()


def test_by_zyx_strict_raises_ngio_error():
    data = np.zeros((16, 16), dtype="uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=data, pixelsize=1.0, axes_names="yx", levels=1
    )
    image = ome_zarr.get_image()
    label = ome_zarr.derive_label("seg")
    iterator = SegmentationIterator(image, label, axes_order="yx")

    with pytest.raises(NgioValueError, match="must be 3D"):
        iterator.by_zyx(strict=True)


def test_masked_segmentation_checks_dimensions():
    """A label on a different spatial grid is a misconfiguration, not a target."""
    data = np.zeros((2, 32, 32), dtype="uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=data, pixelsize=1.0, axes_names="cyx", levels=1
    )
    masking = ome_zarr.derive_label("masking")
    masking.set_array(np.ones(masking.shape, dtype="uint8"))
    masking.consolidate()
    ome_zarr.add_table("masking_ROI_table", masking.build_masking_roi_table())
    masked = ome_zarr.get_masked_image(masking_label_name="masking")

    mismatched = ome_zarr.derive_label("half_res", shape=(16, 16))

    with pytest.raises(NgioValueError):
        MaskedSegmentationIterator(masked, mismatched, axes_order="yx")


def test_feature_iterator_still_refuses_a_halo():
    """Only iterators that reconcile the margin themselves may read grown."""
    data = np.zeros((16, 16), dtype="uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=data, pixelsize=1.0, axes_names="yx", levels=1
    )
    iterator = FeatureExtractorIterator(
        input_image=ome_zarr.get_image(),
        input_label=ome_zarr.derive_label("lbl"),
        axes_order="yx",
    )
    with pytest.raises(NgioValueError, match="read-only"):
        iterator.with_halo(x=4, y=4)


def test_incomplete_get_init_kwargs_is_refused():
    """Forgotten constructor state must fail loudly, not vanish on reshaping."""
    data = np.zeros((16, 16), dtype="uint8")
    ome_zarr = create_ome_zarr_from_array(
        store=MemoryStore(), array=data, pixelsize=1.0, axes_names="yx", levels=1
    )

    class Forgetful(SegmentationIterator):
        def __init__(self, input_image, output_label, *, favourite=None, **kwargs):
            super().__init__(input_image, output_label, **kwargs)
            self._favourite = favourite

        # get_init_kwargs inherited: 'favourite' is silently missing from it.

    iterator = Forgetful(
        ome_zarr.get_image(), ome_zarr.derive_label("seg"), favourite=42
    )
    with pytest.raises(NgioValueError, match="favourite"):
        iterator.by_yx()
