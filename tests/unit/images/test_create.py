from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from zarr.storage import MemoryStore

from ngio import (
    OmeZarrContainer,
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    create_synthetic_ome_zarr,
)
from ngio.images._create_utils import init_image_like_from_shapes
from ngio.ome_zarr_meta import NgioImageMeta
from ngio.utils import NgioValueError


@pytest.mark.parametrize(
    "create_kwargs",
    [
        {
            "store": "test_image_yx._zarr",
            "shape": (64, 64),
            "xy_pixelsize": 0.5,
            "axes_names": ["y", "x"],
        },
        {
            "store": "test_image_cyx.zarr",
            "shape": (2, 64, 64),
            "xy_pixelsize": 0.5,
            "axes_names": ["c", "y", "x"],
            "channel_labels": ["channel1", "channel2"],
        },
        {
            "store": "test_image_zyx.zarr",
            "shape": (3, 64, 64),
            "xy_pixelsize": 0.5,
            "z_spacing": 2.0,
            "axes_names": ["z", "y", "x"],
        },
        {
            "store": "test_image_czyx.zarr",
            "shape": (2, 3, 64, 64),
            "xy_pixelsize": 0.5,
            "z_spacing": 2.0,
            "axes_names": ["c", "z", "y", "x"],
            "channel_labels": ["channel1", "channel2"],
        },
        {
            "store": "test_image_c1yx.zarr",
            "shape": (2, 1, 64, 64),
            "xy_pixelsize": 0.5,
            "z_spacing": 1.0,
            "axes_names": ["c", "z", "y", "x"],
            "channel_labels": ["channel1", "channel2"],
        },
        {
            "store": "test_image_tyx.zarr",
            "shape": (4, 64, 64),
            "xy_pixelsize": 0.5,
            "time_spacing": 4.0,
            "axes_names": ["t", "y", "x"],
        },
        {
            "store": "test_image_tcyx.zarr",
            "shape": (4, 2, 64, 64),
            "xy_pixelsize": 0.5,
            "time_spacing": 4.0,
            "axes_names": ["t", "c", "y", "x"],
            "channel_labels": ["channel1", "channel2"],
        },
        {
            "store": "test_image_tzyx.zarr",
            "shape": (4, 3, 64, 64),
            "xy_pixelsize": 0.5,
            "z_spacing": 2.0,
            "time_spacing": 4.0,
            "axes_names": ["t", "z", "y", "x"],
        },
        {
            "store": "test_image_tczyx.zarr",
            "shape": (4, 2, 3, 64, 64),
            "xy_pixelsize": 0.5,
            "z_spacing": 2.0,
            "time_spacing": 4.0,
            "axes_names": ["t", "c", "z", "y", "x"],
            "channel_labels": ["channel1", "channel2"],
        },
    ],
)
def test_create_empty(tmp_path: Path, create_kwargs: dict):
    create_kwargs["store"] = tmp_path / create_kwargs["store"]
    ome_zarr = create_empty_ome_zarr(**create_kwargs, dtype="uint8", levels=1)
    ome_zarr.derive_label("label1")

    shape = create_kwargs.pop("shape")
    array = np.random.randint(0, 255, shape, dtype="uint8")
    create_ome_zarr_from_array(array=array, **create_kwargs, levels=1, overwrite=True)

    store = create_kwargs["store"]
    create_synthetic_ome_zarr(store=store, shape=shape, overwrite=True)


def test_large_synt(tmp_path: Path):
    store = tmp_path / "test_image_large.zarr"
    create_synthetic_ome_zarr(store=store, shape=(1, 1, 513, 513), overwrite=True)


def test_create_fail(tmp_path: Path):
    with pytest.raises(NgioValueError):
        create_ome_zarr_from_array(
            array=np.random.randint(0, 255, (64, 64), dtype="uint8"),
            store=tmp_path / "fail.zarr",
            xy_pixelsize=0.5,
            axes_names=["z", "y", "x"],  # should fail expected yx
            levels=1,
            overwrite=True,
        )

    with pytest.raises(NgioValueError):
        create_ome_zarr_from_array(
            array=np.random.randint(0, 255, (2, 64, 64), dtype="uint8"),
            store=tmp_path / "fail.zarr",
            xy_pixelsize=0.5,
            axes_names=["c", "y", "x"],
            levels=1,
            channel_labels=[
                "channel1",
                "channel2",
                "channel3",
            ],  # should fail expected 2 channels
            overwrite=True,
        )

    with pytest.raises(ValidationError):
        create_ome_zarr_from_array(
            array=np.random.randint(0, 255, (2, 64, 64), dtype="uint8"),
            store=tmp_path / "fail.zarr",
            xy_pixelsize=0.5,
            axes_names=["c", "y", "x"],
            levels=1,
            chunks=(1, 64, 64, 64),  # should fail expected 3 axes
            overwrite=True,
        )


def test_derive_label_channels_policy():
    store = MemoryStore()
    ome_zarr = create_synthetic_ome_zarr(store, shape=(3, 1, 64, 65))

    label = ome_zarr.derive_label("test-label-singleton", channels_policy="singleton")
    assert label.dimensions.get("c") == 1
    label = ome_zarr.derive_label("test-label-same", channels_policy="same")
    assert label.dimensions.get("c") == 3
    label = ome_zarr.derive_label("test-label-squeeze", channels_policy="squeeze")
    assert "c" not in label.axes
    assert label.dimensions.get("c") is None
    label = ome_zarr.derive_label("test-label-int", channels_policy=2)
    assert label.dimensions.get("c") == 2


def test_derive_from_non_dishogeneus_shapes():
    # Yes those shapes are intentionally weird
    shapes = [
        (4, 3, 64, 65),
        (4, 3, 64, 50),
        (4, 3, 32, 25),
    ]
    store = MemoryStore()
    image_handler = init_image_like_from_shapes(
        store=store,
        meta_type=NgioImageMeta,
        shapes=shapes,
        pixelsize=0.5,
    )
    ome_zarr = OmeZarrContainer(group_handler=image_handler)
    ome_zarr.derive_label("test-label-same", channels_policy="same")
    for path in ome_zarr.levels_paths:
        img = ome_zarr.get_image(path=path)
        lbl = ome_zarr.get_label(name="test-label-same", path=path)
        assert img.shape == lbl.shape

    image = ome_zarr.get_image(path="1")
    ome_zarr.derive_label("test-label-level-1", ref_image=image, channels_policy="same")

    for path_img, path_lbl in zip(["1", "2"], ["0", "1"], strict=True):
        img = ome_zarr.get_image(path=path_img)
        lbl = ome_zarr.get_label(name="test-label-level-1", path=path_lbl)
        assert img.shape == lbl.shape

    lbl = ome_zarr.get_label(name="test-label-level-1", path="2")
    scaling_factor = tuple(s1 / s2 for s1, s2 in zip(shapes[0], shapes[1], strict=True))
    assert image.meta.scaling_factor() == scaling_factor
    assert lbl.shape == (4, 3, 32, 19)


def test_create_with_sharding(tmp_path: Path):
    store = tmp_path / "test_image_sharded.zarr"
    ome_zarr = create_empty_ome_zarr(
        store=store,
        shape=(4, 3, 64, 64),
        pixelsize=0.5,
        chunks=(2, 1, 32, 32),
        shards=(4, 3, 64, 64),
        dtype="uint8",
        levels=3,
        overwrite=True,
        ngff_version="0.5",
    )
    ome_zarr.derive_label("label_sharded")
    img = ome_zarr.get_image(path="0")
    assert img.zarr_array.shards is not None
    assert img.zarr_array.chunks == (2, 1, 32, 32)
    assert img.zarr_array.shards == (4, 3, 64, 64)

    img = ome_zarr.get_image(path="2")
    assert img.zarr_array.shards is not None
    # Check clipping of chunks/shards at the smallest level
    assert img.zarr_array.chunks == (2, 1, 16, 16)
    assert img.zarr_array.shards == (4, 3, 16, 16)


def test_fail_derive_singleton():
    store = MemoryStore()
    ome_zarr = create_empty_ome_zarr(store=store, shape=(1, 1, 64, 4), pixelsize=0.5)
    expected_shapes = [
        (1, 1, 64, 4),
        (1, 1, 32, 2),
        (1, 1, 16, 1),
        (1, 1, 8, 1),
        (1, 1, 4, 1),
    ]
    expected_pixel_size_x = [0.5, 1.0, 2.0, 2.0, 2.0]
    for path, shape, px_x in zip(
        ome_zarr.levels_paths, expected_shapes, expected_pixel_size_x, strict=True
    ):
        img = ome_zarr.get_image(path=path)
        assert img.shape == shape
        assert img.pixel_size.x == px_x


def test_fail_create_from_non_decreasing_shapes():
    # Yes those shapes are intentionally weird
    shapes = [
        (4, 3, 64, 64),
        (4, 3, 128, 128),
        (4, 3, 256, 256),
    ]
    store = MemoryStore()

    with pytest.raises(NgioValueError):
        _ = init_image_like_from_shapes(
            store=store,
            meta_type=NgioImageMeta,
            shapes=shapes,
            pixelsize=0.5,
        )
