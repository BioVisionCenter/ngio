from pathlib import Path

import numpy as np
import pytest

from ngio import (
    create_empty_ome_zarr,
    create_ome_zarr_from_array,
    create_synthetic_ome_zarr,
)
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

    with pytest.raises(NgioValueError):
        create_ome_zarr_from_array(
            array=np.random.randint(0, 255, (2, 64, 64), dtype="uint8"),
            store=tmp_path / "fail.zarr",
            xy_pixelsize=0.5,
            axes_names=["c", "y", "x"],
            levels=1,
            chunks=(1, 64, 64, 64),  # should fail expected 3 axes
            overwrite=True,
        )
