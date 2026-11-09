"""A remote `.ozx` served over plain HTTP, read in place."""

from __future__ import annotations

import numpy as np
from utils import random_zarr_path

from ngio import check_ozx, create_empty_ome_zarr, open_ome_zarr_container, to_ozx

SHAPE = (2, 4, 16, 16)


def test_http_ozx(http_static_server: dict, tmp_path) -> None:
    url, root = http_static_server["url"], http_static_server["root"]
    name = random_zarr_path().replace(".zarr", ".ozx")

    img = create_empty_ome_zarr(
        tmp_path / "src.zarr", shape=SHAPE, pixelsize=0.5, ngff_version="0.5"
    )
    data = np.random.randint(0, 255, size=SHAPE, dtype=np.uint8)
    img.get_image().set_array(data)
    to_ozx(tmp_path / "src.zarr", root / name)

    assert check_ozx(f"{url}/{name}").must == []
    remote = open_ome_zarr_container(f"{url}/{name}", mode="r")
    np.testing.assert_array_equal(remote.get_image().get_array(), data)
