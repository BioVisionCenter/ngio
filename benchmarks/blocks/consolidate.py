"""Pyramid consolidation: which mode should I use, and will it fit?

`numpy` materialises a whole level (plus the zoom output twice, via
`_stacked_zoom`), so its peak tracks the data and it has a hard ceiling.
`dask` and `coarsen` are chunk-bounded and stay roughly flat as the data
grows. That difference is invisible in a timing -- `numpy` is also the
fastest -- which is why peak memory is a first-class column here and not an
extra.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks._fixtures import synthetic_data
from benchmarks._measure import MB, Measured
from ngio import create_empty_ome_zarr

if TYPE_CHECKING:
    from pathlib import Path

AXES = {
    "mode": ["dask", "numpy", "coarsen"],
    # Open, so `--axis consolidate.z=1024` sweeps past what is declared here.
    # z * 512 * 512 * 2 bytes: 16 -> 8 MB, 64 -> 32 MB, 256 -> 128 MB.
    "z": [16, 64, 256],
}

REPEATS = 3


def run(root: Path, *, mode: str, z: int) -> Measured:
    """Build a fresh pyramid; measure only its consolidation."""
    shape = (1, z, 512, 512)
    target = root / f"consolidate_{mode}_{z}.zarr"
    if target.exists():
        shutil.rmtree(target)
    container = create_empty_ome_zarr(
        store=target,
        shape=shape,
        axes_names=["c", "z", "y", "x"],
        channels_meta=["Channel 1"],
        levels=3,
        pixelsize=(0.65, 0.65),
        chunks=(1, 1, 256, 256),
        overwrite=True,
    )
    image = container.get_image(path="0")
    # Realistic data, not `np.ones`: consolidation writes every level, and
    # uniform data compresses ~2000:1, which would make the write half of the
    # work almost free and flatter all three modes equally but unrealistically.
    image.set_array(patch=synthetic_data(shape))
    # `mode` is an *open* axis, so `--axis mode=whatever` reaches this as a
    # plain str. Validating it here would only duplicate the check ngio
    # already makes, and worse, would go stale when ngio grows a fourth mode.
    return Measured(
        lambda: image.consolidate(mode=cast("Any", mode)),
        f"{int(np.prod(shape)) * 2 / MB:.0f} MB",
    )
