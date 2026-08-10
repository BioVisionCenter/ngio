"""Synthetic data and reusable on-disk fixtures.

Everything here goes through ngio's **public API only**. That is what lets a
block run unmodified inside an environment holding a different ngio (see
`--env`): a private module path that moved would otherwise break the import
and take the fixture down with it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from ngio import create_empty_ome_zarr

if TYPE_CHECKING:
    from pathlib import Path

#: Tuned against the real bundled sample image, which compresses 1.74x and
#: holds 221 distinct values across the full uint16 range. These settings give
#: ~1.84x, i.e. the same regime.
_QUANT_LEVELS = 128
_SMOOTHING = 16


def synthetic_data(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """A seeded, spatially correlated uint16 volume.

    Compressibility is the point, not realism of appearance. Uniform data
    (`np.ones`) compresses ~2000:1 and pure noise 1:1; either would make the
    `layout` block's comparison of chunk shapes and codecs meaningless without
    looking obviously wrong. Real microscopy sits near 1.7x because it is
    spatially correlated and quantised to relatively few levels, so this
    upsamples coarse noise for structure and then quantises to match.
    """
    from scipy.ndimage import zoom  # scipy is an ngio dependency

    rng = np.random.default_rng(seed)
    coarse = rng.random(size=tuple(max(s // _SMOOTHING, 1) for s in shape[-2:]))
    plane = zoom(coarse.astype(np.float32), _SMOOTHING, order=1)
    plane = plane[: shape[-2], : shape[-1]]
    if plane.shape != tuple(shape[-2:]):  # zoom can land a pixel short
        plane = np.pad(
            plane,
            [(0, t - s) for s, t in zip(plane.shape, shape[-2:], strict=True)],
            mode="edge",
        )
    plane = np.clip(plane + rng.normal(0, 0.02, plane.shape).astype(np.float32), 0, 1)
    quantised = (plane * (_QUANT_LEVELS - 1)).round() * (65535 // (_QUANT_LEVELS - 1))
    return np.broadcast_to(quantised, shape).astype(np.uint16)


def image_fixture(
    root: Path,
    shape: tuple[int, ...],
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
    compressors: Any = "auto",
) -> Path:
    """Create (or reuse) an image fixture and return its path.

    The name is derived from the spec, not passed in, so two blocks asking for
    the same image share one store under `--keep` instead of each building its
    own. It also means a changed spec lands on a new path rather than silently
    reusing a stale store, which the old name-per-caller form got wrong.
    """
    spec = (shape, chunks, shards, repr(compressors))
    digest = hashlib.blake2s(repr(spec).encode(), digest_size=4).hexdigest()
    path = root / f"img_{'x'.join(str(s) for s in shape)}_{digest}.zarr"
    if path.exists():
        return path
    container = create_empty_ome_zarr(
        store=path,
        shape=shape,
        axes_names=["c", "z", "y", "x"],
        channels_meta=["Channel 1", "Channel 2"][: shape[0]],
        levels=3,
        pixelsize=(0.65, 0.65),
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        # NGFF 0.4 maps to zarr format 2, which cannot shard.
        ngff_version="0.5",
        overwrite=True,
    )
    container.get_image(path="0").set_array(patch=synthetic_data(shape))
    return path


def segmentation(n_labels: int, size: int = 512) -> np.ndarray:
    """A label image holding `n_labels` square, non-touching labels."""
    seg = np.zeros((size, size), dtype=np.uint16)
    side = int(np.ceil(np.sqrt(n_labels)))
    step = max(size // side, 2)
    label = 0
    for r in range(side):
        for c in range(side):
            label += 1
            if label > n_labels:
                return seg
            seg[r * step : r * step + step - 1, c * step : c * step + step - 1] = label
    return seg


def roi_frame(n: int):
    """A dataframe in the shape a v1 ROI table backend hands to ngio."""
    import pandas as pd

    return pd.DataFrame(
        {
            "FieldIndex": [f"roi_{i}" for i in range(n)],
            "x_micrometer": [float(i) for i in range(n)],
            "y_micrometer": [float(i) for i in range(n)],
            "z_micrometer": [0.0] * n,
            "len_x_micrometer": [10.0] * n,
            "len_y_micrometer": [10.0] * n,
            "len_z_micrometer": [1.0] * n,
        }
    ).set_index("FieldIndex")
