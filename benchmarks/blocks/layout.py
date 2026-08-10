"""Same bytes, different on-disk layout: what does the layout cost to read?

`get`/`set` are deliberately absent as blocks of their own -- they are a thin
layer over zarr, adding a roughly constant ~0.8 ms, so timing them mostly
re-measures zarr. What is worth measuring is the layout underneath them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks._fixtures import image_fixture
from benchmarks._measure import Measured
from ngio import open_image

if TYPE_CHECKING:
    from pathlib import Path

AXES = {
    # A *closed* axis: each value is a bundle of `create_empty_ome_zarr`
    # kwargs, so the labels are the vocabulary. `--axis layout=sharded`
    # subsets these but cannot invent a new one, which is right -- chunk and
    # shard shapes are not independent knobs (`sharded` + `uncompressed` is
    # not a case anyone wants), and a shard shape is not something to type at
    # a shell prompt.
    "layout": {
        "chunks512": {"chunks": (1, 1, 512, 512)},
        "chunks128": {"chunks": (1, 1, 128, 128)},
        "sharded": {"chunks": (1, 1, 128, 128), "shards": (1, 4, 1024, 1024)},
        "uncompressed": {"chunks": (1, 1, 512, 512), "compressors": None},
    },
    # Open, so `--axis layout.z=16,64,128` crosses depth with layout for free.
    # The old hardcoded `cases` dict could not express that product at all.
    "z": [16],
}

REPEATS = 3


def run(root: Path, *, layout: dict[str, Any], z: int) -> Measured:
    """Read a whole level back as numpy."""
    path = image_fixture(root, (2, z, 1024, 1024), **layout)
    return Measured(open_image(path, path="0", mode="r").get_as_numpy)
