"""Scaling of ngio's own algorithms.

Reported as a series so the *shape* is visible: one timing cannot tell O(n)
from O(n^2), three can.

This is the one block that must reach into private modules -- it measures
internal algorithms, which have no public entry point. Those imports stay
inside `run` so that under `--env`, a version where the paths differ degrades
this block to a note rather than failing the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarks._fixtures import roi_frame, segmentation
from benchmarks._measure import Measured, Skip

if TYPE_CHECKING:
    from pathlib import Path

AXES = {
    "kernel": ["overlap", "masking_roi", "roi_table"],
    "n": [64, 256, 1024, 2048, 4096, 10_000, 50_000],
}

REPEATS = 3

#: Per-kernel bounds on `n`. The product is rectangular and these kernels are
#: not: `overlap` is O(n^2) and would not return at the sizes `roi_table`
#: needs, while `masking_roi` allocates a label image per label and saturates
#: a 512x512 frame well before then.
_RANGE = {
    "overlap": (64, 2048),
    "masking_roi": (256, 4096),
    "roi_table": (1000, 50_000),
}

#: Attached to every `overlap` row: the reason the block exists.
_NOTES = {"overlap": "O(n^2)"}


def run(root: Path, *, kernel: str, n: int) -> Measured:
    """Build the input for one kernel; measure only the kernel."""
    low, high = _RANGE[kernel]
    if not low <= n <= high:
        raise Skip(f"{kernel} is not measured at n={n}")

    if kernel == "overlap":
        from ngio.io_pipes._ops_slices_utils import check_if_regions_overlap

        side = int(np.ceil(np.sqrt(n)))
        regions = [
            (slice(r * 10, r * 10 + 8), slice(c * 10, c * 10 + 8))
            for r in range(side)
            for c in range(side)
        ][:n]
        return Measured(lambda: check_if_regions_overlap(regions), _NOTES[kernel])

    if kernel == "masking_roi":
        from ngio.common._masking_roi import compute_masking_roi
        from ngio.ome_zarr_meta.ngio_specs import PixelSize

        seg = segmentation(n)
        pixel_size = PixelSize(x=0.65, y=0.65, z=1.0)
        return Measured(lambda: compute_masking_roi(seg, pixel_size, ("y", "x")))

    from ngio.tables.v1._roi_table import _dataframe_to_rois

    frame = roi_frame(n)
    return Measured(lambda: _dataframe_to_rois(frame))
