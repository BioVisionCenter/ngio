"""Chunk-aligned versus straddling ROI reads: read amplification.

A read whose bounds fall inside chunk boundaries touches the minimum number
of chunks. Shift it by a few pixels and every chunk on each edge is fetched
and discarded. The ratio between the two is the cost of not aligning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks._fixtures import image_fixture
from benchmarks._measure import Measured
from ngio import open_image

if TYPE_CHECKING:
    from pathlib import Path

#: Chunked at 512 so `aligned` really is aligned and `straddling` really is not.
_CHUNKS = (1, 1, 512, 512)

AXES = {
    # Closed: the labels are the point, and the offsets are only meaningful
    # relative to `_CHUNKS`.
    "alignment": {"aligned": 0, "straddling": 137},
    # Open, so read amplification can be traced against the ROI size --
    # it should shrink as the ROI grows and the wasted edge chunks amortise.
    "size": [512],
}


def run(root: Path, *, alignment: int, size: int) -> Measured:
    """Read one ROI out of a chunked image."""
    # Kept out of module scope: `ngio.common` is not part of the public API
    # surface these blocks promise to hold to, so a move must degrade this
    # block rather than break discovery for every other one.
    from ngio.common import Roi, RoiSlice

    path = image_fixture(root, (2, 16, 1024, 1024), chunks=_CHUNKS)
    image = open_image(path, path="0", mode="r")
    roi = Roi(
        name="probe",
        space="pixel",
        slices=[
            RoiSlice(axis_name="y", start=alignment, length=size),
            RoiSlice(axis_name="x", start=alignment, length=size),
        ],
    )
    return Measured(lambda: image.get_roi_as_numpy(roi))
