"""Snippets for docs/tutorials/object_detection.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/object_detection.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, table_html
# --8<-- [end:plot_helpers]


# --8<-- [start:detector]
import math

import numpy as np
from skimage.feature import blob_log

from ngio import Roi


def find_nuclei(patch: np.ndarray) -> list[Roi]:
    """A spot detector: Laplacian-of-Gaussian blobs, as bounding boxes.

    The boxes are `Roi` objects in the patch's own pixel coordinates
    (`space="pixel"`); the iterator anchors them into world coordinates.
    Every extra field — here the peak intensity as a confidence — rides
    along into the table.
    """
    blobs = blob_log(patch, min_sigma=2, max_sigma=6, threshold=0.05)
    boxes = []
    for y, x, sigma in blobs:
        radius = sigma * math.sqrt(2)
        x_min, y_min = max(0.0, x - radius), max(0.0, y - radius)
        boxes.append(
            Roi.from_values(
                slices={
                    "x": (x_min, x + radius - x_min),
                    "y": (y_min, y + radius - y_min),
                },
                name=None,
                space="pixel",
                confidence=float(patch[int(y), int(x)]),
            )
        )
    return boxes


# --8<-- [end:detector]

# --8<-- [start:create]
import skimage

from ngio import create_ome_zarr_from_array

data = skimage.data.human_mitosis().astype("float32") / 255.0
ome_zarr = create_ome_zarr_from_array(
    store="./data/human_mitosis_detection.zarr",
    array=data,
    pixelsize=0.1,  # Just a guess
    consolidation_mode="auto",
    overwrite=True,
)
image = ome_zarr.get_image()
print(ome_zarr)
# --8<-- [end:create]

# --8<-- [start:detect]
from ngio import NmsConfig, ObjectDetectionIterator, ThreadedMapper

iterator = (
    ObjectDetectionIterator(image, nms=NmsConfig(iou_threshold=0.4))
    .by_grid(size_x=128, size_y=128)
    .with_halo(x=16, y=16)
)

detections = iterator.detect(find_nuclei, mapper=ThreadedMapper("auto"))
assert detections is not None  # a serial run always returns the table
ome_zarr.add_table("nuclei_detections", detections, overwrite=True)
print(f"Detected {len(detections.rois())} nuclei")
# --8<-- [end:detect]

# --8<-- [start:read_table_back]
table = ome_zarr.get_table("nuclei_detections")
print(table_html(table.dataframe.head()))
# --8<-- [end:read_table_back]

# --8<-- [start:plot_detections]
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image.get_as_numpy(), cmap="gray")
for roi in table.rois():
    box = roi.to_pixel(pixel_size=image.pixel_size)
    ax.add_patch(
        Rectangle(
            (box["x"].start, box["y"].start),
            box["x"].length,
            box["y"].length,
            fill=False,
            edgecolor="#f4a63a",
            linewidth=0.8,
        )
    )
ax.axis("off")
print(figure_html(fig))
# --8<-- [end:plot_detections]
