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
from ngio.iterators import GreedyNms, ObjectDetectionIterator, ThreadedMapper

iterator = (
    ObjectDetectionIterator(image)
    .with_nms(GreedyNms(iou_threshold=0.4))
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
print(figure_html(fig, alt="Every detected nucleus outlined on the full image."))
# --8<-- [end:plot_detections]

# --8<-- [start:nms_raw]
# Iterate the same (haloed) tiles yourself — the view `detect` sees before
# suppression.
raw_boxes = []
for patch, tile in iterator.iter_as_numpy():
    for box in find_nuclei(patch):
        raw_boxes.append(tile.anchor(box, pixel_size=image.pixel_size))

print(f"raw boxes: {len(raw_boxes)}, after NMS: {len(detections.rois())}")
# --8<-- [end:nms_raw]


# --8<-- [start:nms_figure]
def draw_boxes(ax, boxes, title):
    ax.imshow(image.get_as_numpy(), cmap="gray")
    for roi in boxes:
        box = roi.to_pixel(pixel_size=image.pixel_size)
        ax.add_patch(
            Rectangle(
                (box["x"].start, box["y"].start),
                box["x"].length,
                box["y"].length,
                fill=False,
                edgecolor="#f4a63a",
                linewidth=1.0,
            )
        )
    for edge in (256, 384):  # the tile boundaries crossing this zoom
        ax.axvline(edge - 0.5, color="white", ls="--", lw=0.8)
        ax.axhline(edge - 0.5, color="white", ls="--", lw=0.8)
    ax.set_xlim(191.5, 319.5)
    ax.set_ylim(447.5, 319.5)  # imshow's y axis grows downwards
    ax.set_title(title)
    ax.axis("off")


fig, (ax_raw, ax_kept) = plt.subplots(1, 2, figsize=(8.6, 4.4))
draw_boxes(ax_raw, raw_boxes, f"raw: {len(raw_boxes)} boxes")
draw_boxes(ax_kept, detections.rois(), f"after NMS: {len(detections.rois())} boxes")
print(
    figure_html(
        fig,
        alt="A zoom onto two tile boundaries: before suppression several "
        "nuclei near the seams carry two overlapping boxes, one from each "
        "neighbouring tile; after NMS each carries exactly one.",
    )
)
# --8<-- [end:nms_figure]

# --8<-- [start:anchor_demo]
tile_roi = iterator.rois[0]
box = Roi.from_values(slices={"x": (12, 30), "y": (4, 25)}, name=None, space="pixel")
abs_roi = tile_roi.anchor(box, pixel_size=image.pixel_size)
print(abs_roi)
# --8<-- [end:anchor_demo]
