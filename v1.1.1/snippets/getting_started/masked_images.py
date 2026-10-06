"""Snippets for docs/getting_started/4_masked_images.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/masked_images.py
"""

# --8<-- [start:plot_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume (`Path("./data")`), so resolve the
# shared module against that.
sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, show_image
# --8<-- [end:plot_helpers]

# --8<-- [start:setup]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download a sample dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the OME-Zarr container
ome_zarr_container = open_ome_zarr_container(image_path)
# --8<-- [end:setup]

# --8<-- [start:get_masked_image]
masked_image = ome_zarr_container.get_masked_image("nuclei")
print(masked_image)
# --8<-- [end:get_masked_image]

# --8<-- [start:masked_roi_numpy]
roi_data = masked_image.get_roi_as_numpy(label=1009, c=0)
print(roi_data.shape)
# --8<-- [end:masked_roi_numpy]

# --8<-- [start:plot_masked_roi]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_image(
    ax,
    masked_image.get_roi_as_numpy(label=1009, c=0),
    title="Label 1009 ROI",
    pixel_size=masked_image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="One nucleus, cropped to the bounding box of its label."))
# --8<-- [end:plot_masked_roi]

# --8<-- [start:masked_roi_zoom]
roi_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
print(roi_data.shape)
# --8<-- [end:masked_roi_zoom]

# --8<-- [start:plot_masked_roi_zoom]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_image(
    ax,
    masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2),
    title="Label 1009 ROI - Zoomed out",
    pixel_size=masked_image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="The same nucleus with twice the surrounding context."))
# --8<-- [end:plot_masked_roi_zoom]

# --8<-- [start:get_roi_masked]
masked_roi_data = masked_image.get_roi_masked_as_numpy(label=1009, c=0, zoom_factor=2)
print(masked_roi_data.shape)
# --8<-- [end:get_roi_masked]

# --8<-- [start:plot_get_roi_masked]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_image(
    ax,
    masked_image.get_roi_masked_as_numpy(label=1009, c=0, zoom_factor=2),
    title="Masked Label 1009 ROI",
    # Everything outside the mask is zero here, and would otherwise take the low end of
    # the window with it, leaving the nucleus washed out.
    ignore_zeros=True,
    pixel_size=masked_image.pixel_size,
)
fig.tight_layout()
print(
    figure_html(fig, alt="The same nucleus with every pixel outside its mask zeroed.")
)
# --8<-- [end:plot_get_roi_masked]

# --8<-- [start:set_roi_masked]
import numpy as np

masked_data = masked_image.get_roi_masked_as_numpy(label=1009, c=0)
masked_data = np.random.randint(0, 255, masked_data.shape, dtype=np.uint8)
masked_image.set_roi_masked(label=1009, c=0, patch=masked_data)
# --8<-- [end:set_roi_masked]

# --8<-- [start:plot_after_set_roi_masked]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_image(
    ax,
    masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2),
    title="Masked Label 1009 ROI - After setting",
    pixel_size=masked_image.pixel_size,
)
fig.tight_layout()
print(
    figure_html(
        fig, alt="The nucleus replaced by random values, its surroundings intact."
    )
)
# --8<-- [end:plot_after_set_roi_masked]

# --8<-- [start:get_masked_label]
masked_label = ome_zarr_container.get_masked_label(
    label_name="wf_2_labels", masking_label_name="nuclei"
)
print(masked_label)
# --8<-- [end:get_masked_label]
