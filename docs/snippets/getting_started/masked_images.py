"""Snippets for docs/getting_started/4_masked_images.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/masked_images.py
"""

# --8<-- [start:plot_helpers]
from io import StringIO

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")


def print_figure(fig: Figure) -> None:
    """Print a figure as inline SVG, for markdown-exec `html` blocks.

    Recolours every bit of chrome to one sentinel, then swaps that sentinel for
    a theme variable in the emitted markup. The figure therefore follows the
    light/dark toggle rather than baking black text on white into the page.
    This only works because the SVG is inline: an `<img src>` would be a
    separate document and would not see the site's custom properties.

    The `.ngio-figure` wrapper is what the stylesheet keys on to strip the
    `OUT` terminal-output treatment that `.result` applies by default.
    """
    ink = "#5b6569"
    for ax in fig.axes:
        ax.tick_params(colors=ink, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(ink)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(ink)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(ink)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color(ink)
    for text in fig.texts:
        text.set_color(ink)

    buffer = StringIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)

    # Drop matplotlib's XML declaration and DOCTYPE — they are meaningless
    # inside an HTML body, and the figure is being embedded, not served.
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = svg.replace(ink, "var(--md-default-fg-color--light)")
    print(f'<div class="ngio-figure">{svg}</div>')


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
import numpy as np

image_data = masked_image.get_roi_as_numpy(label=1009, c=0)
image_data = np.squeeze(image_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Label 1009 ROI")
ax.imshow(image_data, cmap="gray")
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_masked_roi]

# --8<-- [start:masked_roi_zoom]
roi_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
print(roi_data.shape)
# --8<-- [end:masked_roi_zoom]

# --8<-- [start:plot_masked_roi_zoom]
image_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
image_data = np.squeeze(image_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Label 1009 ROI - Zoomed out")
ax.imshow(image_data, cmap="gray")
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_masked_roi_zoom]

# --8<-- [start:get_roi_masked]
masked_roi_data = masked_image.get_roi_masked(label=1009, c=0, zoom_factor=2)
print(masked_roi_data.shape)
# --8<-- [end:get_roi_masked]

# --8<-- [start:plot_get_roi_masked]
masked_roi_data = masked_image.get_roi_masked(label=1009, c=0, zoom_factor=2)
masked_roi_data = np.squeeze(masked_roi_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Masked Label 1009 ROI")
ax.imshow(masked_roi_data, cmap="gray")
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_get_roi_masked]

# --8<-- [start:set_roi_masked]
import numpy as np

masked_data = masked_image.get_roi_masked(label=1009, c=0)
masked_data = np.random.randint(0, 255, masked_data.shape, dtype=np.uint8)
masked_image.set_roi_masked(label=1009, c=0, patch=masked_data)
# --8<-- [end:set_roi_masked]

# --8<-- [start:plot_after_set_roi_masked]
masked_data = masked_image.get_roi_as_numpy(label=1009, c=0, zoom_factor=2)
masked_data = np.squeeze(masked_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Masked Label 1009 ROI - After setting")
ax.imshow(masked_data, cmap="gray")
ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_after_set_roi_masked]

# --8<-- [start:get_masked_label]
masked_label = ome_zarr_container.get_masked_label(
    label_name="wf_2_labels", masking_label_name="nuclei"
)
print(masked_label)
# --8<-- [end:get_masked_label]
