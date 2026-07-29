"""Snippets for docs/tutorials/image_segmentation.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/tutorials/image_segmentation.py
"""

# --8<-- [start:plot_helpers]
from io import StringIO

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

matplotlib.use("Agg")


def random_label_cmap(n_labels: int = 1000, seed: int = 0) -> ListedColormap:
    """Build a reproducible random colormap for label images."""
    rng = np.random.default_rng(seed)
    colors = rng.random((n_labels, 3))
    colors[0] = 0.0
    return ListedColormap(colors)


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

# --8<-- [start:segmentation_fn]
# Setup a simple segmentation function
import numpy as np
import skimage


def otsu_threshold_segmentation(image: np.ndarray, max_label: int) -> np.ndarray:
    """Simple segmentation using Otsu thresholding."""
    threshold = skimage.filters.threshold_otsu(image)
    binary = image > threshold
    label_image = skimage.measure.label(binary)
    label_image += max_label
    label_image = np.where(binary, label_image, 0)
    return label_image.astype(np.uint32)


# --8<-- [end:segmentation_fn]

# --8<-- [start:open_container]
from pathlib import Path

from ngio import open_ome_zarr_container
from ngio.utils import download_ome_zarr_dataset

# Download the dataset
download_dir = Path("./data").absolute()
hcs_path = download_ome_zarr_dataset("CardiomyocyteTiny", download_dir=download_dir)
image_path = hcs_path / "B" / "03" / "0"

# Open the ome-zarr container
ome_zarr = open_ome_zarr_container(image_path)
# --8<-- [end:open_container]

# --8<-- [start:segment]
from ngio.experimental.iterators import SegmentationIterator

# First we will need the image object and the FOVs table
image = ome_zarr.get_image()
roi_table = ome_zarr.get_roi_table("FOV_ROI_table")

# Second we need to derive a new label image to use as target for the segmentation

label = ome_zarr.derive_label("new_label", overwrite=True)

# Setup the segmentation iterator
seg_iterator = SegmentationIterator(
    input_image=image,
    output_label=label,
    channel_selection="DAPI",
    axes_order=["z", "y", "x"],
)
seg_iterator = seg_iterator.product(roi_table)

# Make sure that if other axes are present they are iterated over
seg_iterator = seg_iterator.by_zyx()

max_label = 0  # We will use this to avoid label collisions
for image_data, label_writer in seg_iterator.iter_as_numpy():
    roi_segmentation = otsu_threshold_segmentation(
        image_data, max_label
    )  # Segment the image

    max_label = roi_segmentation.max()  # Get the max label for the next iteration

    label_writer(patch=roi_segmentation)  # Write the segmentation back to the label

# No need to consolidate, the iterator does it automatically after the last write
# --8<-- [end:segment]

# --8<-- [start:plot_segmentation]
rand_cmap = random_label_cmap()

fig, axs = plt.subplots(2, 1, figsize=(8, 4))
axs[0].set_title("Original image")
axs[0].imshow(image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]), cmap="gray")
axs[1].set_title("Final segmentation")
axs[1].imshow(label.get_as_numpy(z=1, axes_order=["y", "x"]), cmap=rand_cmap)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_segmentation]

# --8<-- [start:create_mask]
# Create a basic mask for illustration purposes
mask = ome_zarr.derive_label("mask", overwrite=True)
mask_data = mask.get_as_numpy(axes_order=["z", "y", "x"])
mask_data[:, 200:-200, 500:2000] = 1
mask_data[:, 200:-200, 3000:-500] = 2
mask_data[:, 600:-600, 1200:-1000] = 0
mask_data[:, 700:-700, 1600:-1500] = 3
mask.set_array(mask_data, axes_order=["z", "y", "x"])
mask.consolidate()
# --8<-- [end:create_mask]

# --8<-- [start:plot_mask]
fig, axs = plt.subplots(2, 1, figsize=(8, 4))
axs[0].set_title("Original image")
axs[0].imshow(image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]), cmap="gray")
axs[1].set_title("Mask")
axs[1].imshow(mask.get_as_numpy(z=1, axes_order=["y", "x"]), cmap=rand_cmap)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_mask]

# --8<-- [start:masked_segment]
from ngio.experimental.iterators import MaskedSegmentationIterator

# First we will need the masked image object
# (that contains the masking table information inside)
image = ome_zarr.get_masked_image(masking_label_name="mask")

# Second we need to derive a new label image to use as target for the segmentation
label = ome_zarr.derive_label("masked_new_label", overwrite=True)

# Setup the masked segmentation iterator
seg_iterator = MaskedSegmentationIterator(
    input_image=image,
    output_label=label,
    channel_selection="DAPI",
    axes_order=["z", "y", "x"],
)

# Make sure that if other axes are present they are iterated over
seg_iterator = seg_iterator.by_zyx()

max_label = 0  # We will use this to avoid label collisions
for image_data, label_writer in seg_iterator.iter_as_numpy():
    roi_segmentation = otsu_threshold_segmentation(
        image_data, max_label
    )  # Segment the image

    max_label = roi_segmentation.max()  # Get the max label for the next iteration

    label_writer(patch=roi_segmentation)  # Write the segmentation back to the label

# No need to consolidate, the iterator does it automatically after the last write
# --8<-- [end:masked_segment]

# --8<-- [start:plot_masked_segmentation]
fig, axs = plt.subplots(2, 1, figsize=(8, 4))
axs[0].set_title("Original image")
axs[0].imshow(image.get_as_numpy(c=0, z=1, axes_order=["y", "x"]), cmap="gray")
axs[1].set_title("Final segmentation")
axs[1].imshow(label.get_as_numpy(z=1, axes_order=["y", "x"]), cmap=rand_cmap)
for ax in axs:
    ax.axis("off")
fig.tight_layout()
print_figure(fig)
# --8<-- [end:plot_masked_segmentation]
